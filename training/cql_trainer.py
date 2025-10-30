"""
Conservative Q-Learning (CQL) trainer for offline RL.

CQL is a value-based offline RL algorithm that learns Q-functions conservatively
to prevent distribution shift when deploying policies trained on offline data.
"""

import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.q_network import QNetwork
from models.config import create_default_network_config
from data.offline_dataset import Episode, OfflineRLDataset
from environment.utils import get_device

logger = logging.getLogger(__name__)


@dataclass
class CQLConfig:
    """Configuration for CQL training."""
    
    # Model
    max_aircraft: int = 20
    
    # Training
    num_epochs: int = 100
    batch_size: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 10.0
    
    # CQL-specific
    cql_alpha: float = 5.0  # Conservative penalty weight
    tau: float = 0.005  # Soft target update coefficient
    gamma: float = 0.99  # Discount factor
    target_update_freq: int = 1  # Update target network every N epochs
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints/cql"
    save_every: int = 10
    
    # Device
    device: str = field(default_factory=lambda: get_device())


class CQLTrainer:
    """
    Conservative Q-Learning trainer for offline RL.
    
    CQL learns Q-functions that are conservative with respect to the dataset,
    preventing overestimation of values for out-of-distribution actions.
    
    Example:
        >>> episodes = OfflineDatasetCollector.load_episodes("data.pkl")
        >>> trainer = CQLTrainer(CQLConfig())
        >>> trainer.train(episodes)
    """
    
    def __init__(self, config: CQLConfig):
        """
        Initialize CQL trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config
        
        # Create Q-networks (Q and target Q)
        network_config = create_default_network_config(max_aircraft=config.max_aircraft)
        
        self.q_network = QNetwork(network_config).to(config.device)
        self.target_q_network = QNetwork(network_config).to(config.device)
        
        # Initialize target network with same weights
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.target_q_network.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # Loss function
        self.mse_loss = nn.MSELoss()
        
        # Create checkpoint directory
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        # Training history
        self.train_losses = []
        self.cql_losses = []
        self.td_losses = []
    
    def train(self, episodes: List[Episode]) -> Dict[str, List[float]]:
        """
        Train CQL on offline dataset.
        
        Args:
            episodes: List of offline episodes
            
        Returns:
            Dictionary with training history
        """
        # Create dataset (for simplicity, we'll create a simple dataset from transitions)
        transitions = self._episodes_to_transitions(episodes)
        dataset = self._create_dataset(transitions)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
        )
        
        logger.info(f"Training CQL on {len(transitions)} transitions")
        
        for epoch in range(self.config.num_epochs):
            epoch_losses = self._train_epoch(dataloader)
            
            self.train_losses.append(epoch_losses["total"])
            self.td_losses.append(epoch_losses["td"])
            self.cql_losses.append(epoch_losses["cql"])
            
            logger.info(
                f"Epoch {epoch+1}/{self.config.num_epochs}: "
                f"loss={epoch_losses['total']:.4f}, "
                f"td={epoch_losses['td']:.4f}, "
                f"cql={epoch_losses['cql']:.4f}"
            )
            
            # Update target network
            if (epoch + 1) % self.config.target_update_freq == 0:
                self._update_target_network()
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint(epoch + 1)
        
        return {
            "train_losses": self.train_losses,
            "td_losses": self.td_losses,
            "cql_losses": self.cql_losses,
        }
    
    def _episodes_to_transitions(self, episodes: List[Episode]) -> List[Dict]:
        """Convert episodes to (s, a, r, s', done) transitions."""
        transitions = []
        
        for episode in episodes:
            for i in range(len(episode.observations) - 1):
                transition = {
                    "state": episode.observations[i],
                    "action": episode.actions[i],
                    "reward": episode.rewards[i],
                    "next_state": episode.observations[i + 1],
                    "done": episode.dones[i],
                }
                transitions.append(transition)
        
        return transitions
    
    def _create_dataset(self, transitions: List[Dict]):
        """Create a simple dataset from transitions."""
        # Simplified: return transitions directly
        # In practice, would convert to tensors and handle batching properly
        return transitions
    
    def _train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        total_loss = 0.0
        total_td_loss = 0.0
        total_cql_loss = 0.0
        num_batches = 0
        
        self.q_network.train()
        
        for batch in tqdm(dataloader, desc="Training"):
            # Convert batch to tensors (simplified - assumes proper batching)
            states = self._batch_to_tensors(batch, "state")
            actions = self._batch_to_tensors(batch, "action")
            rewards = torch.tensor([t["reward"] for t in batch], dtype=torch.float32).to(self.config.device)
            next_states = self._batch_to_tensors(batch, "next_state")
            dones = torch.tensor([t["done"] for t in batch], dtype=torch.bool).to(self.config.device)
            
            # Compute Q-values
            q_values = self.q_network(states, actions).squeeze()
            
            # TD target: r + gamma * max_a' Q_target(s', a') * (played - done)
            with torch.no_grad():
                # Simplified: sample random next actions for target (in practice, would use policy)
                next_actions = self._sample_actions(next_states)
                next_q_values = self.target_q_network(next_states, next_actions).squeeze()
                td_targets = rewards + self.config.gamma * next_q_values * (~dones).float()
            
            # TD loss
            td_loss = self.mse_loss(q_values, td_targets)
            
            # CQL conservative penalty: log sum exp Q(s,a) - mean Q(s,a_dataset)
            # Sample random actions for penalty computation
            random_actions = self._sample_random_actions(states, num_samples=10)
            q_random = torch.stack([
                self.q_network(states, rand_act).squeeze() for rand_act in random_actions
            ], dim=-1)  # (batch_size, num_samples)
            
            # log sum exp Q(s, a_random)
            log_sum_exp = torch.logsumexp(q_random, dim=-1)
            
            # mean Q(s, a_dataset)
            mean_q_dataset = q_values.mean()
            
            # CQL penalty
            cql_loss = (log_sum_exp.mean() - mean_q_dataset) * self.config.cql_alpha
            
            # Total loss
            loss = td_loss + cql_loss
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.config.grad_clip)
            self.optimizer.step()
            
            total_loss += loss.item()
            total_td_loss += td_loss.item()
            total_cql_loss += cql_loss.item()
            num_batches += 1
        
        return {
            "total": total_loss / num_batches if num_batches > 0 else 0.0,
            "td": total_td_loss / num_batches if num_batches > 0 else 0.0,
            "cql": total_cql_loss / num_batches if num_batches > 0 else 0.0,
        }
    
    def _batch_to_tensors(self, batch, key: str):
        """Convert batch to tensor format (simplified)."""
        # In practice, would properly handle dict-based observations/actions
        # For now, return as-is (assumes they're already dicts)
        if key in ["state", "next_state"]:
            return [item[key] for item in batch]
        elif key == "action":
            return [item[key] for item in batch]
        return None
    
    def _sample_actions(self, states) -> List[Dict]:
        """Sample actions for given states (simplified - random sampling)."""
        # In practice, would use a policy or all possible actions
        batch_size = len(states)
        return [self._sample_random_action() for _ in range(batch_size)]
    
    def _sample_random_action(self) -> Dict:
        """Sample a random action."""
        return {
            "aircraft_id": np.random.randint(0, self.config.max_aircraft + 1),
            "command_type": np.random.randint(0, 5),
            "altitude": np.random.randint(0, 18),
            "heading": np.random.randint(0, 13),
            "speed": np.random.randint(0, 8),
        }
    
    def _sample_random_actions(self, states, num_samples: int) -> List[List[Dict]]:
        """Sample multiple random actions for each state."""
        batch_size = len(states)
        return [[self._sample_random_action() for _ in range(num_samples)] for _ in range(batch_size)]
    
    def _update_target_network(self):
        """Soft update target network."""
        for target_param, param in zip(
            self.target_q_network.parameters(),
            self.q_network.parameters(),
        ):
            target_param.data.copy_(
                self.config.tau * param.data + (1 - self.config.tau) * target_param.data
            )
    
    def save_checkpoint(self, epoch: int):
        """Save model checkpoint."""
        checkpoint_path = Path(self.config.checkpoint_dir) / f"checkpoint_epoch_{epoch}.pt"
        torch.save({
            "epoch": epoch,
            "q_network_state_dict": self.q_network.state_dict(),
            "target_q_network_state_dict": self.target_q_network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "train_losses": self.train_losses,
        }, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device)
        self.q_network.load_state_dict(checkpoint["q_network_state_dict"])
        self.target_q_network.load_state_dict(checkpoint["target_q_network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.train_losses = checkpoint.get("train_losses", [])
        logger.info(f"Loaded checkpoint from {checkpoint_path}")

