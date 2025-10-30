"""
Trainer for simple MLP-based world model.

This module provides training utilities for learning a dynamics model from
collected trajectories, enabling model-based RL training.
"""

import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from models.dynamics_model import SimpleDynamicsModel, flatten_observation, flatten_action
from environment.utils import get_device
from data.offline_dataset import Episode

logger = logging.getLogger(__name__)


@dataclass
class WorldModelConfig:
    """Configuration for world model training."""
    
    # Model
    hidden_dim: int = 256
    num_layers: int = 3
    dropout: float = 0.1
    
    # Training
    num_epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    
    # Data
    train_split: float = 0.8
    val_split: float = 0.1
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints/world_model"
    save_every: int = 10
    
    # Device
    device: str = field(default_factory=lambda: get_device())


class DynamicsDataset(Dataset):
    """Dataset for dynamics model training."""
    
    def __init__(
        self,
        episodes: List[Episode],
        state_dim: int,
        action_dim: int,
    ):
        """
        Initialize dataset.
        
        Args:
            episodes: List of episodes
            state_dim: State dimension (after flattening)
            action_dim: Action dimension (after flattening)
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Extract transitions
        self.states = []
        self.actions = []
        self.next_states = []
        self.rewards = []
        
        for episode in episodes:
            for i in range(len(episode.observations) - 1):
                obs = episode.observations[i]
                action = episode.actions[i]
                next_obs = episode.observations[i + 1]
                reward = episode.rewards[i]
                
                # Flatten
                state = flatten_observation(obs)
                next_state = flatten_observation(next_obs)
                action_vec = flatten_action(action)
                
                self.states.append(state)
                self.actions.append(action_vec)
                self.next_states.append(next_state)
                self.rewards.append(reward)
        
        logger.info(f"Created dataset with {len(self.states)} transitions")
    
    def __len__(self) -> int:
        return len(self.states)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "state": torch.from_numpy(self.states[idx]).float(),
            "action": torch.from_numpy(self.actions[idx]).float(),
            "next_state": torch.from_numpy(self.next_states[idx]).float(),
            "reward": torch.tensor(self.rewards[idx]).float(),
        }


class WorldModelTrainer:
    """Trainer for simple dynamics model."""
    
    def __init__(
        self,
        config: WorldModelConfig,
        state_dim: int,
        action_dim: int,
    ):
        """
        Initialize trainer.
        
        Args:
            config: Training configuration
            state_dim: State dimension
            action_dim: Action dimension
        """
        self.config = config
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Create model
        self.model = SimpleDynamicsModel(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
        ).to(config.device)
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # Loss function
        self.state_loss_fn = nn.MSELoss()
        self.reward_loss_fn = nn.MSELoss()
        
        # Create checkpoint directory
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> Dict[str, List[float]]:
        """
        Train the dynamics model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            
        Returns:
            Dictionary with training history
        """
        self.model.train()
        
        for epoch in range(self.config.num_epochs):
            # Training
            train_loss = self._train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validation
            if val_loader is not None:
                val_loss = self._validate(val_loader)
                self.val_losses.append(val_loss)
                logger.info(f"Epoch {epoch+1}/{self.config.num_epochs}: "
                          f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
            else:
                logger.info(f"Epoch {epoch+1}/{self.config.num_epochs}: "
                          f"train_loss={train_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint(epoch + 1)
        
        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
        }
    
    def _train_epoch(self, loader: DataLoader) -> float:
        """Train for one epoch."""
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(loader, desc="Training"):
            states = batch["state"].to(self.config.device)
            actions = batch["action"].to(self.config.device)
            next_states = batch["next_state"].to(self.config.device)
            rewards = batch["reward"].to(self.config.device).unsqueeze(1)
            
            # Predict
            pred_next_states, pred_rewards = self.model(states, actions)
            
            # Compute loss
            state_loss = self.state_loss_fn(pred_next_states, next_states)
            reward_loss = self.reward_loss_fn(pred_rewards, rewards)
            loss = state_loss + reward_loss
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def _validate(self, loader: DataLoader) -> float:
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in loader:
                states = batch["state"].to(self.config.device)
                actions = batch["action"].to(self.config.device)
                next_states = batch["next_state"].to(self.config.device)
                rewards = batch["reward"].to(self.config.device).unsqueeze(1)
                
                # Predict
                pred_next_states, pred_rewards = self.model(states, actions)
                
                # Compute loss
                state_loss = self.state_loss_fn(pred_next_states, next_states)
                reward_loss = self.reward_loss_fn(pred_rewards, rewards)
                loss = state_loss + reward_loss
                
                total_loss += loss.item()
                num_batches += 1
        
        self.model.train()
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def save_checkpoint(self, epoch: int):
        """Save model checkpoint."""
        checkpoint_path = Path(self.config.checkpoint_dir) / f"checkpoint_epoch_{epoch}.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
        }, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.train_losses = checkpoint.get("train_losses", [])
        self.val_losses = checkpoint.get("val_losses", [])
        logger.info(f"Loaded checkpoint from {checkpoint_path}")


def compute_state_dim(max_aircraft: int = 20) -> int:
    """Compute state dimension after flattening."""
    from environment.constants import AIRCRAFT_FEATURE_DIM, GLOBAL_STATE_DIM
    
    aircraft_dim = max_aircraft * AIRCRAFT_FEATURE_DIM
    global_dim = GLOBAL_STATE_DIM
    conflict_dim = max_aircraft * max_aircraft
    
    return aircraft_dim + global_dim + conflict_dim


def compute_action_dim() -> int:
    """Compute action dimension after flattening."""
    return 5  # aircraft_id, command_type, altitude, heading, speed

