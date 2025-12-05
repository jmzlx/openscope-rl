"""
TD-MPC 2 Trainer.

This module provides training utilities for TD-MPC 2, including:
- Joint world model training (dynamics + reward)
- Q-function training with TD targets
- Replay buffer management
- Online data collection with MPC policy
- WandB logging and experiment tracking
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from models.tdmpc2 import TDMPC2Model, TDMPC2Config, TDMPC2QNetwork
from training.tdmpc2_planner import MPCPlanner, MPCPlannerConfig
from environment.utils import get_device

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logging.warning("wandb not available, logging will be limited")

logger = logging.getLogger(__name__)


class ReplayBuffer:
    """
    Experience replay buffer for TD-MPC 2.
    
    Stores transitions (obs, action, reward, next_obs, done) for training.
    """
    
    def __init__(self, capacity: int = 100000, device: Optional[str] = None):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum buffer size
            device: Device to place tensors on
            
        Raises:
            ValueError: If capacity is invalid
        """
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        
        self.capacity = capacity
        self.device = get_device(device)
        
        self.observations = deque(maxlen=capacity)
        self.actions = deque(maxlen=capacity)
        self.rewards = deque(maxlen=capacity)
        self.next_observations = deque(maxlen=capacity)
        self.dones = deque(maxlen=capacity)
    
    def add(
        self,
        obs: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: float,
        next_obs: Dict[str, np.ndarray],
        done: bool,
    ) -> None:
        """
        Add a transition to the buffer.
        
        Args:
            obs: Observation dictionary
            action: Action array
            reward: Reward scalar
            next_obs: Next observation dictionary
            done: Done flag
            
        Raises:
            ValueError: If required keys are missing
        """
        required_keys = ["aircraft", "aircraft_mask", "global_state"]
        for key in required_keys:
            if key not in obs:
                raise ValueError(f"Missing required key in obs: {key}")
            if key not in next_obs:
                raise ValueError(f"Missing required key in next_obs: {key}")
        
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_observations.append(next_obs)
        self.dones.append(done)
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Sample a batch from the buffer.
        
        Args:
            batch_size: Batch size
            
        Returns:
            Dictionary with batched tensors
            
        Raises:
            ValueError: If buffer is empty or batch_size is invalid
        """
        if len(self.observations) == 0:
            raise ValueError("Cannot sample from empty buffer")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        
        if len(self.observations) < batch_size:
            batch_size = len(self.observations)
            logger.warning(f"Reduced batch_size to {batch_size} (buffer size)")
        
        indices = np.random.choice(len(self.observations), batch_size, replace=False)
        
        # Batch observations
        aircraft_batch = torch.stack([
            torch.from_numpy(self.observations[i]["aircraft"]).float()
            for i in indices
        ]).to(self.device)
        
        aircraft_mask_batch = torch.stack([
            torch.from_numpy(self.observations[i]["aircraft_mask"]).bool()
            for i in indices
        ]).to(self.device)
        
        global_state_batch = torch.stack([
            torch.from_numpy(self.observations[i]["global_state"]).float()
            for i in indices
        ]).to(self.device)
        
        # Batch next observations
        next_aircraft_batch = torch.stack([
            torch.from_numpy(self.next_observations[i]["aircraft"]).float()
            for i in indices
        ]).to(self.device)
        
        next_aircraft_mask_batch = torch.stack([
            torch.from_numpy(self.next_observations[i]["aircraft_mask"]).bool()
            for i in indices
        ]).to(self.device)
        
        next_global_state_batch = torch.stack([
            torch.from_numpy(self.next_observations[i]["global_state"]).float()
            for i in indices
        ]).to(self.device)
        
        # Batch actions, rewards, dones
        actions_batch = torch.stack([
            torch.from_numpy(self.actions[i]).float()
            for i in indices
        ]).to(self.device)
        
        rewards_batch = torch.tensor([self.rewards[i] for i in indices]).float().to(self.device)
        dones_batch = torch.tensor([self.dones[i] for i in indices]).bool().to(self.device)
        
        return {
            "aircraft": aircraft_batch,
            "aircraft_mask": aircraft_mask_batch,
            "global_state": global_state_batch,
            "action": actions_batch,
            "reward": rewards_batch,
            "next_aircraft": next_aircraft_batch,
            "next_aircraft_mask": next_aircraft_mask_batch,
            "next_global_state": next_global_state_batch,
            "done": dones_batch,
        }
    
    def __len__(self) -> int:
        """Get buffer size."""
        return len(self.observations)


@dataclass
class TDMPC2TrainingConfig:
    """Configuration for TD-MPC 2 training."""
    
    # Model config
    model_config: TDMPC2Config = field(default_factory=TDMPC2Config)
    planner_config: MPCPlannerConfig = field(default_factory=MPCPlannerConfig)
    
    # Training
    num_steps: int = 1000000
    batch_size: int = 64
    learning_rate_model: float = 1e-3
    learning_rate_q: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    
    # Loss weights
    dynamics_loss_weight: float = 1.0
    reward_loss_weight: float = 1.0
    q_loss_weight: float = 1.0
    
    # Replay buffer
    buffer_capacity: int = 100000
    min_buffer_size: int = 1000
    
    # Data collection
    collect_steps_per_update: int = 1
    update_frequency: int = 1
    
    # Q-learning
    gamma: float = 0.99
    tau: float = 0.01  # Soft target update
    
    # Evaluation
    eval_frequency: int = 10000
    eval_episodes: int = 5
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints/tdmpc2"
    save_frequency: int = 25000
    
    # Logging
    log_frequency: int = 100
    use_wandb: bool = False
    wandb_project: str = "tdmpc2-openscope"
    wandb_entity: Optional[str] = None
    
    # Device
    device: str = field(default_factory=lambda: get_device())
    
    def __post_init__(self):
        """Validate configuration."""
        if self.num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {self.num_steps}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if not (0.0 < self.gamma <= 1.0):
            raise ValueError(f"gamma ({self.gamma}) must be in (0.0, 1.0]")
        if not (0.0 <= self.tau <= 1.0):
            raise ValueError(f"tau ({self.tau}) must be in [0.0, 1.0]")
        if self.min_buffer_size <= 0:
            raise ValueError(f"min_buffer_size must be positive, got {self.min_buffer_size}")
        if self.min_buffer_size > self.buffer_capacity:
            raise ValueError(
                f"min_buffer_size ({self.min_buffer_size}) must be <= buffer_capacity ({self.buffer_capacity})"
            )
        if self.eval_episodes <= 0:
            raise ValueError(f"eval_episodes must be positive, got {self.eval_episodes}")
    
    def to_dict(self) -> Dict:
        """
        Convert to dictionary for logging.
        
        Returns:
            Dictionary containing all training hyperparameters and configs.
        """
        return {
            "num_steps": self.num_steps,
            "batch_size": self.batch_size,
            "learning_rate_model": self.learning_rate_model,
            "learning_rate_q": self.learning_rate_q,
            "weight_decay": self.weight_decay,
            "grad_clip": self.grad_clip,
            "dynamics_loss_weight": self.dynamics_loss_weight,
            "reward_loss_weight": self.reward_loss_weight,
            "q_loss_weight": self.q_loss_weight,
            "buffer_capacity": self.buffer_capacity,
            "min_buffer_size": self.min_buffer_size,
            "collect_steps_per_update": self.collect_steps_per_update,
            "update_frequency": self.update_frequency,
            "gamma": self.gamma,
            "tau": self.tau,
            "eval_frequency": self.eval_frequency,
            "eval_episodes": self.eval_episodes,
            "save_frequency": self.save_frequency,
            "log_frequency": self.log_frequency,
            **self.model_config.to_dict(),
            **self.planner_config.to_dict(),
        }


class TDMPC2Trainer:
    """
    Trainer for TD-MPC 2.
    
    Implements joint training of world model and Q-function with MPC planning.
    
    Example:
        >>> from environment import PlaywrightEnv
        >>> env = PlaywrightEnv(airport="KLAS", max_aircraft=10)
        >>> config = TDMPC2TrainingConfig(num_steps=100000)
        >>> trainer = TDMPC2Trainer(env, config)
        >>> trainer.train()
    """
    
    def __init__(
        self,
        env,
        config: TDMPC2TrainingConfig,
    ):
        """
        Initialize trainer.
        
        Args:
            env: Gymnasium environment
            config: Training configuration
            
        Raises:
            ValueError: If environment is invalid
            RuntimeError: If WandB initialization fails
        """
        if env is None:
            raise ValueError("env cannot be None")
        
        self.env = env
        self.config = config
        
        # Create model
        self.model = TDMPC2Model(config.model_config)
        self.model.to(config.device)
        
        # Create target Q-network (for TD targets)
        self.target_q_network = TDMPC2QNetwork(config.model_config)
        self.target_q_network.load_state_dict(self.model.q_network.state_dict())
        self.target_q_network.to(config.device)
        self.target_q_network.eval()  # Target network is always in eval mode
        
        # Create planner
        self.planner = MPCPlanner(self.model, config.planner_config)
        
        # Optimizers
        # Separate optimizers for model and Q-network
        model_params = list(self.model.encoder.parameters()) + \
                      list(self.model.dynamics.parameters()) + \
                      list(self.model.reward.parameters())
        
        self.model_optimizer = torch.optim.AdamW(
            model_params,
            lr=config.learning_rate_model,
            weight_decay=config.weight_decay,
        )
        
        self.q_optimizer = torch.optim.AdamW(
            self.model.q_network.parameters(),
            lr=config.learning_rate_q,
            weight_decay=config.weight_decay,
        )
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            capacity=config.buffer_capacity,
            device=config.device,
        )
        
        # Training state
        self.training_step = 0  # Training loop iteration
        self.env_step = 0  # Total environment steps
        self.episode = 0
        self.best_eval_return = float("-inf")
        
        # Checkpoint directory
        self.checkpoint_dir = Path(config.checkpoint_dir)
        try:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise RuntimeError(f"Failed to create checkpoint directory {self.checkpoint_dir}: {e}")
        
        # Initialize WandB if requested
        if config.use_wandb:
            if not WANDB_AVAILABLE:
                raise RuntimeError(
                    "WandB requested but not available. Install with: pip install wandb"
                )
            try:
                wandb.init(
                    project=config.wandb_project,
                    entity=config.wandb_entity,
                    config=config.to_dict(),
                    name=f"tdmpc2-{config.model_config.latent_dim}l",
                )
                wandb.watch(self.model, log="all", log_freq=config.log_frequency)
            except Exception as e:
                raise RuntimeError(f"Failed to initialize WandB: {e}")
        
        logger.info(f"Initialized TD-MPC 2 trainer")
        logger.info(f"Model parameters: {self._count_parameters():,}")
        logger.info(f"Device: {config.device}")
        logger.info(f"Checkpoint dir: {self.checkpoint_dir}")
    
    def _count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def train(self):
        """Main training loop."""
        logger.info(f"Starting training for {self.config.num_steps} steps")
        
        # Initial data collection
        logger.info(f"Collecting initial data (min {self.config.min_buffer_size} transitions)...")
        while len(self.replay_buffer) < self.config.min_buffer_size:
            try:
                self._collect_episode()
            except Exception as e:
                logger.error(f"Error during initial data collection: {e}", exc_info=True)
                raise
        
        logger.info(f"Initial buffer size: {len(self.replay_buffer)}")
        
        # Training loop
        pbar = tqdm(range(self.config.num_steps), desc="Training")
        
        for training_step in pbar:
            self.training_step = training_step
            
            # Collect data
            if training_step % self.config.collect_steps_per_update == 0:
                try:
                    self._collect_episode()
                except Exception as e:
                    logger.error(f"Error during data collection at step {training_step}: {e}", exc_info=True)
                    # Continue training even if one episode fails
                    continue
            
            # Update model
            if training_step % self.config.update_frequency == 0 and len(self.replay_buffer) >= self.config.min_buffer_size:
                try:
                    metrics = self._update_model()
                    
                    # Logging
                    if training_step % self.config.log_frequency == 0:
                        pbar.set_postfix(metrics)
                        self._log_metrics(metrics, training_step)
                except Exception as e:
                    logger.error(f"Error during model update at step {training_step}: {e}", exc_info=True)
                    # Continue training even if one update fails
                    continue
            
            # Soft update target Q-network
            if training_step % self.config.update_frequency == 0:
                self._soft_update_target_q()
            
            # Evaluation
            if training_step % self.config.eval_frequency == 0 and training_step > 0:
                try:
                    eval_metrics = self._evaluate()
                    self._log_metrics(eval_metrics, training_step)
                except Exception as e:
                    logger.error(f"Error during evaluation at step {training_step}: {e}", exc_info=True)
                    # Continue training even if evaluation fails
            
            # Checkpointing
            if training_step % self.config.save_frequency == 0 and training_step > 0:
                try:
                    self.save_checkpoint(f"checkpoint_step_{training_step}.pt")
                except Exception as e:
                    logger.error(f"Error saving checkpoint at step {training_step}: {e}", exc_info=True)
        
        logger.info("Training complete")
        try:
            self.save_checkpoint("final_model.pt")
        except Exception as e:
            logger.error(f"Error saving final checkpoint: {e}", exc_info=True)
            raise
    
    def _collect_episode(self):
        """Collect one episode of data using MPC policy."""
        obs, info = self.env.reset()
        episode_reward = 0.0
        episode_length = 0
        max_episode_length = 1000  # Prevent infinite episodes
        
        done = False
        while not done and episode_length < max_episode_length:
            # Plan action using MPC
            aircraft_tensor = torch.from_numpy(obs["aircraft"]).float().unsqueeze(0).to(self.config.device)
            mask_tensor = torch.from_numpy(obs["aircraft_mask"]).bool().unsqueeze(0).to(self.config.device)
            global_tensor = torch.from_numpy(obs["global_state"]).float().unsqueeze(0).to(self.config.device)
            
            with torch.no_grad():
                action_tensor = self.planner.plan(aircraft_tensor, mask_tensor, global_tensor)
                action = action_tensor.squeeze(0).cpu().numpy()
            
            # Step environment
            next_obs, reward, terminated, truncated, info = self.env.step(self._tensor_to_action(action))
            done = terminated or truncated
            
            # Store transition
            self.replay_buffer.add(obs, action, reward, next_obs, done)
            
            obs = next_obs
            episode_reward += reward
            episode_length += 1
            self.env_step += 1
        
        self.episode += 1
        
        if self.config.use_wandb and WANDB_AVAILABLE:
            try:
                wandb.log({
                    "episode/reward": episode_reward,
                    "episode/length": episode_length,
                    "episode/num": self.episode,
                }, step=self.training_step)
            except Exception as e:
                logger.error(f"Error logging episode metrics to WandB: {e}", exc_info=True)
    
    def _tensor_to_action(self, action_tensor: np.ndarray) -> Dict[str, int]:
        """
        Convert action tensor to environment action format.
        
        Args:
            action_tensor: Action as numpy array (action_dim,)
            
        Returns:
            Action dictionary
            
        Raises:
            ValueError: If action tensor has wrong shape
        """
        if action_tensor.shape != (self.config.model_config.action_dim,):
            raise ValueError(
                f"Expected action shape ({self.config.model_config.action_dim},), "
                f"got {action_tensor.shape}"
            )
        
        # Discretize continuous actions
        # This is a simplified version - in practice, you'd want proper discretization
        return {
            "aircraft_id": int(np.clip(action_tensor[0], 0, self.config.model_config.max_aircraft)),
            "command_type": int(np.clip(action_tensor[1], 0, 4)),
            "altitude": int(np.clip(action_tensor[2], 0, 40)),
            "heading": int(np.clip(action_tensor[3], -180, 180)),
            "speed": int(np.clip(action_tensor[4], 140, 520)),
        }
    
    def _update_model(self) -> Dict[str, float]:
        """
        Update model and Q-network from replay buffer.
        
        Returns:
            Dictionary of training metrics
            
        Raises:
            ValueError: If buffer is too small
        """
        if len(self.replay_buffer) < self.config.min_buffer_size:
            raise ValueError(
                f"Buffer too small: {len(self.replay_buffer)} < {self.config.min_buffer_size}"
            )
        
        # Sample batch
        batch = self.replay_buffer.sample(self.config.batch_size)
        
        # Encode observations
        latent = self.model.encode(
            batch["aircraft"],
            batch["aircraft_mask"],
            batch["global_state"],
        )
        
        next_latent = self.model.encode(
            batch["next_aircraft"],
            batch["next_aircraft_mask"],
            batch["next_global_state"],
        )
        
        # World model loss
        # Predict next latent and reward
        pred_next_latent = self.model.dynamics(latent, batch["action"])
        pred_reward = self.model.reward(pred_next_latent)
        
        dynamics_loss = F.mse_loss(pred_next_latent, next_latent.detach())
        reward_loss = F.mse_loss(pred_reward, batch["reward"].unsqueeze(-1))
        
        world_model_loss = (
            self.config.dynamics_loss_weight * dynamics_loss +
            self.config.reward_loss_weight * reward_loss
        )
        
        # Q-learning loss
        q_values = self.model.q_network(latent, batch["action"])
        
        with torch.no_grad():
            # Target Q-values using max_a' Q(s', a') for off-policy Q-learning
            # Sample candidate actions and take max (more efficient than enumerating all actions)
            batch_size = next_latent.size(0)
            num_candidates = 10  # Sample 10 candidate actions per state
            
            # Sample random actions in normalized space [-1, 1]
            candidate_actions = torch.randn(
                batch_size, num_candidates, self.config.model_config.action_dim,
                device=next_latent.device
            )
            candidate_actions = torch.clamp(candidate_actions, -1.0, 1.0)
            
            # Expand next_latent for all candidates: (batch_size, latent_dim) -> (batch_size, num_candidates, latent_dim)
            next_latent_expanded = next_latent.unsqueeze(1).expand(-1, num_candidates, -1)
            next_latent_flat = next_latent_expanded.reshape(batch_size * num_candidates, -1)
            candidate_actions_flat = candidate_actions.reshape(batch_size * num_candidates, -1)
            
            # Compute Q-values for all candidates
            candidate_q_values = self.target_q_network(next_latent_flat, candidate_actions_flat)
            candidate_q_values = candidate_q_values.reshape(batch_size, num_candidates)
            
            # Take max over candidates
            next_q_values = candidate_q_values.max(dim=1, keepdim=True)[0]
            
            # TD target: r + gamma * max_a' Q(s', a') * (1 - done)
            target_q = batch["reward"].unsqueeze(-1) + \
                      self.config.gamma * next_q_values * (~batch["done"]).unsqueeze(-1).float()
        
        q_loss = F.mse_loss(q_values, target_q)
        
        # Total loss
        total_loss = world_model_loss + self.config.q_loss_weight * q_loss
        
        # Backward pass
        self.model_optimizer.zero_grad()
        world_model_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        self.model_optimizer.step()
        
        self.q_optimizer.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.q_network.parameters(), self.config.grad_clip)
        self.q_optimizer.step()
        
        return {
            "loss": total_loss.item(),
            "dynamics_loss": dynamics_loss.item(),
            "reward_loss": reward_loss.item(),
            "q_loss": q_loss.item(),
        }
    
    def _soft_update_target_q(self):
        """Soft update target Q-network."""
        for target_param, param in zip(
            self.target_q_network.parameters(),
            self.model.q_network.parameters(),
        ):
            target_param.data.copy_(
                self.config.tau * param.data + (1.0 - self.config.tau) * target_param.data
            )
    
    def _evaluate(self) -> Dict[str, float]:
        """
        Evaluate model in environment.
        
        Returns:
            Dictionary of evaluation metrics
            
        Raises:
            RuntimeError: If evaluation fails
        """
        self.model.eval()
        
        episode_returns = []
        episode_lengths = []
        
        for episode_idx in range(self.config.eval_episodes):
            try:
                obs, info = self.env.reset()
                episode_reward = 0.0
                episode_length = 0
                done = False
                max_steps = 1000
                
                while not done and episode_length < max_steps:
                    # Plan action
                    aircraft_tensor = torch.from_numpy(obs["aircraft"]).float().unsqueeze(0).to(self.config.device)
                    mask_tensor = torch.from_numpy(obs["aircraft_mask"]).bool().unsqueeze(0).to(self.config.device)
                    global_tensor = torch.from_numpy(obs["global_state"]).float().unsqueeze(0).to(self.config.device)
                    
                    with torch.no_grad():
                        action_tensor = self.planner.plan(aircraft_tensor, mask_tensor, global_tensor)
                        action = self._tensor_to_action(action_tensor.squeeze(0).cpu().numpy())
                    
                    obs, reward, terminated, truncated, info = self.env.step(action)
                    done = terminated or truncated
                    
                    episode_reward += reward
                    episode_length += 1
                
                episode_returns.append(episode_reward)
                episode_lengths.append(episode_length)
            except Exception as e:
                logger.error(f"Error during evaluation episode {episode_idx}: {e}", exc_info=True)
                # Continue with other episodes
                continue
        
        self.model.train()
        
        if len(episode_returns) == 0:
            raise RuntimeError("All evaluation episodes failed")
        
        metrics = {
            "eval/return_mean": np.mean(episode_returns),
            "eval/return_std": np.std(episode_returns),
            "eval/length_mean": np.mean(episode_lengths),
        }
        
        if np.mean(episode_returns) > self.best_eval_return:
            self.best_eval_return = np.mean(episode_returns)
            try:
                self.save_checkpoint("best_model.pt")
            except Exception as e:
                logger.error(f"Error saving best model: {e}", exc_info=True)
        
        return metrics
    
    def _log_metrics(self, metrics: Dict[str, float], step: int):
        """Log metrics to console and WandB."""
        # Console logging
        logger.info(f"Step {step}: {metrics}")
        
        # WandB logging
        if self.config.use_wandb and WANDB_AVAILABLE:
            try:
                wandb.log(metrics, step=step)
            except Exception as e:
                logger.error(f"Error logging to WandB: {e}", exc_info=True)
    
    def save_checkpoint(self, filename: str):
        """
        Save model checkpoint.
        
        Args:
            filename: Checkpoint filename
            
        Raises:
            OSError: If file cannot be written
        """
        checkpoint_path = self.checkpoint_dir / filename
        
        checkpoint = {
            "training_step": self.training_step,
            "env_step": self.env_step,
            "episode": self.episode,
            "model_state_dict": self.model.state_dict(),
            "target_q_state_dict": self.target_q_network.state_dict(),
            "model_optimizer_state_dict": self.model_optimizer.state_dict(),
            "q_optimizer_state_dict": self.q_optimizer.state_dict(),
            "config": self.config.to_dict(),
            "best_eval_return": self.best_eval_return,
        }
        
        try:
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
        except OSError as e:
            raise OSError(f"Failed to save checkpoint to {checkpoint_path}: {e}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint
            
        Raises:
            FileNotFoundError: If checkpoint doesn't exist
            RuntimeError: If checkpoint loading fails
        """
        checkpoint_path_obj = Path(checkpoint_path)
        if not checkpoint_path_obj.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.config.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint from {checkpoint_path}: {e}")
        
        try:
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.target_q_network.load_state_dict(checkpoint["target_q_state_dict"])
            self.model_optimizer.load_state_dict(checkpoint["model_optimizer_state_dict"])
            self.q_optimizer.load_state_dict(checkpoint["q_optimizer_state_dict"])
            # Handle both old and new checkpoint formats
            self.training_step = checkpoint.get("training_step", checkpoint.get("step", 0))
            self.env_step = checkpoint.get("env_step", 0)
            self.episode = checkpoint["episode"]
            self.best_eval_return = checkpoint.get("best_eval_return", float("-inf"))
        except KeyError as e:
            raise RuntimeError(f"Missing key in checkpoint: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model state: {e}")
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")

