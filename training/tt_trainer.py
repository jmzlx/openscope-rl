"""
Trainer for Trajectory Transformer.

This module provides training functionality for the Trajectory Transformer model,
including data preparation, training loops, evaluation, and checkpointing.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
import json

from models.trajectory_transformer import TrajectoryTransformer, TrajectoryTransformerConfig


logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training Trajectory Transformer."""

    # Training
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0

    # Loss weights
    state_loss_weight: float = 1.0
    action_loss_weight: float = 1.0
    reward_loss_weight: float = 0.1

    # Data
    context_length: int = 20
    train_split: float = 0.8

    # Optimization
    warmup_steps: int = 1000
    lr_decay: bool = True
    lr_decay_steps: int = 10000

    # Evaluation
    eval_every: int = 100
    save_every: int = 1000

    # Logging
    log_every: int = 10
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        """Create directories."""
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)


class TrajectoryDataset(Dataset):
    """Dataset for trajectory data."""

    def __init__(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        context_length: int = 20,
    ):
        """
        Initialize trajectory dataset.

        Args:
            states: Array of states (num_episodes, max_steps, state_dim)
            actions: Array of actions (num_episodes, max_steps)
            rewards: Array of rewards (num_episodes, max_steps)
            context_length: Maximum context length for sequences
        """
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.context_length = context_length

        # Get valid episode lengths (non-padded steps)
        self.episode_lengths = []
        for i in range(len(states)):
            # Find first all-zero state as end marker
            valid_steps = len(states[i])
            for j in range(len(states[i])):
                if np.all(states[i, j] == 0):
                    valid_steps = j
                    break
            self.episode_lengths.append(max(1, valid_steps))

    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.states)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a training sample.

        Returns:
            Dictionary with states, actions, rewards, next_states
        """
        episode_length = self.episode_lengths[idx]

        # Get full episode
        states = self.states[idx, :episode_length]
        actions = self.actions[idx, :episode_length]
        rewards = self.rewards[idx, :episode_length]

        # Sample a random context window
        if episode_length > self.context_length:
            start_idx = np.random.randint(0, episode_length - self.context_length)
            end_idx = start_idx + self.context_length
        else:
            start_idx = 0
            end_idx = episode_length

        # Extract context
        ctx_states = states[start_idx:end_idx]
        ctx_actions = actions[start_idx:end_idx]
        ctx_rewards = rewards[start_idx:end_idx]

        # Next states for prediction
        next_states = states[start_idx + 1:end_idx + 1] if end_idx < episode_length else states[start_idx + 1:]

        # Pad if necessary
        ctx_len = len(ctx_states)
        if ctx_len < self.context_length:
            pad_len = self.context_length - ctx_len
            ctx_states = np.concatenate([ctx_states, np.zeros((pad_len, ctx_states.shape[1]))])
            ctx_actions = np.concatenate([ctx_actions, np.zeros(pad_len, dtype=np.int64)])
            ctx_rewards = np.concatenate([ctx_rewards, np.zeros(pad_len)])
            next_states = np.concatenate([next_states, np.zeros((pad_len, next_states.shape[1]))])

        return {
            "states": torch.FloatTensor(ctx_states),
            "actions": torch.LongTensor(ctx_actions),
            "rewards": torch.FloatTensor(ctx_rewards).unsqueeze(-1),
            "next_states": torch.FloatTensor(next_states),
            "mask": torch.ones(ctx_len, dtype=torch.bool)
            if ctx_len < self.context_length
            else torch.ones(self.context_length, dtype=torch.bool),
        }


class TrajectoryTransformerTrainer:
    """
    Trainer for Trajectory Transformer model.

    Example:
        >>> config = TrainingConfig(batch_size=32, num_epochs=100)
        >>> model_config = TrajectoryTransformerConfig(state_dim=128, action_dim=65)
        >>> trainer = TrajectoryTransformerTrainer(model_config, config)
        >>>
        >>> # Prepare data
        >>> states = np.random.randn(1000, 50, 128)
        >>> actions = np.random.randint(0, 65, (1000, 50))
        >>> rewards = np.random.randn(1000, 50)
        >>>
        >>> # Train
        >>> trainer.train(states, actions, rewards)
    """

    def __init__(
        self,
        model_config: TrajectoryTransformerConfig,
        training_config: TrainingConfig,
    ):
        """
        Initialize trainer.

        Args:
            model_config: Model configuration
            training_config: Training configuration
        """
        self.model_config = model_config
        self.config = training_config

        # Create model
        self.model = TrajectoryTransformer(model_config)
        self.model.to(self.config.device)

        # Update loss weights
        self.model.config.state_loss_weight = self.config.state_loss_weight
        self.model.config.action_loss_weight = self.config.action_loss_weight
        self.model.config.reward_loss_weight = self.config.reward_loss_weight

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Learning rate scheduler
        if self.config.lr_decay:
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.lr_decay_steps,
                eta_min=self.config.learning_rate * 0.1,
            )
        else:
            self.scheduler = None

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float("inf")

        # Metrics
        self.train_losses = []
        self.eval_losses = []

    def prepare_data(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Prepare training and validation dataloaders.

        Args:
            states: State array (num_episodes, max_steps, state_dim)
            actions: Action array (num_episodes, max_steps)
            rewards: Reward array (num_episodes, max_steps)

        Returns:
            Tuple of (train_loader, val_loader)
        """
        # Split data
        num_episodes = len(states)
        split_idx = int(num_episodes * self.config.train_split)

        train_dataset = TrajectoryDataset(
            states[:split_idx],
            actions[:split_idx],
            rewards[:split_idx],
            context_length=self.config.context_length,
        )

        val_dataset = TrajectoryDataset(
            states[split_idx:],
            actions[split_idx:],
            rewards[split_idx:],
            context_length=self.config.context_length,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
        )

        return train_loader, val_loader

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform a single training step.

        Args:
            batch: Batch of data

        Returns:
            Dictionary of losses
        """
        # Move to device
        states = batch["states"].to(self.config.device)
        actions = batch["actions"].to(self.config.device)
        rewards = batch["rewards"].to(self.config.device)
        next_states = batch["next_states"].to(self.config.device)

        # Forward pass
        self.model.train()
        losses = self.model.compute_loss(states, actions, rewards, next_states)

        # Backward pass
        self.optimizer.zero_grad()
        losses["total_loss"].backward()

        # Gradient clipping
        if self.config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

        self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        # Convert to Python floats
        return {k: v.item() for k, v in losses.items()}

    def eval_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform a single evaluation step.

        Args:
            batch: Batch of data

        Returns:
            Dictionary of losses
        """
        # Move to device
        states = batch["states"].to(self.config.device)
        actions = batch["actions"].to(self.config.device)
        rewards = batch["rewards"].to(self.config.device)
        next_states = batch["next_states"].to(self.config.device)

        # Forward pass
        self.model.eval()
        with torch.no_grad():
            losses = self.model.compute_loss(states, actions, rewards, next_states)

        return {k: v.item() for k, v in losses.items()}

    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model on validation set.

        Args:
            val_loader: Validation data loader

        Returns:
            Dictionary of average losses
        """
        total_losses = {
            "total_loss": 0.0,
            "state_loss": 0.0,
            "action_loss": 0.0,
            "reward_loss": 0.0,
        }

        num_batches = 0
        for batch in val_loader:
            losses = self.eval_step(batch)
            for k, v in losses.items():
                total_losses[k] += v
            num_batches += 1

        # Average losses
        avg_losses = {k: v / num_batches for k, v in total_losses.items()}
        return avg_losses

    def train(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
    ):
        """
        Train the model.

        Args:
            states: State array (num_episodes, max_steps, state_dim)
            actions: Action array (num_episodes, max_steps)
            rewards: Reward array (num_episodes, max_steps)
        """
        logger.info("Preparing data...")
        train_loader, val_loader = self.prepare_data(states, actions, rewards)

        logger.info(f"Training on {len(train_loader.dataset)} episodes")
        logger.info(f"Validating on {len(val_loader.dataset)} episodes")

        for epoch in range(self.config.num_epochs):
            self.epoch = epoch

            # Training loop
            epoch_losses = []
            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.config.num_epochs}")

            for batch in pbar:
                losses = self.train_step(batch)
                epoch_losses.append(losses)
                self.global_step += 1

                # Logging
                if self.global_step % self.config.log_every == 0:
                    avg_loss = np.mean([l["total_loss"] for l in epoch_losses[-10:]])
                    pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

                # Evaluation
                if self.global_step % self.config.eval_every == 0:
                    eval_losses = self.evaluate(val_loader)
                    self.eval_losses.append(eval_losses)

                    logger.info(
                        f"Step {self.global_step}: "
                        f"Eval Loss: {eval_losses['total_loss']:.4f}, "
                        f"State: {eval_losses['state_loss']:.4f}, "
                        f"Action: {eval_losses['action_loss']:.4f}, "
                        f"Reward: {eval_losses['reward_loss']:.4f}"
                    )

                    # Save best model
                    if eval_losses["total_loss"] < self.best_loss:
                        self.best_loss = eval_losses["total_loss"]
                        self.save_checkpoint("best_model.pt")

                # Checkpointing
                if self.global_step % self.config.save_every == 0:
                    self.save_checkpoint(f"checkpoint_step_{self.global_step}.pt")

            # Epoch summary
            avg_epoch_loss = np.mean([l["total_loss"] for l in epoch_losses])
            self.train_losses.append(avg_epoch_loss)
            logger.info(f"Epoch {epoch + 1} - Average Loss: {avg_epoch_loss:.4f}")

        logger.info("Training complete!")

    def save_checkpoint(self, filename: str):
        """
        Save model checkpoint.

        Args:
            filename: Checkpoint filename
        """
        checkpoint_path = Path(self.config.checkpoint_dir) / filename

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_loss": self.best_loss,
            "model_config": self.model_config.__dict__,
            "training_config": self.config.__dict__,
        }

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

    def load_checkpoint(self, filename: str):
        """
        Load model checkpoint.

        Args:
            filename: Checkpoint filename
        """
        checkpoint_path = Path(self.config.checkpoint_dir) / filename

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.config.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler and checkpoint["scheduler_state_dict"]:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.global_step = checkpoint["global_step"]
        self.epoch = checkpoint["epoch"]
        self.best_loss = checkpoint["best_loss"]

        logger.info(f"Loaded checkpoint from {checkpoint_path}")

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get training metrics.

        Returns:
            Dictionary of metrics
        """
        return {
            "train_losses": self.train_losses,
            "eval_losses": self.eval_losses,
            "best_loss": self.best_loss,
            "global_step": self.global_step,
            "epoch": self.epoch,
        }


def create_trainer(
    state_dim: int,
    action_dim: int,
    **kwargs
) -> TrajectoryTransformerTrainer:
    """
    Create a Trajectory Transformer trainer with default configurations.

    Args:
        state_dim: State dimension
        action_dim: Action dimension
        **kwargs: Additional configuration parameters

    Returns:
        Initialized trainer

    Example:
        >>> trainer = create_trainer(
        ...     state_dim=128,
        ...     action_dim=65,
        ...     batch_size=64,
        ...     num_epochs=200
        ... )
    """
    # Separate model and training config parameters
    model_params = {
        "state_dim": state_dim,
        "action_dim": action_dim,
    }

    training_params = {}

    for k, v in kwargs.items():
        if k in ["embed_dim", "num_layers", "num_heads", "ff_dim", "dropout", "context_length"]:
            model_params[k] = v
        else:
            training_params[k] = v

    model_config = TrajectoryTransformerConfig(**model_params)
    training_config = TrainingConfig(**training_params)

    return TrajectoryTransformerTrainer(model_config, training_config)
