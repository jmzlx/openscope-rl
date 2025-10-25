"""
Decision Transformer trainer.

This module provides training utilities for Decision Transformer with:
- Supervised learning on offline trajectories
- WandB logging and experiment tracking
- Checkpoint management
- Evaluation on different target returns
"""

import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logging.warning("wandb not available, logging will be limited")

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for Decision Transformer training."""

    # Model
    state_dim: int = 14
    max_aircraft: int = 20
    hidden_size: int = 256
    n_layer: int = 6
    n_head: int = 8
    dropout: float = 0.1
    context_len: int = 20
    max_ep_len: int = 1000

    # Training
    num_epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    warmup_steps: int = 1000

    # Data
    return_scale: float = 1000.0
    scale_returns: bool = True

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every: int = 10
    eval_every: int = 5

    # Logging
    use_wandb: bool = False
    wandb_project: str = "decision-transformer-atc"
    wandb_entity: Optional[str] = None
    log_every: int = 100

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


class DecisionTransformerTrainer:
    """
    Trainer for Decision Transformer.

    This trainer implements supervised learning on offline trajectories,
    treating RL as a sequence modeling problem.

    Example:
        >>> from models.decision_transformer import MultiDiscreteDecisionTransformer
        >>> from data import create_dataloader, OfflineDatasetCollector
        >>>
        >>> # Load data
        >>> episodes = OfflineDatasetCollector.load_episodes("data.pkl")
        >>> train_loader = create_dataloader(episodes, context_len=20, ...)
        >>>
        >>> # Create model
        >>> model = MultiDiscreteDecisionTransformer(...)
        >>>
        >>> # Train
        >>> config = TrainingConfig(num_epochs=100, batch_size=64)
        >>> trainer = DecisionTransformerTrainer(model, config)
        >>> trainer.train(train_loader)
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        eval_env=None,
    ):
        """
        Initialize trainer.

        Args:
            model: Decision Transformer model
            config: Training configuration
            eval_env: Optional environment for evaluation
        """
        self.model = model
        self.config = config
        self.eval_env = eval_env

        # Move model to device
        self.device = torch.device(config.device)
        self.model.to(self.device)

        # Optimizer with weight decay
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999),
        )

        # Learning rate scheduler with warmup
        self.scheduler = self._create_scheduler()

        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_loss = float("inf")

        # Checkpoint directory
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Initialize WandB if requested
        if config.use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                config=config.to_dict(),
                name=f"dt-{config.hidden_size}h-{config.n_layer}l",
            )
            wandb.watch(self.model, log="all", log_freq=100)

        logger.info(f"Initialized trainer with device: {self.device}")
        logger.info(f"Model parameters: {self._count_parameters():,}")

    def _create_scheduler(self):
        """Create learning rate scheduler with warmup."""

        def lr_lambda(step):
            if step < self.config.warmup_steps:
                return step / max(1, self.config.warmup_steps)
            return 1.0

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def _count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        """
        Main training loop.

        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
        """
        logger.info(f"Starting training for {self.config.num_epochs} epochs")

        for epoch in range(self.config.num_epochs):
            self.epoch = epoch

            # Training
            train_metrics = self.train_epoch(train_loader)

            # Validation
            if val_loader is not None and (epoch + 1) % self.config.eval_every == 0:
                val_metrics = self.validate(val_loader)
                metrics = {**train_metrics, **val_metrics}
            else:
                metrics = train_metrics

            # Evaluation in environment
            if self.eval_env is not None and (epoch + 1) % self.config.eval_every == 0:
                eval_metrics = self.evaluate_in_env(num_episodes=5)
                metrics.update(eval_metrics)

            # Logging
            self._log_metrics(metrics, epoch)

            # Checkpointing
            if (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pt")

            # Save best model
            if val_loader is not None and val_metrics["val_loss"] < self.best_loss:
                self.best_loss = val_metrics["val_loss"]
                self.save_checkpoint("best_model.pt")

        logger.info("Training complete")

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader

        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        metrics = defaultdict(list)

        pbar = tqdm(train_loader, desc=f"Epoch {self.epoch+1}")

        for batch in pbar:
            # Move batch to device
            batch = self._batch_to_device(batch)

            # Forward pass
            loss, loss_dict = self.compute_loss(batch)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

            # Optimizer step
            self.optimizer.step()
            self.scheduler.step()

            # Track metrics
            metrics["loss"].append(loss.item())
            for key, value in loss_dict.items():
                metrics[key].append(value)

            self.global_step += 1

            # Update progress bar
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            # Log to WandB
            if self.config.use_wandb and WANDB_AVAILABLE:
                if self.global_step % self.config.log_every == 0:
                    wandb.log(
                        {
                            "train/loss": loss.item(),
                            "train/lr": self.scheduler.get_last_lr()[0],
                            **{f"train/{k}": v for k, v in loss_dict.items()},
                        },
                        step=self.global_step,
                    )

        # Aggregate metrics
        return {f"train_{k}": np.mean(v) for k, v in metrics.items()}

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate on validation set.

        Args:
            val_loader: Validation data loader

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        metrics = defaultdict(list)

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                batch = self._batch_to_device(batch)
                loss, loss_dict = self.compute_loss(batch)

                metrics["loss"].append(loss.item())
                for key, value in loss_dict.items():
                    metrics[key].append(value)

        return {f"val_{k}": np.mean(v) for k, v in metrics.items()}

    def compute_loss(self, batch: Dict) -> tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute loss for a batch.

        Args:
            batch: Batch dictionary

        Returns:
            Tuple of (total loss, loss components dictionary)
        """
        # Extract batch components
        returns = batch["returns"]
        states = batch["states"]
        aircraft_masks = batch["aircraft_masks"]
        actions = batch["actions"]
        timesteps = batch["timesteps"]
        attention_mask = batch["attention_mask"]

        # Forward pass
        action_preds = self.model(
            returns=returns,
            states=states,
            aircraft_masks=aircraft_masks,
            actions=actions,
            timesteps=timesteps,
            attention_mask=attention_mask,
        )

        # Compute loss for each action component
        losses = {}
        total_loss = 0.0

        for key, logits in action_preds.items():
            # logits: (batch_size, seq_len, action_dim)
            # targets: (batch_size, seq_len)
            targets = actions[key]

            # Flatten for cross-entropy
            logits_flat = logits.reshape(-1, logits.size(-1))
            targets_flat = targets.reshape(-1)

            # Compute cross-entropy loss
            loss = F.cross_entropy(logits_flat, targets_flat, reduction="none")

            # Apply attention mask
            mask_flat = attention_mask.reshape(-1)
            loss = (loss * mask_flat).sum() / (mask_flat.sum() + 1e-6)

            losses[f"loss_{key}"] = loss.item()
            total_loss += loss

        losses["total_loss"] = total_loss.item()

        return total_loss, losses

    def evaluate_in_env(
        self,
        num_episodes: int = 10,
        target_returns: Optional[List[float]] = None,
        temperature: float = 1.0,
        deterministic: bool = False,
    ) -> Dict[str, float]:
        """
        Evaluate model in environment.

        Args:
            num_episodes: Number of episodes to run
            target_returns: List of target returns to condition on
            temperature: Sampling temperature
            deterministic: Use greedy action selection

        Returns:
            Dictionary of evaluation metrics
        """
        if self.eval_env is None:
            return {}

        if target_returns is None:
            # Use different return targets
            target_returns = [50.0, 100.0, 200.0]

        metrics = defaultdict(list)

        for target_return in target_returns:
            episode_returns = []

            for _ in range(num_episodes):
                episode_return = self._run_episode(
                    target_return=target_return,
                    temperature=temperature,
                    deterministic=deterministic,
                )
                episode_returns.append(episode_return)

            metrics[f"eval_return_target_{int(target_return)}"] = np.mean(episode_returns)

        return dict(metrics)

    def _run_episode(
        self,
        target_return: float,
        temperature: float = 1.0,
        deterministic: bool = False,
        max_steps: int = 1000,
    ) -> float:
        """
        Run a single episode with the model.

        Args:
            target_return: Target return to condition on
            temperature: Sampling temperature
            deterministic: Use greedy action selection
            max_steps: Maximum steps

        Returns:
            Total episode return
        """
        self.model.eval()

        obs, info = self.eval_env.reset()

        # Initialize context buffers
        returns_buffer = []
        states_buffer = []
        masks_buffer = []
        actions_buffer = {key: [] for key in ["aircraft_id", "command_type", "altitude", "heading", "speed"]}
        timesteps_buffer = []

        episode_return = 0.0
        current_return = target_return / self.config.return_scale

        with torch.no_grad():
            for step in range(max_steps):
                # Add current state to buffer
                returns_buffer.append(current_return)
                states_buffer.append(obs["aircraft"])
                masks_buffer.append(obs["aircraft_mask"])
                timesteps_buffer.append(step)

                # Truncate to context length
                if len(returns_buffer) > self.config.context_len:
                    returns_buffer = returns_buffer[-self.config.context_len :]
                    states_buffer = states_buffer[-self.config.context_len :]
                    masks_buffer = masks_buffer[-self.config.context_len :]
                    timesteps_buffer = timesteps_buffer[-self.config.context_len :]
                    for key in actions_buffer.keys():
                        actions_buffer[key] = actions_buffer[key][-self.config.context_len :]

                # Prepare tensors
                returns_tensor = torch.tensor(returns_buffer, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
                states_tensor = torch.tensor(np.stack(states_buffer), dtype=torch.float32).unsqueeze(0)
                masks_tensor = torch.tensor(np.stack(masks_buffer), dtype=torch.bool).unsqueeze(0)
                timesteps_tensor = torch.tensor(timesteps_buffer, dtype=torch.long).unsqueeze(0)

                # Pad actions if needed
                if len(actions_buffer["aircraft_id"]) == 0:
                    # First step - use dummy action
                    actions_tensor = {
                        key: torch.zeros(1, 1, dtype=torch.long)
                        for key in actions_buffer.keys()
                    }
                else:
                    actions_tensor = {
                        key: torch.tensor(values, dtype=torch.long).unsqueeze(0)
                        for key, values in actions_buffer.items()
                    }

                # Move to device
                returns_tensor = returns_tensor.to(self.device)
                states_tensor = states_tensor.to(self.device)
                masks_tensor = masks_tensor.to(self.device)
                actions_tensor = {k: v.to(self.device) for k, v in actions_tensor.items()}
                timesteps_tensor = timesteps_tensor.to(self.device)

                # Get action from model
                action_dict, _ = self.model.get_action(
                    returns=returns_tensor,
                    states=states_tensor,
                    aircraft_masks=masks_tensor,
                    actions=actions_tensor,
                    timesteps=timesteps_tensor,
                    temperature=temperature,
                    deterministic=deterministic,
                )

                # Convert to environment format
                action = {
                    "aircraft_id": int(action_dict["aircraft_id"]),
                    "command_type": int(action_dict["command_type"]),
                    "altitude": int(action_dict["altitude"]),
                    "heading": int(action_dict["heading"]),
                    "speed": int(action_dict["speed"]),
                }

                # Step environment
                obs, reward, terminated, truncated, info = self.eval_env.step(action)

                # Update buffers
                for key, value in action.items():
                    actions_buffer[key].append(value)

                episode_return += reward

                # Update return-to-go
                current_return -= reward / self.config.return_scale

                if terminated or truncated:
                    break

        return episode_return

    def _batch_to_device(self, batch: Dict) -> Dict:
        """Move batch to device."""
        device_batch = {}

        for key, value in batch.items():
            if key == "actions":
                device_batch[key] = {k: v.to(self.device) for k, v in value.items()}
            else:
                device_batch[key] = value.to(self.device)

        return device_batch

    def _log_metrics(self, metrics: Dict[str, float], epoch: int):
        """Log metrics to console and WandB."""
        # Console logging
        logger.info(f"Epoch {epoch+1}:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value:.4f}")

        # WandB logging
        if self.config.use_wandb and WANDB_AVAILABLE:
            wandb.log(metrics, step=epoch)

    def save_checkpoint(self, filename: str):
        """
        Save model checkpoint.

        Args:
            filename: Checkpoint filename
        """
        checkpoint_path = self.checkpoint_dir / filename

        checkpoint = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_loss": self.best_loss,
            "config": self.config.to_dict(),
        }

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_loss = checkpoint["best_loss"]

        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        logger.info(f"  Epoch: {self.epoch}")
        logger.info(f"  Global step: {self.global_step}")
