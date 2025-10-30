"""
Behavioral cloning pre-training module.

This module implements supervised learning from expert demonstrations using
cross-entropy loss. The pre-trained model provides a good initialization
for subsequent RL fine-tuning, significantly improving sample efficiency.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pickle
from tqdm import tqdm

from models.networks import ATCActorCritic
from models.config import create_default_network_config, NetworkConfig
from training.rule_based_expert import Demonstration, load_demonstrations
from environment.utils import get_device


logger = logging.getLogger(__name__)


class DemonstrationDataset(Dataset):
    """
    PyTorch dataset for expert demonstrations.

    This dataset loads and provides expert (observation, action) pairs
    for supervised learning.
    """

    def __init__(self, demonstrations: List[Demonstration], device: str = "cpu"):
        """
        Initialize demonstration dataset.

        Args:
            demonstrations: List of expert demonstrations
            device: Device to place tensors on
        """
        self.device = device
        self.observations = []
        self.actions = []

        # Extract all (obs, action) pairs from demonstrations
        for demo in demonstrations:
            for obs, action in zip(demo.observations, demo.actions):
                self.observations.append(obs)
                self.actions.append(action)

        logger.info(f"Dataset created with {len(self.observations)} transitions")

    def __len__(self) -> int:
        return len(self.observations)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Get a single (observation, action) pair.

        Args:
            idx: Index of the transition

        Returns:
            Tuple of (observation dict, action tensor)
        """
        obs = self.observations[idx]
        action = self.actions[idx]

        # Convert to tensors
        obs_tensor = {
            'aircraft': torch.from_numpy(obs['aircraft']).float().to(self.device),
            'aircraft_mask': torch.from_numpy(obs['aircraft_mask']).bool().to(self.device),
            'global_state': torch.from_numpy(obs['global_state']).float().to(self.device),
            'conflict_matrix': torch.from_numpy(obs['conflict_matrix']).float().to(self.device),
        }

        action_tensor = torch.from_numpy(action).long().to(self.device)

        return obs_tensor, action_tensor


class BehavioralCloningTrainer:
    """
    Trainer for behavioral cloning from expert demonstrations.

    This trainer uses cross-entropy loss to train a policy network to
    imitate expert actions given observations.

    Example:
        >>> # Load demonstrations
        >>> demos = load_demonstrations("data/expert_demonstrations.pkl")
        >>>
        >>> # Create trainer
        >>> trainer = BehavioralCloningTrainer(
        ...     demonstrations=demos,
        ...     network_config=create_default_network_config(max_aircraft=5)
        ... )
        >>>
        >>> # Train model
        >>> trainer.train(num_epochs=50, batch_size=64)
        >>>
        >>> # Save trained model
        >>> trainer.save_model("checkpoints/bc_pretrained.pth")
    """

    def __init__(
        self,
        demonstrations: List[Demonstration],
        network_config: Optional[NetworkConfig] = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        device: Optional[str] = None,
    ):
        """
        Initialize behavioral cloning trainer.

        Args:
            demonstrations: List of expert demonstrations
            network_config: Network configuration (uses default if None)
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
            device: Device to train on
        """
        self.device = get_device(device)
        self.demonstrations = demonstrations

        # Create network
        if network_config is None:
            network_config = create_default_network_config()

        self.model = ATCActorCritic(network_config).to(device)

        # Create dataset and dataloader
        self.dataset = DemonstrationDataset(demonstrations, device=device)

        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Loss function (cross-entropy for each action component)
        self.loss_fn = nn.CrossEntropyLoss()

        # Training history
        self.train_losses = []
        self.val_losses = []

        logger.info(f"Initialized BC trainer with {len(self.dataset)} transitions")
        logger.info(f"Using device: {device}")

    def train(
        self,
        num_epochs: int = 50,
        batch_size: int = 64,
        validation_split: float = 0.1,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Train the model using behavioral cloning.

        Args:
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of data to use for validation
            verbose: Whether to print progress

        Returns:
            Dictionary with training and validation losses
        """
        # Split into train and validation
        dataset_size = len(self.dataset)
        val_size = int(dataset_size * validation_split)
        train_size = dataset_size - val_size

        train_dataset, val_dataset = torch.utils.data.random_split(
            self.dataset, [train_size, val_size]
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
        )

        logger.info(f"Training on {train_size} samples, validating on {val_size} samples")

        # Training loop
        for epoch in range(num_epochs):
            # Training phase
            train_loss = self._train_epoch(train_loader)
            self.train_losses.append(train_loss)

            # Validation phase
            val_loss = self._validate_epoch(val_loader)
            self.val_losses.append(val_loss)

            if verbose and (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{num_epochs} - "
                           f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }

    def _train_epoch(self, dataloader: DataLoader) -> float:
        """
        Train for one epoch.

        Args:
            dataloader: Training data loader

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for obs_batch, action_batch in dataloader:
            # Forward pass
            action_logits, _ = self.model(obs_batch)

            # Compute loss for each action component
            loss = 0.0

            # Action components: [aircraft_id, command_type, altitude, heading, speed]
            action_names = ['aircraft_id', 'command_type', 'altitude', 'heading', 'speed']

            for i, name in enumerate(action_names):
                if name in action_logits:
                    target = action_batch[:, i]
                    loss += self.loss_fn(action_logits[name], target)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def _validate_epoch(self, dataloader: DataLoader) -> float:
        """
        Validate for one epoch.

        Args:
            dataloader: Validation data loader

        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for obs_batch, action_batch in dataloader:
                # Forward pass
                action_logits, _ = self.model(obs_batch)

                # Compute loss for each action component
                loss = 0.0
                action_names = ['aircraft_id', 'command_type', 'altitude', 'heading', 'speed']

                for i, name in enumerate(action_names):
                    if name in action_logits:
                        target = action_batch[:, i]
                        loss += self.loss_fn(action_logits[name], target)

                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches

    def save_model(self, save_path: str) -> None:
        """
        Save trained model to disk.

        Args:
            save_path: Path to save model checkpoint
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.model.config,
        }

        torch.save(checkpoint, save_path)
        logger.info(f"Model saved to {save_path}")

    def load_model(self, load_path: str) -> None:
        """
        Load trained model from disk.

        Args:
            load_path: Path to model checkpoint
        """
        checkpoint = torch.load(load_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']

        logger.info(f"Model loaded from {load_path}")

    def evaluate_accuracy(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Evaluate action prediction accuracy.

        Args:
            dataloader: Data loader for evaluation

        Returns:
            Dictionary with accuracy for each action component
        """
        self.model.eval()
        action_names = ['aircraft_id', 'command_type', 'altitude', 'heading', 'speed']
        correct_counts = {name: 0 for name in action_names}
        total_count = 0

        with torch.no_grad():
            for obs_batch, action_batch in dataloader:
                # Forward pass
                action_logits, _ = self.model(obs_batch)

                # Check accuracy for each component
                for i, name in enumerate(action_names):
                    if name in action_logits:
                        predictions = torch.argmax(action_logits[name], dim=1)
                        targets = action_batch[:, i]
                        correct_counts[name] += (predictions == targets).sum().item()

                total_count += len(action_batch)

        # Calculate accuracies
        accuracies = {name: count / total_count for name, count in correct_counts.items()}

        return accuracies


def train_bc_model(
    demonstrations_path: str,
    save_path: str,
    num_epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    network_config: Optional[NetworkConfig] = None,
) -> BehavioralCloningTrainer:
    """
    Train a behavioral cloning model from demonstrations.

    Args:
        demonstrations_path: Path to expert demonstrations
        save_path: Path to save trained model
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        network_config: Network configuration

    Returns:
        Trained BC trainer
    """
    # Load demonstrations
    demonstrations = load_demonstrations(demonstrations_path)

    # Create trainer
    trainer = BehavioralCloningTrainer(
        demonstrations=demonstrations,
        network_config=network_config,
        learning_rate=learning_rate,
    )

    # Train model
    logger.info(f"Training BC model for {num_epochs} epochs...")
    history = trainer.train(
        num_epochs=num_epochs,
        batch_size=batch_size,
        verbose=True,
    )

    # Save model
    trainer.save_model(save_path)

    # Evaluate accuracy
    val_loader = DataLoader(trainer.dataset, batch_size=batch_size, shuffle=False)
    accuracies = trainer.evaluate_accuracy(val_loader)

    logger.info("\nAction prediction accuracies:")
    for name, acc in accuracies.items():
        logger.info(f"  {name}: {acc:.2%}")

    return trainer


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Train BC model
    trainer = train_bc_model(
        demonstrations_path="data/expert_demonstrations.pkl",
        save_path="checkpoints/bc_pretrained.pth",
        num_epochs=100,
        batch_size=64,
        learning_rate=1e-3,
    )
