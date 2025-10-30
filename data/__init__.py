"""
Data collection modules for OpenScope RL training.

This package contains data collectors for various training approaches:
- cosmos_collector: Collect video + state data for Cosmos world model training
- offline_dataset: Dataset utilities for offline RL (decision transformer, etc.)
- training_data_collector: Comprehensive training data extraction for all ATC models
"""

from .cosmos_collector import CosmosDataCollector, EpisodeData, CosmosDataset

# Offline dataset for decision transformer
try:
    from .offline_dataset import (
        OfflineDatasetCollector,
        OfflineRLDataset,
        Episode,
        create_dataloader,
    )
except ImportError:
    pass

# Training data collector
try:
    from .training_data_collector import (
        TrainingDataCollector,
        TrainingEpisode,
        TrainingTransition,
        load_training_episodes,
    )
except ImportError:
    pass

__all__ = [
    "CosmosDataCollector",
    "EpisodeData",
    "CosmosDataset",
    # Offline dataset
    "OfflineDatasetCollector",
    "OfflineRLDataset",
    "Episode",
    "create_dataloader",
    # Training data collector
    "TrainingDataCollector",
    "TrainingEpisode",
    "TrainingTransition",
    "load_training_episodes",
]
