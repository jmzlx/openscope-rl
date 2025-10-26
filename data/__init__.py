"""
Data collection modules for OpenScope RL training.

This package contains data collectors for various training approaches:
- cosmos_collector: Collect video + state data for Cosmos world model training
"""

from .cosmos_collector import CosmosDataCollector, EpisodeData, CosmosDataset

__all__ = ["CosmosDataCollector", "EpisodeData", "CosmosDataset"]
