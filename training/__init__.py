"""
Training module for multi-agent reinforcement learning.

This module provides training algorithms and utilities for multi-agent RL,
including MAPPO (Multi-Agent PPO) for cooperative control.
"""

from .mappo_trainer import MAPPOTrainer, MAPPOConfig, train_mappo

__all__ = [
    "MAPPOTrainer",
    "MAPPOConfig",
    "train_mappo",
]

__version__ = "0.1.0"
