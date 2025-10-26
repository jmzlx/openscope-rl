"""
Training modules for OpenScope RL.

This package contains training scripts and utilities:
- cosmos_finetuner: Fine-tune NVIDIA Cosmos on OpenScope data
- cosmos_rl_trainer: Train RL policies in Cosmos-simulated environment
"""

from .cosmos_finetuner import CosmosFineTuner, CosmosTrainingConfig
from .cosmos_rl_trainer import CosmosRLTrainer

__all__ = ["CosmosFineTuner", "CosmosTrainingConfig", "CosmosRLTrainer"]
