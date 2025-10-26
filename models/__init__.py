"""
Models module for OpenScope RL.

This module provides neural network architectures and supporting components
for reinforcement learning with the OpenScope air traffic control simulator.
"""

from .networks import ATCActorCritic, create_model, create_default_model
from .config import (
    NetworkConfig,
    EncoderConfig,
    AttentionPoolingConfig,
    PolicyHeadConfig,
    ValueHeadConfig,
    create_default_network_config,
    validate_network_config,
    get_model_info,
)
from .encoders import ATCTransformerEncoder, GlobalStateEncoder, AttentionPooling
from .heads import PolicyHead, ValueHead, ActorCriticHeads
from .trajectory_transformer import (
    TrajectoryTransformer,
    TrajectoryTransformerConfig,
    create_trajectory_transformer,
)

# Public API
__all__ = [
    # Main model
    "ATCActorCritic",
    "create_model",
    "create_default_model",

    # Configuration
    "NetworkConfig",
    "EncoderConfig",
    "AttentionPoolingConfig",
    "PolicyHeadConfig",
    "ValueHeadConfig",
    "create_default_network_config",
    "validate_network_config",
    "get_model_info",

    # Encoders
    "ATCTransformerEncoder",
    "GlobalStateEncoder",
    "AttentionPooling",

    # Heads
    "PolicyHead",
    "ValueHead",
    "ActorCriticHeads",

    # Trajectory Transformer
    "TrajectoryTransformer",
    "TrajectoryTransformerConfig",
    "create_trajectory_transformer",
]

# Version info
__version__ = "0.1.0"