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
from .hierarchical_policy import (
    HierarchicalPolicy,
    HierarchicalPolicyConfig,
    HighLevelPolicyConfig,
    LowLevelPolicyConfig,
    create_hierarchical_policy,
)

# Public API
__all__ = [
    # Main model
    "ATCActorCritic",
    "create_model",
    "create_default_model",

    # Hierarchical model
    "HierarchicalPolicy",
    "HierarchicalPolicyConfig",
    "HighLevelPolicyConfig",
    "LowLevelPolicyConfig",
    "create_hierarchical_policy",

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
]

# Version info
__version__ = "0.1.0"