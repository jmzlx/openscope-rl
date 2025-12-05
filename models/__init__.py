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

# Hierarchical policy
try:
    from .hierarchical_policy import (
        HierarchicalPolicy,
        HierarchicalPolicyConfig,
        HighLevelPolicyConfig,
        LowLevelPolicyConfig,
        create_hierarchical_policy,
    )
except ImportError:
    pass

# Multi-agent policy
try:
    from .multi_agent_policy import (
        MultiAgentPolicy,
        CommunicationModule,
        LocalObservationEncoder,
        DecentralizedActor,
        CentralizedCritic,
    )
except ImportError:
    pass

# Decision Transformer
try:
    from .decision_transformer import (
        DecisionTransformer,
        MultiDiscreteDecisionTransformer,
    )
except ImportError:
    pass

# Trajectory Transformer
try:
    from .trajectory_transformer import (
        TrajectoryTransformer,
        TrajectoryTransformerConfig,
    )
except ImportError:
    pass

# TD-MPC 2
try:
    from .tdmpc2 import (
        TDMPC2Model,
        TDMPC2Config,
        LatentEncoder,
        TransformerDynamics,
        RewardPredictor,
        TDMPC2QNetwork,
    )
except ImportError:
    pass

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
    
    # Hierarchical policy
    "HierarchicalPolicy",
    "HierarchicalPolicyConfig",
    "HighLevelPolicyConfig",
    "LowLevelPolicyConfig",
    "create_hierarchical_policy",
    
    # Multi-agent policy
    "MultiAgentPolicy",
    "CommunicationModule",
    "LocalObservationEncoder",
    "DecentralizedActor",
    "CentralizedCritic",
    
    # Decision Transformer
    "DecisionTransformer",
    "MultiDiscreteDecisionTransformer",
    
    # Trajectory Transformer
    "TrajectoryTransformer",
    "TrajectoryTransformerConfig",
    
    # TD-MPC 2
    "TDMPC2Model",
    "TDMPC2Config",
    "LatentEncoder",
    "TransformerDynamics",
    "RewardPredictor",
    "TDMPC2QNetwork",
]

# Version info
__version__ = "0.1.0"