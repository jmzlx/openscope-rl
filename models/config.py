"""
Configuration module for neural network models.

This module defines dataclasses for network architecture configuration,
providing type safety and validation for model parameters.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum


class ActivationType(Enum):
    """Available activation functions."""
    RELU = "relu"
    GELU = "gelu"
    TANH = "tanh"
    SWISH = "swish"


class AttentionType(Enum):
    """Available attention mechanisms."""
    MULTIHEAD = "multihead"
    SELF = "self"
    CROSS = "cross"


@dataclass
class EncoderConfig:
    """Configuration for transformer encoder."""
    
    # Architecture
    input_dim: int = 14
    hidden_dim: int = 256
    num_heads: int = 8
    num_layers: int = 4
    dropout: float = 0.1
    
    # Activation
    activation: ActivationType = ActivationType.GELU
    
    # Normalization
    use_layer_norm: bool = True
    norm_first: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        
        if not (0.0 <= self.dropout <= 1.0):
            raise ValueError("dropout must be in [0.0, 1.0]")
        
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")


@dataclass
class AttentionPoolingConfig:
    """Configuration for attention pooling."""
    
    hidden_dim: int = 256
    num_heads: int = 8
    dropout: float = 0.1
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        
        if not (0.0 <= self.dropout <= 1.0):
            raise ValueError("dropout must be in [0.0, 1.0]")


@dataclass
class PolicyHeadConfig:
    """Configuration for policy head."""
    
    # Architecture
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.1
    
    # Action space sizes
    aircraft_id_size: int = 21  # max_aircraft + 1
    command_type_size: int = 5
    altitude_size: int = 18
    heading_size: int = 13
    speed_size: int = 8
    
    # Activation
    activation: ActivationType = ActivationType.RELU
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        
        if not (0.0 <= self.dropout <= 1.0):
            raise ValueError("dropout must be in [0.0, 1.0]")
        
        if self.aircraft_id_size <= 0:
            raise ValueError("aircraft_id_size must be positive")
        
        if self.command_type_size <= 0:
            raise ValueError("command_type_size must be positive")
        
        if self.altitude_size <= 0:
            raise ValueError("altitude_size must be positive")
        
        if self.heading_size <= 0:
            raise ValueError("heading_size must be positive")
        
        if self.speed_size <= 0:
            raise ValueError("speed_size must be positive")


@dataclass
class ValueHeadConfig:
    """Configuration for value head."""
    
    # Architecture
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.1
    
    # Activation
    activation: ActivationType = ActivationType.RELU
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        
        if not (0.0 <= self.dropout <= 1.0):
            raise ValueError("dropout must be in [0.0, 1.0]")


@dataclass
class NetworkConfig:
    """Main configuration for ATCActorCritic network."""
    
    # Model dimensions
    aircraft_feature_dim: int = 14
    global_feature_dim: int = 4
    max_aircraft: int = 20
    
    # Component configurations
    encoder_config: EncoderConfig = field(default_factory=EncoderConfig)
    attention_pooling_config: AttentionPoolingConfig = field(default_factory=AttentionPoolingConfig)
    policy_head_config: PolicyHeadConfig = field(default_factory=PolicyHeadConfig)
    value_head_config: ValueHeadConfig = field(default_factory=ValueHeadConfig)
    
    # Training
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    
    # Initialization
    init_method: str = "xavier_uniform"
    init_gain: float = 1.0
    
    def __post_init__(self):
        """Validate configuration consistency."""
        # Update policy head aircraft_id_size to match max_aircraft
        self.policy_head_config.aircraft_id_size = self.max_aircraft + 1
        
        # Validate dimensions
        if self.aircraft_feature_dim <= 0:
            raise ValueError("aircraft_feature_dim must be positive")
        
        if self.global_feature_dim <= 0:
            raise ValueError("global_feature_dim must be positive")
        
        if self.max_aircraft <= 0:
            raise ValueError("max_aircraft must be positive")
        
        if not (0.0 < self.learning_rate <= 1.0):
            raise ValueError("learning_rate must be in (0.0, 1.0]")
        
        if not (0.0 <= self.weight_decay <= 1.0):
            raise ValueError("weight_decay must be in [0.0, 1.0]")


def create_default_network_config(**overrides) -> NetworkConfig:
    """
    Create default network configuration with optional overrides.
    
    Args:
        **overrides: Configuration values to override
        
    Returns:
        NetworkConfig: Configured network settings
        
    Example:
        >>> config = create_default_network_config(
        ...     max_aircraft=10,
        ...     hidden_dim=128,
        ...     num_layers=2
        ... )
    """
    config = NetworkConfig()
    
    # Apply overrides
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
        elif hasattr(config.encoder_config, key):
            setattr(config.encoder_config, key, value)
        elif hasattr(config.attention_pooling_config, key):
            setattr(config.attention_pooling_config, key, value)
        elif hasattr(config.policy_head_config, key):
            setattr(config.policy_head_config, key, value)
        elif hasattr(config.value_head_config, key):
            setattr(config.value_head_config, key, value)
        else:
            raise ValueError(f"Unknown configuration key: {key}")
    
    return config


def validate_network_config(config: NetworkConfig) -> bool:
    """
    Validate network configuration for consistency and correctness.
    
    Args:
        config: Configuration to validate
        
    Returns:
        bool: True if configuration is valid
        
    Raises:
        ValueError: If configuration is invalid
    """
    # Validate encoder config
    if config.encoder_config.input_dim != config.aircraft_feature_dim:
        raise ValueError("encoder_config.input_dim must match aircraft_feature_dim")
    
    # Validate attention pooling config
    if config.attention_pooling_config.hidden_dim != config.encoder_config.hidden_dim:
        raise ValueError("attention_pooling_config.hidden_dim must match encoder_config.hidden_dim")
    
    # Validate policy head config
    if config.policy_head_config.hidden_dim != config.encoder_config.hidden_dim:
        raise ValueError("policy_head_config.hidden_dim must match encoder_config.hidden_dim")
    
    # Validate value head config
    if config.value_head_config.hidden_dim != config.encoder_config.hidden_dim:
        raise ValueError("value_head_config.hidden_dim must match encoder_config.hidden_dim")
    
    return True


def get_model_info(config: NetworkConfig) -> Dict[str, Any]:
    """
    Get information about model architecture.
    
    Args:
        config: Network configuration
        
    Returns:
        Dictionary with model information
    """
    return {
        "model_type": "ATCActorCritic",
        "aircraft_feature_dim": config.aircraft_feature_dim,
        "global_feature_dim": config.global_feature_dim,
        "max_aircraft": config.max_aircraft,
        "encoder": {
            "input_dim": config.encoder_config.input_dim,
            "hidden_dim": config.encoder_config.hidden_dim,
            "num_heads": config.encoder_config.num_heads,
            "num_layers": config.encoder_config.num_layers,
            "dropout": config.encoder_config.dropout,
        },
        "attention_pooling": {
            "hidden_dim": config.attention_pooling_config.hidden_dim,
            "num_heads": config.attention_pooling_config.num_heads,
            "dropout": config.attention_pooling_config.dropout,
        },
        "policy_head": {
            "hidden_dim": config.policy_head_config.hidden_dim,
            "num_layers": config.policy_head_config.num_layers,
            "dropout": config.policy_head_config.dropout,
            "action_space_sizes": {
                "aircraft_id": config.policy_head_config.aircraft_id_size,
                "command_type": config.policy_head_config.command_type_size,
                "altitude": config.policy_head_config.altitude_size,
                "heading": config.policy_head_config.heading_size,
                "speed": config.policy_head_config.speed_size,
            }
        },
        "value_head": {
            "hidden_dim": config.value_head_config.hidden_dim,
            "num_layers": config.value_head_config.num_layers,
            "dropout": config.value_head_config.dropout,
        },
        "training": {
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
        }
    }
