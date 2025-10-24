"""
Refactored neural network architectures for OpenScope RL agent.

This module provides a clean, modular implementation of the ATCActorCritic network
using separated concerns for encoders, heads, and configuration.
"""

import logging
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn

from .config import NetworkConfig, create_default_network_config, validate_network_config
from .encoders import ATCTransformerEncoder, GlobalStateEncoder, AttentionPooling
from .heads import ActorCriticHeads


logger = logging.getLogger(__name__)


class ATCActorCritic(nn.Module):
    """
    Combined actor-critic network for PPO with shared encoder.
    
    This network uses a transformer-based architecture to handle variable numbers
    of aircraft and provides both policy and value estimates for actor-critic
    reinforcement learning algorithms.
    
    Architecture:
    - Shared transformer encoder for aircraft
    - Shared global state encoder
    - Attention pooling for variable aircraft count
    - Separate policy and value heads
    
    Example:
        >>> config = create_default_network_config(max_aircraft=10)
        >>> model = ATCActorCritic(config)
        >>> obs = {
        ...     "aircraft": torch.randn(2, 10, 14),
        ...     "aircraft_mask": torch.ones(2, 10, dtype=torch.bool),
        ...     "global_state": torch.randn(2, 4),
        ...     "conflict_matrix": torch.randn(2, 10, 10)
        ... }
        >>> action_logits, value = model(obs)
        >>> print(action_logits["aircraft_id"].shape)  # torch.Size([2, 11])
        >>> print(value.shape)  # torch.Size([2, 1])
    """
    
    def __init__(self, config: Optional[NetworkConfig] = None):
        """
        Initialize ATCActorCritic network.
        
        Args:
            config: Network configuration (uses default if None)
            
        Raises:
            ValueError: If configuration is invalid
        """
        super().__init__()
        
        # Use default config if none provided
        if config is None:
            config = create_default_network_config()
        
        # Validate configuration
        validate_network_config(config)
        
        self.config = config
        self.max_aircraft = config.max_aircraft
        self.aircraft_feature_dim = config.aircraft_feature_dim
        self.global_feature_dim = config.global_feature_dim
        
        # Initialize encoders
        self.aircraft_encoder = ATCTransformerEncoder(config.encoder_config)
        self.global_encoder = GlobalStateEncoder(
            input_dim=self.global_feature_dim,
            hidden_dim=config.encoder_config.hidden_dim,
            num_layers=2,
            dropout=config.encoder_config.dropout,
            activation=config.encoder_config.activation.value
        )
        
        # Attention pooling
        self.attention_pooling = AttentionPooling(config.attention_pooling_config)
        
        # Combined feature dimension
        combined_dim = config.encoder_config.hidden_dim + config.encoder_config.hidden_dim
        
        # Shared feature layers
        self.shared_layers = nn.Sequential(
            nn.Linear(combined_dim, config.encoder_config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.encoder_config.hidden_dim, config.encoder_config.hidden_dim),
            nn.ReLU(),
        )
        
        # Actor-critic heads
        self.heads = ActorCriticHeads(config.policy_head_config, config.value_head_config)
        
        # Initialize weights
        self._init_weights()
        
        logger.info(f"ATCActorCritic initialized: {self.max_aircraft} aircraft, "
                   f"{config.encoder_config.hidden_dim} hidden dim")
    
    def _init_weights(self) -> None:
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if self.config.init_method == "xavier_uniform":
                    nn.init.xavier_uniform_(module.weight, gain=self.config.init_gain)
                elif self.config.init_method == "xavier_normal":
                    nn.init.xavier_normal_(module.weight, gain=self.config.init_gain)
                elif self.config.init_method == "kaiming_uniform":
                    nn.init.kaiming_uniform_(module.weight)
                elif self.config.init_method == "kaiming_normal":
                    nn.init.kaiming_normal_(module.weight)
                else:
                    raise ValueError(f"Unknown init method: {self.config.init_method}")
                
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def _encode_state(self, obs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode observation through shared encoder.
        
        Args:
            obs: Observation dictionary containing aircraft, aircraft_mask, global_state
            
        Returns:
            Tuple of (shared_features, aircraft_encoded)
        """
        aircraft = obs["aircraft"]
        aircraft_mask = obs["aircraft_mask"]
        global_state = obs["global_state"]
        
        # Encode aircraft through transformer
        aircraft_encoded = self.aircraft_encoder(aircraft, aircraft_mask)
        
        # Encode global state
        global_encoded = self.global_encoder(global_state)
        
        # Pool aircraft features using attention
        aircraft_pooled = self.attention_pooling(aircraft_encoded, aircraft_mask)
        
        # Combine features
        combined = torch.cat([aircraft_pooled, global_encoded], dim=-1)
        shared_features = self.shared_layers(combined)
        
        return shared_features, aircraft_encoded
    
    def forward(self, obs: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Forward pass through both policy and value networks.
        
        Args:
            obs: Observation dictionary containing:
                - aircraft: (batch_size, max_aircraft, aircraft_feature_dim)
                - aircraft_mask: (batch_size, max_aircraft)
                - global_state: (batch_size, global_feature_dim)
                - conflict_matrix: (batch_size, max_aircraft, max_aircraft) [unused]
        
        Returns:
            Tuple of (action_logits, value)
        """
        # Validate input
        self._validate_observation(obs)
        
        # Encode state
        shared_features, aircraft_encoded = self._encode_state(obs)
        
        # Get action logits and value
        action_logits, value = self.heads(shared_features, aircraft_encoded, obs["aircraft_mask"])
        
        return action_logits, value
    
    def get_action_and_value(
        self, 
        obs: Dict[str, torch.Tensor], 
        action: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action or compute log prob of given action.
        
        Args:
            obs: Observation dictionary
            action: Optional action dictionary for computing log probabilities
            
        Returns:
            Tuple of (action, log_prob, entropy, value)
        """
        logits, value = self(obs)
        
        # Create distributions for each action component
        aircraft_dist = torch.distributions.Categorical(logits=logits["aircraft_id"])
        command_dist = torch.distributions.Categorical(logits=logits["command_type"])
        altitude_dist = torch.distributions.Categorical(logits=logits["altitude"])
        heading_dist = torch.distributions.Categorical(logits=logits["heading"])
        speed_dist = torch.distributions.Categorical(logits=logits["speed"])
        
        # Sample or use provided action
        if action is None:
            action = {
                "aircraft_id": aircraft_dist.sample(),
                "command_type": command_dist.sample(),
                "altitude": altitude_dist.sample(),
                "heading": heading_dist.sample(),
                "speed": speed_dist.sample(),
            }
        
        # Compute log probabilities
        log_prob = (
            aircraft_dist.log_prob(action["aircraft_id"]) +
            command_dist.log_prob(action["command_type"]) +
            altitude_dist.log_prob(action["altitude"]) +
            heading_dist.log_prob(action["heading"]) +
            speed_dist.log_prob(action["speed"])
        )
        
        # Compute entropy
        entropy = (
            aircraft_dist.entropy() +
            command_dist.entropy() +
            altitude_dist.entropy() +
            heading_dist.entropy() +
            speed_dist.entropy()
        )
        
        return action, log_prob, entropy, value.squeeze(-1)
    
    def _validate_observation(self, obs: Dict[str, torch.Tensor]) -> None:
        """
        Validate observation structure and shapes.
        
        Args:
            obs: Observation dictionary
            
        Raises:
            ValueError: If observation is invalid
        """
        required_keys = ["aircraft", "aircraft_mask", "global_state"]
        
        for key in required_keys:
            if key not in obs:
                raise ValueError(f"Missing observation key: {key}")
        
        # Validate shapes
        batch_size = obs["aircraft"].size(0)
        
        expected_shapes = {
            "aircraft": (batch_size, self.max_aircraft, self.aircraft_feature_dim),
            "aircraft_mask": (batch_size, self.max_aircraft),
            "global_state": (batch_size, self.global_feature_dim),
        }
        
        for key, expected_shape in expected_shapes.items():
            if obs[key].shape != expected_shape:
                raise ValueError(f"Invalid shape for {key}: expected {expected_shape}, "
                               f"got {obs[key].shape}")
    
    def get_action_space_sizes(self) -> Dict[str, int]:
        """Get action space sizes."""
        return self.heads.get_action_space_sizes()
    
    def get_model_info(self) -> Dict[str, any]:
        """Get model architecture information."""
        from .config import get_model_info
        return get_model_info(self.config)
    
    def count_parameters(self) -> Dict[str, int]:
        """
        Count model parameters.
        
        Returns:
            Dictionary with parameter counts
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "non_trainable_parameters": total_params - trainable_params,
        }


# Backward compatibility aliases
ATCTransformerEncoder = ATCTransformerEncoder  # Already imported from encoders
AttentionPooling = AttentionPooling  # Already imported from encoders


def create_model(config: Optional[NetworkConfig] = None) -> ATCActorCritic:
    """
    Create ATCActorCritic model with given configuration.
    
    Args:
        config: Network configuration (uses default if None)
        
    Returns:
        ATCActorCritic model instance
    """
    return ATCActorCritic(config)


def create_default_model(**overrides) -> ATCActorCritic:
    """
    Create ATCActorCritic model with default configuration and optional overrides.
    
    Args:
        **overrides: Configuration values to override
        
    Returns:
        ATCActorCritic model instance
    """
    config = create_default_network_config(**overrides)
    return ATCActorCritic(config)
