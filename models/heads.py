"""
Policy and value head modules for neural network architectures.

This module provides policy and value head classes for actor-critic
architectures in the ATC reinforcement learning environment.
"""

import logging
from typing import Dict, Tuple

import torch
import torch.nn as nn

from .config import PolicyHeadConfig, ValueHeadConfig


logger = logging.getLogger(__name__)


class PolicyHead(nn.Module):
    """
    Policy head for actor-critic architecture.
    
    This module outputs action logits for all action components including
    aircraft selection, command type, and command parameters.
    
    Example:
        >>> config = PolicyHeadConfig(hidden_dim=256, aircraft_id_size=21)
        >>> policy_head = PolicyHead(config)
        >>> x = torch.randn(2, 256)  # batch_size=2, hidden_dim=256
        >>> logits = policy_head(x)
        >>> print(logits["aircraft_id"].shape)  # torch.Size([2, 21])
    """
    
    def __init__(self, config: PolicyHeadConfig):
        """
        Initialize policy head.
        
        Args:
            config: Policy head configuration
        """
        super().__init__()
        
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers
        self.dropout = config.dropout
        
        # Build shared layers
        shared_layers = []
        current_dim = self.hidden_dim
        
        for i in range(self.num_layers):
            shared_layers.append(nn.Linear(current_dim, self.hidden_dim))
            
            if config.activation.value == "relu":
                shared_layers.append(nn.ReLU())
            elif config.activation.value == "gelu":
                shared_layers.append(nn.GELU())
            elif config.activation.value == "tanh":
                shared_layers.append(nn.Tanh())
            else:
                raise ValueError(f"Unknown activation: {config.activation.value}")
            
            if self.dropout > 0:
                shared_layers.append(nn.Dropout(self.dropout))
        
        self.shared_layers = nn.Sequential(*shared_layers)
        
        # Aircraft selection via attention
        self.aircraft_attention = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.aircraft_key = nn.Linear(self.hidden_dim, self.hidden_dim)
        
        # Action output heads
        self.command_head = nn.Linear(self.hidden_dim, config.command_type_size)
        self.altitude_head = nn.Linear(self.hidden_dim, config.altitude_size)
        self.heading_head = nn.Linear(self.hidden_dim, config.heading_size)
        self.speed_head = nn.Linear(self.hidden_dim, config.speed_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, shared_features: torch.Tensor, aircraft_encoded: torch.Tensor, 
                aircraft_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through policy head.
        
        Args:
            shared_features: Shared features of shape (batch_size, hidden_dim)
            aircraft_encoded: Aircraft encodings of shape (batch_size, max_aircraft, hidden_dim)
            aircraft_mask: Boolean mask of shape (batch_size, max_aircraft)
            
        Returns:
            Dictionary of action logits
        """
        # Validate input shapes
        if shared_features.dim() != 2:
            raise ValueError(f"Expected 2D shared_features, got {shared_features.dim()}D")
        
        if aircraft_encoded.dim() != 3:
            raise ValueError(f"Expected 3D aircraft_encoded, got {aircraft_encoded.dim()}D")
        
        if aircraft_mask.dim() != 2:
            raise ValueError(f"Expected 2D aircraft_mask, got {aircraft_mask.dim()}D")
        
        batch_size = aircraft_mask.size(0)
        
        # Process shared features
        processed_features = self.shared_layers(shared_features)
        
        # Aircraft selection via attention
        query = self.aircraft_attention(processed_features).unsqueeze(1)  # (batch, 1, hidden)
        keys = self.aircraft_key(aircraft_encoded)  # (batch, max_aircraft, hidden)
        
        # Attention scores for aircraft selection
        aircraft_logits = torch.bmm(query, keys.transpose(1, 2)).squeeze(1)  # (batch, max_aircraft)
        
        # Add "no action" option
        no_action_logit = torch.zeros(batch_size, 1, device=aircraft_logits.device)
        aircraft_logits = torch.cat([aircraft_logits, no_action_logit], dim=1)
        
        # Mask out invalid aircraft
        mask_expanded = torch.cat([
            aircraft_mask,
            torch.ones(batch_size, 1, dtype=torch.bool, device=aircraft_mask.device),
        ], dim=1)
        aircraft_logits = aircraft_logits.masked_fill(~mask_expanded, float("-inf"))
        
        # Command type and parameter logits
        command_logits = self.command_head(processed_features)
        altitude_logits = self.altitude_head(processed_features)
        heading_logits = self.heading_head(processed_features)
        speed_logits = self.speed_head(processed_features)
        
        return {
            "aircraft_id": aircraft_logits,
            "command_type": command_logits,
            "altitude": altitude_logits,
            "heading": heading_logits,
            "speed": speed_logits,
        }
    
    def get_action_space_sizes(self) -> Dict[str, int]:
        """Get action space sizes."""
        return {
            "aircraft_id": self.config.aircraft_id_size,
            "command_type": self.config.command_type_size,
            "altitude": self.config.altitude_size,
            "heading": self.config.heading_size,
            "speed": self.config.speed_size,
        }


class ValueHead(nn.Module):
    """
    Value head for actor-critic architecture.
    
    This module outputs state value estimates for the critic component
    of the actor-critic algorithm.
    
    Example:
        >>> config = ValueHeadConfig(hidden_dim=256, num_layers=2)
        >>> value_head = ValueHead(config)
        >>> x = torch.randn(2, 256)  # batch_size=2, hidden_dim=256
        >>> value = value_head(x)
        >>> print(value.shape)  # torch.Size([2, 1])
    """
    
    def __init__(self, config: ValueHeadConfig):
        """
        Initialize value head.
        
        Args:
            config: Value head configuration
        """
        super().__init__()
        
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers
        self.dropout = config.dropout
        
        # Build layers
        layers = []
        current_dim = self.hidden_dim
        
        for i in range(self.num_layers):
            layers.append(nn.Linear(current_dim, self.hidden_dim))
            
            if config.activation.value == "relu":
                layers.append(nn.ReLU())
            elif config.activation.value == "gelu":
                layers.append(nn.GELU())
            elif config.activation.value == "tanh":
                layers.append(nn.Tanh())
            else:
                raise ValueError(f"Unknown activation: {config.activation.value}")
            
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
        
        # Output layer
        layers.append(nn.Linear(self.hidden_dim, 1))
        
        self.value_network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through value head.
        
        Args:
            x: Input tensor of shape (batch_size, hidden_dim)
            
        Returns:
            Value tensor of shape (batch_size, 1)
        """
        if x.dim() != 2:
            raise ValueError(f"Expected 2D input tensor, got {x.dim()}D")
        
        if x.size(1) != self.hidden_dim:
            raise ValueError(f"Expected hidden_dim={self.hidden_dim}, got {x.size(1)}")
        
        return self.value_network(x)
    
    def get_output_dim(self) -> int:
        """Get output dimension of the value head."""
        return 1


class ActorCriticHeads(nn.Module):
    """
    Combined actor-critic heads module.
    
    This module combines policy and value heads for actor-critic algorithms,
    providing a unified interface for both components.
    
    Example:
        >>> policy_config = PolicyHeadConfig(hidden_dim=256)
        >>> value_config = ValueHeadConfig(hidden_dim=256)
        >>> heads = ActorCriticHeads(policy_config, value_config)
        >>> shared_features = torch.randn(2, 256)
        >>> aircraft_encoded = torch.randn(2, 20, 256)
        >>> aircraft_mask = torch.ones(2, 20, dtype=torch.bool)
        >>> action_logits, value = heads(shared_features, aircraft_encoded, aircraft_mask)
    """
    
    def __init__(self, policy_config: PolicyHeadConfig, value_config: ValueHeadConfig):
        """
        Initialize actor-critic heads.
        
        Args:
            policy_config: Policy head configuration
            value_config: Value head configuration
        """
        super().__init__()
        
        self.policy_head = PolicyHead(policy_config)
        self.value_head = ValueHead(value_config)
    
    def forward(self, shared_features: torch.Tensor, aircraft_encoded: torch.Tensor, 
                aircraft_mask: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Forward pass through both policy and value heads.
        
        Args:
            shared_features: Shared features of shape (batch_size, hidden_dim)
            aircraft_encoded: Aircraft encodings of shape (batch_size, max_aircraft, hidden_dim)
            aircraft_mask: Boolean mask of shape (batch_size, max_aircraft)
            
        Returns:
            Tuple of (action_logits, value)
        """
        action_logits = self.policy_head(shared_features, aircraft_encoded, aircraft_mask)
        value = self.value_head(shared_features)
        
        return action_logits, value
    
    def get_action_space_sizes(self) -> Dict[str, int]:
        """Get action space sizes from policy head."""
        return self.policy_head.get_action_space_sizes()
