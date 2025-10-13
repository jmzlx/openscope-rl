"""
Neural Network architectures for OpenScope RL agent
Uses Transformer-based architecture to handle variable numbers of aircraft
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
import math


class ATCTransformerEncoder(nn.Module):
    """
    Transformer encoder for processing variable number of aircraft
    Uses self-attention to handle aircraft interactions
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding (optional, for ordering)
        self.pos_encoding = PositionalEncoding(hidden_dim, dropout, max_len=20)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, max_aircraft, input_dim)
            mask: (batch, max_aircraft) - True for valid aircraft, False for padding
        
        Returns:
            encoded: (batch, max_aircraft, hidden_dim)
        """
        # Project to hidden dimension
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Create attention mask (True values are ignored)
        # Transformer expects True for positions to be masked out
        attn_mask = ~mask  # Invert: False for valid, True for padding
        
        # Apply transformer
        # Note: src_key_padding_mask expects (batch, seq_len)
        x = self.transformer(x, src_key_padding_mask=attn_mask)
        
        x = self.layer_norm(x)
        
        return x


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 20):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class AttentionPooling(nn.Module):
    """Attention-based pooling to get fixed-size representation from variable aircraft"""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.attention = nn.MultiheadAttention(
            hidden_dim, 
            num_heads=8, 
            batch_first=True
        )
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, max_aircraft, hidden_dim)
            mask: (batch, max_aircraft)
        
        Returns:
            pooled: (batch, hidden_dim)
        """
        batch_size = x.size(0)
        query = self.query.expand(batch_size, -1, -1)
        
        # Attention mask
        attn_mask = ~mask  # Invert for attention
        
        # Apply attention
        pooled, _ = self.attention(
            query, 
            x, 
            x, 
            key_padding_mask=attn_mask
        )
        
        return pooled.squeeze(1)


class ATCPolicyNetwork(nn.Module):
    """
    Policy network for selecting actions
    Outputs logits for aircraft selection and command parameters
    """
    
    def __init__(
        self,
        aircraft_feature_dim: int,
        global_feature_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        max_aircraft: int = 20,
        action_space_sizes: Dict[str, int] = None
    ):
        super().__init__()
        
        self.max_aircraft = max_aircraft
        
        if action_space_sizes is None:
            action_space_sizes = {
                "aircraft_id": max_aircraft + 1,
                "command_type": 5,
                "altitude": 18,
                "heading": 13,
                "speed": 8
            }
        self.action_space_sizes = action_space_sizes
        
        # Aircraft encoder
        self.aircraft_encoder = ATCTransformerEncoder(
            input_dim=aircraft_feature_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers
        )
        
        # Global state encoder
        self.global_encoder = nn.Sequential(
            nn.Linear(global_feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # Pooling to get fixed representation
        self.attention_pooling = AttentionPooling(hidden_dim)
        
        # Combined feature dimension
        combined_dim = hidden_dim + 128
        
        # Shared layers
        self.shared_layers = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Action heads
        # Aircraft selection: use attention over encoded aircraft
        self.aircraft_attention = nn.Linear(hidden_dim, hidden_dim)
        self.aircraft_key = nn.Linear(hidden_dim, hidden_dim)
        
        # Command type
        self.command_head = nn.Linear(hidden_dim, action_space_sizes["command_type"])
        
        # Command parameters (conditional on command type)
        self.altitude_head = nn.Linear(hidden_dim, action_space_sizes["altitude"])
        self.heading_head = nn.Linear(hidden_dim, action_space_sizes["heading"])
        self.speed_head = nn.Linear(hidden_dim, action_space_sizes["speed"])
    
    def forward(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            obs: Dictionary containing:
                - aircraft: (batch, max_aircraft, aircraft_feature_dim)
                - aircraft_mask: (batch, max_aircraft)
                - global_state: (batch, global_feature_dim)
        
        Returns:
            logits: Dictionary of action logits
        """
        aircraft = obs["aircraft"]
        aircraft_mask = obs["aircraft_mask"]
        global_state = obs["global_state"]
        
        batch_size = aircraft.size(0)
        
        # Encode aircraft
        aircraft_encoded = self.aircraft_encoder(aircraft, aircraft_mask)
        
        # Encode global state
        global_encoded = self.global_encoder(global_state)
        
        # Pool aircraft features
        aircraft_pooled = self.attention_pooling(aircraft_encoded, aircraft_mask)
        
        # Combine features
        combined = torch.cat([aircraft_pooled, global_encoded], dim=-1)
        
        # Shared processing
        shared_features = self.shared_layers(combined)
        
        # Aircraft selection via attention
        query = self.aircraft_attention(shared_features).unsqueeze(1)  # (batch, 1, hidden)
        keys = self.aircraft_key(aircraft_encoded)  # (batch, max_aircraft, hidden)
        
        # Attention scores for aircraft selection
        aircraft_logits = torch.bmm(query, keys.transpose(1, 2)).squeeze(1)  # (batch, max_aircraft)
        
        # Add "no action" option
        no_action_logit = torch.zeros(batch_size, 1, device=aircraft_logits.device)
        aircraft_logits = torch.cat([aircraft_logits, no_action_logit], dim=1)
        
        # Mask out invalid aircraft
        mask_expanded = torch.cat([
            aircraft_mask,
            torch.ones(batch_size, 1, dtype=torch.bool, device=aircraft_mask.device)
        ], dim=1)
        aircraft_logits = aircraft_logits.masked_fill(~mask_expanded, float('-inf'))
        
        # Command type and parameters
        command_logits = self.command_head(shared_features)
        altitude_logits = self.altitude_head(shared_features)
        heading_logits = self.heading_head(shared_features)
        speed_logits = self.speed_head(shared_features)
        
        return {
            "aircraft_id": aircraft_logits,
            "command_type": command_logits,
            "altitude": altitude_logits,
            "heading": heading_logits,
            "speed": speed_logits
        }


class ATCValueNetwork(nn.Module):
    """
    Value network for estimating state value
    """
    
    def __init__(
        self,
        aircraft_feature_dim: int,
        global_feature_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        max_aircraft: int = 20
    ):
        super().__init__()
        
        # Aircraft encoder (shared architecture with policy)
        self.aircraft_encoder = ATCTransformerEncoder(
            input_dim=aircraft_feature_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers
        )
        
        # Global state encoder
        self.global_encoder = nn.Sequential(
            nn.Linear(global_feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # Pooling
        self.attention_pooling = AttentionPooling(hidden_dim)
        
        # Value head
        combined_dim = hidden_dim + 128
        self.value_head = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            obs: Dictionary containing observation tensors
        
        Returns:
            value: (batch, 1)
        """
        aircraft = obs["aircraft"]
        aircraft_mask = obs["aircraft_mask"]
        global_state = obs["global_state"]
        
        # Encode
        aircraft_encoded = self.aircraft_encoder(aircraft, aircraft_mask)
        global_encoded = self.global_encoder(global_state)
        
        # Pool
        aircraft_pooled = self.attention_pooling(aircraft_encoded, aircraft_mask)
        
        # Combine
        combined = torch.cat([aircraft_pooled, global_encoded], dim=-1)
        
        # Value
        value = self.value_head(combined)
        
        return value


class ATCActorCritic(nn.Module):
    """
    Combined actor-critic network for PPO
    """
    
    def __init__(
        self,
        aircraft_feature_dim: int = 32,
        global_feature_dim: int = 16,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        max_aircraft: int = 20,
        action_space_sizes: Dict[str, int] = None
    ):
        super().__init__()
        
        self.policy = ATCPolicyNetwork(
            aircraft_feature_dim,
            global_feature_dim,
            hidden_dim,
            num_heads,
            num_layers,
            max_aircraft,
            action_space_sizes
        )
        
        self.value = ATCValueNetwork(
            aircraft_feature_dim,
            global_feature_dim,
            hidden_dim,
            num_heads,
            num_layers,
            max_aircraft
        )
    
    def forward(self, obs: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Forward pass through both policy and value networks
        
        Returns:
            action_logits: Dictionary of logits for each action component
            value: State value estimate
        """
        action_logits = self.policy(obs)
        value = self.value(obs)
        
        return action_logits, value
    
    def get_action_and_value(
        self, 
        obs: Dict[str, torch.Tensor],
        action: Dict[str, torch.Tensor] = None
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action or compute log prob of given action
        
        Returns:
            action: Sampled action (or input action)
            log_prob: Log probability of action
            entropy: Entropy of policy
            value: State value
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
                "speed": speed_dist.sample()
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

