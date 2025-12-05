"""
TD-MPC 2: Temporal Difference Model Predictive Control 2.

This module implements the TD-MPC 2 algorithm, which combines:
- Transformer-based world model for dynamics prediction
- Model Predictive Control (MPC) for action planning
- Q-learning for long-term value estimation

Key features:
- Latent state representation using transformer encoder
- Transformer-based dynamics model
- MPC planning with Cross-Entropy Method (CEM)
- Q-function for terminal value estimation
- Sample-efficient model-based RL

References:
    Hansen et al., "TD-MPC2: Scalable, Robust World Models for Continuous Control"
    https://arxiv.org/abs/2310.16828
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .encoders import ATCTransformerEncoder, AttentionPooling, GlobalStateEncoder
from .config import EncoderConfig, AttentionPoolingConfig
from environment.utils import get_device

logger = logging.getLogger(__name__)


@dataclass
class TDMPC2Config:
    """Configuration for TD-MPC 2 model."""
    
    # Observation dimensions
    aircraft_feature_dim: int = 14
    global_feature_dim: int = 4
    max_aircraft: int = 20
    
    # Latent space
    latent_dim: int = 512
    
    # Encoder architecture
    encoder_hidden_dim: int = 256
    encoder_num_heads: int = 8
    encoder_num_layers: int = 4
    encoder_dropout: float = 0.1
    
    # Dynamics architecture
    dynamics_hidden_dim: int = 512
    dynamics_num_layers: int = 3
    dynamics_num_heads: int = 8
    dynamics_dropout: float = 0.1
    
    # Action space
    action_dim: int = 5  # aircraft_id, command_type, altitude, heading, speed
    
    # Reward predictor
    reward_hidden_dim: int = 256
    reward_num_layers: int = 2
    
    # Q-network
    q_hidden_dim: int = 512
    q_num_layers: int = 3
    
    # Device
    device: str = field(default_factory=lambda: get_device())
    
    def __post_init__(self):
        """Validate configuration."""
        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if self.encoder_hidden_dim <= 0:
            raise ValueError("encoder_hidden_dim must be positive")
        if self.dynamics_hidden_dim <= 0:
            raise ValueError("dynamics_hidden_dim must be positive")
        if self.dynamics_num_heads <= 0:
            raise ValueError("dynamics_num_heads must be positive")
        if self.latent_dim % self.dynamics_num_heads != 0:
            raise ValueError(f"latent_dim ({self.latent_dim}) must be divisible by dynamics_num_heads ({self.dynamics_num_heads})")
        if not (0.0 <= self.encoder_dropout <= 1.0):
            raise ValueError("encoder_dropout must be in [0.0, 1.0]")
        if self.encoder_hidden_dim % self.encoder_num_heads != 0:
            raise ValueError("encoder_hidden_dim must be divisible by encoder_num_heads")
        if self.max_aircraft <= 0:
            raise ValueError("max_aircraft must be positive")
        if self.action_dim <= 0:
            raise ValueError("action_dim must be positive")
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for logging."""
        return {
            "aircraft_feature_dim": self.aircraft_feature_dim,
            "global_feature_dim": self.global_feature_dim,
            "max_aircraft": self.max_aircraft,
            "latent_dim": self.latent_dim,
            "encoder_hidden_dim": self.encoder_hidden_dim,
            "encoder_num_heads": self.encoder_num_heads,
            "encoder_num_layers": self.encoder_num_layers,
            "dynamics_hidden_dim": self.dynamics_hidden_dim,
            "dynamics_num_layers": self.dynamics_num_layers,
            "dynamics_num_heads": self.dynamics_num_heads,
            "action_dim": self.action_dim,
        }


class LatentEncoder(nn.Module):
    """
    Encodes observations to latent space.
    
    Uses transformer encoder to process variable number of aircraft,
    then pools to fixed-size latent representation.
    
    Example:
        >>> config = TDMPC2Config()
        >>> encoder = LatentEncoder(config)
        >>> aircraft = torch.randn(2, 20, 14)  # batch, max_aircraft, features
        >>> mask = torch.ones(2, 20, dtype=torch.bool)
        >>> global_state = torch.randn(2, 4)
        >>> latent = encoder(aircraft, mask, global_state)
        >>> print(latent.shape)  # torch.Size([2, 512])
    """
    
    def __init__(self, config: TDMPC2Config):
        """
        Initialize latent encoder.
        
        Args:
            config: TD-MPC 2 configuration
        """
        super().__init__()
        
        self.config = config
        
        # Aircraft encoder (transformer)
        encoder_config = EncoderConfig(
            input_dim=config.aircraft_feature_dim,
            hidden_dim=config.encoder_hidden_dim,
            num_heads=config.encoder_num_heads,
            num_layers=config.encoder_num_layers,
            dropout=config.encoder_dropout,
        )
        self.aircraft_encoder = ATCTransformerEncoder(encoder_config)
        
        # Attention pooling to aggregate aircraft
        pooling_config = AttentionPoolingConfig(
            hidden_dim=config.encoder_hidden_dim,
            num_heads=config.encoder_num_heads,
            dropout=config.encoder_dropout,
        )
        self.attention_pooling = AttentionPooling(pooling_config)
        
        # Global state encoder
        self.global_encoder = GlobalStateEncoder(
            input_dim=config.global_feature_dim,
            hidden_dim=config.encoder_hidden_dim,
            num_layers=2,
            dropout=config.encoder_dropout,
        )
        
        # Project to latent dimension
        combined_dim = config.encoder_hidden_dim + config.encoder_hidden_dim
        self.latent_projection = nn.Sequential(
            nn.Linear(combined_dim, config.latent_dim),
            nn.LayerNorm(config.latent_dim),
            nn.GELU(),
        )
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        aircraft: torch.Tensor,
        aircraft_mask: torch.Tensor,
        global_state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode observation to latent space.
        
        Args:
            aircraft: Aircraft features (batch_size, max_aircraft, aircraft_feature_dim)
            aircraft_mask: Boolean mask (batch_size, max_aircraft)
            global_state: Global state (batch_size, global_feature_dim)
            
        Returns:
            Latent representation (batch_size, latent_dim)
            
        Raises:
            ValueError: If input shapes are invalid
        """
        # Validate input shapes
        if aircraft.dim() != 3:
            raise ValueError(
                f"Expected 3D aircraft tensor (batch, max_aircraft, features), "
                f"got {aircraft.dim()}D with shape {aircraft.shape}"
            )
        if aircraft_mask.dim() != 2:
            raise ValueError(
                f"Expected 2D mask tensor (batch, max_aircraft), "
                f"got {aircraft_mask.dim()}D with shape {aircraft_mask.shape}"
            )
        if global_state.dim() != 2:
            raise ValueError(
                f"Expected 2D global_state tensor (batch, features), "
                f"got {global_state.dim()}D with shape {global_state.shape}"
            )
        
        batch_size = aircraft.size(0)
        if aircraft.size(0) != aircraft_mask.size(0):
            raise ValueError(
                f"Batch size mismatch: aircraft={aircraft.size(0)}, "
                f"mask={aircraft_mask.size(0)}"
            )
        if aircraft.size(0) != global_state.size(0):
            raise ValueError(
                f"Batch size mismatch: aircraft={aircraft.size(0)}, "
                f"global_state={global_state.size(0)}"
            )
        if aircraft.size(1) != self.config.max_aircraft:
            raise ValueError(
                f"Aircraft sequence length mismatch: expected {self.config.max_aircraft}, "
                f"got {aircraft.size(1)}"
            )
        
        # Encode aircraft
        aircraft_encoded = self.aircraft_encoder(aircraft, aircraft_mask)
        
        # Pool aircraft to fixed size
        aircraft_pooled = self.attention_pooling(aircraft_encoded, aircraft_mask)
        
        # Encode global state
        global_encoded = self.global_encoder(global_state)
        
        # Combine and project to latent
        combined = torch.cat([aircraft_pooled, global_encoded], dim=-1)
        latent = self.latent_projection(combined)
        
        return latent
    
    def get_output_dim(self) -> int:
        """Get output dimension."""
        return self.config.latent_dim


class TransformerDynamics(nn.Module):
    """
    Transformer-based dynamics model for latent state transitions.
    
    Predicts next latent state from current latent state and action.
    
    Example:
        >>> config = TDMPC2Config()
        >>> dynamics = TransformerDynamics(config)
        >>> latent = torch.randn(2, 512)
        >>> action = torch.randn(2, 5)
        >>> next_latent = dynamics(latent, action)
        >>> print(next_latent.shape)  # torch.Size([2, 512])
    """
    
    def __init__(self, config: TDMPC2Config):
        """
        Initialize dynamics model.
        
        Args:
            config: TD-MPC 2 configuration
        """
        super().__init__()
        
        self.config = config
        
        # Action embedding
        self.action_embedding = nn.Sequential(
            nn.Linear(config.action_dim, config.latent_dim),
            nn.LayerNorm(config.latent_dim),
            nn.GELU(),
        )
        
        # Transformer for dynamics
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.latent_dim,
            nhead=config.dynamics_num_heads,
            dim_feedforward=config.dynamics_hidden_dim * 4,
            dropout=config.dynamics_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.dynamics_num_layers,
        )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(config.latent_dim, config.latent_dim),
            nn.LayerNorm(config.latent_dim),
            nn.GELU(),
            nn.Linear(config.latent_dim, config.latent_dim),
        )
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        latent: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict next latent state.
        
        Args:
            latent: Current latent state (batch_size, latent_dim)
            action: Action taken (batch_size, action_dim)
            
        Returns:
            Next latent state (batch_size, latent_dim)
            
        Raises:
            ValueError: If input shapes are invalid
        """
        if latent.dim() != 2:
            raise ValueError(
                f"Expected 2D latent tensor (batch, latent_dim), "
                f"got {latent.dim()}D with shape {latent.shape}"
            )
        if action.dim() != 2:
            raise ValueError(
                f"Expected 2D action tensor (batch, action_dim), "
                f"got {action.dim()}D with shape {action.shape}"
            )
        if latent.size(0) != action.size(0):
            raise ValueError(
                f"Batch size mismatch: latent={latent.size(0)}, action={action.size(0)}"
            )
        if latent.size(1) != self.config.latent_dim:
            raise ValueError(
                f"Latent dimension mismatch: expected {self.config.latent_dim}, "
                f"got {latent.size(1)}"
            )
        if action.size(1) != self.config.action_dim:
            raise ValueError(
                f"Action dimension mismatch: expected {self.config.action_dim}, "
                f"got {action.size(1)}"
            )
        
        # Embed action
        action_emb = self.action_embedding(action)
        
        # Combine latent and action (as sequence)
        # Shape: (batch_size, 2, latent_dim)
        sequence = torch.stack([latent, action_emb], dim=1)
        
        # Apply transformer
        transformed = self.transformer(sequence)
        
        # Take the first element (latent) and add residual
        next_latent = transformed[:, 0, :]
        next_latent = next_latent + latent  # Residual connection
        
        # Output projection
        next_latent = self.output_projection(next_latent)
        
        return next_latent


class RewardPredictor(nn.Module):
    """
    Predicts rewards from latent states.
    
    Example:
        >>> config = TDMPC2Config()
        >>> reward_pred = RewardPredictor(config)
        >>> latent = torch.randn(2, 512)
        >>> reward = reward_pred(latent)
        >>> print(reward.shape)  # torch.Size([2, 1])
    """
    
    def __init__(self, config: TDMPC2Config):
        """
        Initialize reward predictor.
        
        Args:
            config: TD-MPC 2 configuration
        """
        super().__init__()
        
        self.config = config
        
        layers = []
        current_dim = config.latent_dim
        
        for i in range(config.reward_num_layers):
            layers.append(nn.Linear(current_dim, config.reward_hidden_dim))
            layers.append(nn.LayerNorm(config.reward_hidden_dim))
            layers.append(nn.GELU())
            current_dim = config.reward_hidden_dim
        
        layers.append(nn.Linear(current_dim, 1))
        
        self.predictor = nn.Sequential(*layers)
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Predict reward from latent state.
        
        Args:
            latent: Latent state (batch_size, latent_dim)
            
        Returns:
            Predicted reward (batch_size, 1)
            
        Raises:
            ValueError: If input shape is invalid
        """
        if latent.dim() != 2:
            raise ValueError(
                f"Expected 2D latent tensor (batch, latent_dim), "
                f"got {latent.dim()}D with shape {latent.shape}"
            )
        if latent.size(1) != self.config.latent_dim:
            raise ValueError(
                f"Latent dimension mismatch: expected {self.config.latent_dim}, "
                f"got {latent.size(1)}"
            )
        
        return self.predictor(latent)


class TDMPC2QNetwork(nn.Module):
    """
    Q-network for terminal value estimation in MPC.
    
    Example:
        >>> config = TDMPC2Config()
        >>> q_net = TDMPC2QNetwork(config)
        >>> latent = torch.randn(2, 512)
        >>> action = torch.randn(2, 5)
        >>> q_value = q_net(latent, action)
        >>> print(q_value.shape)  # torch.Size([2, 1])
    """
    
    def __init__(self, config: TDMPC2Config):
        """
        Initialize Q-network.
        
        Args:
            config: TD-MPC 2 configuration
        """
        super().__init__()
        
        self.config = config
        
        # Action embedding
        self.action_embedding = nn.Sequential(
            nn.Linear(config.action_dim, config.latent_dim),
            nn.LayerNorm(config.latent_dim),
            nn.GELU(),
        )
        
        # Combine latent and action
        combined_dim = config.latent_dim + config.latent_dim
        
        layers = []
        current_dim = combined_dim
        
        for i in range(config.q_num_layers):
            layers.append(nn.Linear(current_dim, config.q_hidden_dim))
            layers.append(nn.LayerNorm(config.q_hidden_dim))
            layers.append(nn.GELU())
            current_dim = config.q_hidden_dim
        
        layers.append(nn.Linear(current_dim, 1))
        
        self.q_network = nn.Sequential(*layers)
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        latent: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimate Q-value.
        
        Args:
            latent: Latent state (batch_size, latent_dim)
            action: Action (batch_size, action_dim)
            
        Returns:
            Q-value (batch_size, 1)
            
        Raises:
            ValueError: If input shapes are invalid
        """
        if latent.dim() != 2:
            raise ValueError(
                f"Expected 2D latent tensor (batch, latent_dim), "
                f"got {latent.dim()}D with shape {latent.shape}"
            )
        if action.dim() != 2:
            raise ValueError(
                f"Expected 2D action tensor (batch, action_dim), "
                f"got {action.dim()}D with shape {action.shape}"
            )
        if latent.size(0) != action.size(0):
            raise ValueError(
                f"Batch size mismatch: latent={latent.size(0)}, action={action.size(0)}"
            )
        if latent.size(1) != self.config.latent_dim:
            raise ValueError(
                f"Latent dimension mismatch: expected {self.config.latent_dim}, "
                f"got {latent.size(1)}"
            )
        if action.size(1) != self.config.action_dim:
            raise ValueError(
                f"Action dimension mismatch: expected {self.config.action_dim}, "
                f"got {action.size(1)}"
            )
        
        # Embed action
        action_emb = self.action_embedding(action)
        
        # Combine
        combined = torch.cat([latent, action_emb], dim=-1)
        
        # Predict Q-value
        q_value = self.q_network(combined)
        
        return q_value


class TDMPC2Model(nn.Module):
    """
    Complete TD-MPC 2 model combining encoder, dynamics, reward, and Q-network.
    
    Example:
        >>> config = TDMPC2Config()
        >>> model = TDMPC2Model(config)
        >>> aircraft = torch.randn(2, 20, 14)
        >>> mask = torch.ones(2, 20, dtype=torch.bool)
        >>> global_state = torch.randn(2, 4)
        >>> action = torch.randn(2, 5)
        >>> next_aircraft = torch.randn(2, 20, 14)
        >>> next_mask = torch.ones(2, 20, dtype=torch.bool)
        >>> next_global_state = torch.randn(2, 4)
        >>> 
        >>> # Forward pass
        >>> latent = model.encode(aircraft, mask, global_state)
        >>> next_latent = model.dynamics(latent, action)
        >>> reward = model.reward(next_latent)
        >>> q_value = model.q_value(latent, action)
    """
    
    def __init__(self, config: TDMPC2Config):
        """
        Initialize TD-MPC 2 model.
        
        Args:
            config: TD-MPC 2 configuration
        """
        super().__init__()
        
        self.config = config
        
        # Components
        self.encoder = LatentEncoder(config)
        self.dynamics = TransformerDynamics(config)
        self.reward = RewardPredictor(config)
        self.q_network = TDMPC2QNetwork(config)
        
        # Move to device
        self.to(config.device)
    
    def encode(
        self,
        aircraft: torch.Tensor,
        aircraft_mask: torch.Tensor,
        global_state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode observation to latent space.
        
        Args:
            aircraft: Aircraft features (batch_size, max_aircraft, aircraft_feature_dim)
            aircraft_mask: Boolean mask (batch_size, max_aircraft)
            global_state: Global state (batch_size, global_feature_dim)
            
        Returns:
            Latent representation (batch_size, latent_dim)
        """
        return self.encoder(aircraft, aircraft_mask, global_state)
    
    def forward(
        self,
        aircraft: torch.Tensor,
        aircraft_mask: torch.Tensor,
        global_state: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass: encode, predict next state, reward, and Q-value.
        
        Args:
            aircraft: Aircraft features (batch_size, max_aircraft, aircraft_feature_dim)
            aircraft_mask: Boolean mask (batch_size, max_aircraft)
            global_state: Global state (batch_size, global_feature_dim)
            action: Action (batch_size, action_dim)
            
        Returns:
            Tuple of (next_latent, reward, q_value)
        """
        # Encode
        latent = self.encode(aircraft, aircraft_mask, global_state)
        
        # Predict next latent
        next_latent = self.dynamics(latent, action)
        
        # Predict reward
        reward = self.reward(next_latent)
        
        # Predict Q-value
        q_value = self.q_network(latent, action)
        
        return next_latent, reward, q_value

