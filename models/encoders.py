"""
Encoder modules for neural network architectures.

This module provides encoder classes for processing aircraft and global state
data in the ATC reinforcement learning environment.
"""

import logging
from typing import Tuple

import torch
import torch.nn as nn

from .config import EncoderConfig, AttentionPoolingConfig


logger = logging.getLogger(__name__)


class ATCTransformerEncoder(nn.Module):
    """
    Transformer encoder for processing variable number of aircraft.
    
    This encoder uses self-attention to handle aircraft interactions
    and provides a robust representation for variable-length sequences.
    
    Example:
        >>> config = EncoderConfig(hidden_dim=256, num_heads=8, num_layers=4)
        >>> encoder = ATCTransformerEncoder(config)
        >>> x = torch.randn(2, 20, 14)  # batch_size=2, max_aircraft=20, features=14
        >>> mask = torch.ones(2, 20, dtype=torch.bool)  # all aircraft valid
        >>> output = encoder(x, mask)
        >>> print(output.shape)  # torch.Size([2, 20, 256])
    """
    
    def __init__(self, config: EncoderConfig):
        """
        Initialize transformer encoder.
        
        Args:
            config: Encoder configuration
        """
        super().__init__()
        
        self.config = config
        self.input_dim = config.input_dim
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.num_layers = config.num_layers
        self.dropout = config.dropout
        
        # Input projection
        self.input_projection = nn.Linear(self.input_dim, self.hidden_dim)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=self.dropout,
            activation=config.activation.value,
            batch_first=True,
            norm_first=config.norm_first,
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=self.num_layers
        )
        
        # Layer normalization
        if config.use_layer_norm:
            self.layer_norm = nn.LayerNorm(self.hidden_dim)
        else:
            self.layer_norm = nn.Identity()
        
        # Initialize weights
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
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through transformer encoder.
        
        Args:
            x: Input tensor of shape (batch_size, max_aircraft, input_dim)
            mask: Boolean mask of shape (batch_size, max_aircraft) where True indicates valid aircraft
            
        Returns:
            Encoded tensor of shape (batch_size, max_aircraft, hidden_dim)
        """
        # Validate input shapes
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input tensor, got {x.dim()}D")
        
        if mask.dim() != 2:
            raise ValueError(f"Expected 2D mask tensor, got {mask.dim()}D")
        
        if x.size(0) != mask.size(0):
            raise ValueError(f"Batch size mismatch: x={x.size(0)}, mask={mask.size(0)}")
        
        if x.size(1) != mask.size(1):
            raise ValueError(f"Sequence length mismatch: x={x.size(1)}, mask={mask.size(1)}")
        
        # Project to hidden dimension
        x = self.input_projection(x)
        
        # Create attention mask (True values are ignored in PyTorch)
        # Transformer expects True for positions to be masked out
        attn_mask = ~mask  # Invert: False for valid, True for padding
        
        # Apply transformer
        x = self.transformer(x, src_key_padding_mask=attn_mask)
        
        # Apply layer normalization
        x = self.layer_norm(x)
        
        return x
    
    def get_output_dim(self) -> int:
        """Get output dimension of the encoder."""
        return self.hidden_dim


class GlobalStateEncoder(nn.Module):
    """
    Encoder for global state information.
    
    This encoder processes global state features such as time, aircraft count,
    conflict count, and score into a fixed-size representation.
    
    Example:
        >>> encoder = GlobalStateEncoder(input_dim=4, hidden_dim=128)
        >>> x = torch.randn(2, 4)  # batch_size=2, global_features=4
        >>> output = encoder(x)
        >>> print(output.shape)  # torch.Size([2, 128])
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2, 
                 dropout: float = 0.1, activation: str = "relu"):
        """
        Initialize global state encoder.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
            num_layers: Number of layers
            dropout: Dropout rate
            activation: Activation function
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Build layers
        layers = []
        current_dim = input_dim
        
        for i in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            
            if i < num_layers - 1:  # Don't add activation after last layer
                if activation == "relu":
                    layers.append(nn.ReLU())
                elif activation == "gelu":
                    layers.append(nn.GELU())
                elif activation == "tanh":
                    layers.append(nn.Tanh())
                else:
                    raise ValueError(f"Unknown activation: {activation}")
                
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
            
            current_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        
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
        Forward pass through global state encoder.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Encoded tensor of shape (batch_size, hidden_dim)
        """
        if x.dim() != 2:
            raise ValueError(f"Expected 2D input tensor, got {x.dim()}D")
        
        if x.size(1) != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {x.size(1)}")
        
        return self.encoder(x)
    
    def get_output_dim(self) -> int:
        """Get output dimension of the encoder."""
        return self.hidden_dim


class AttentionPooling(nn.Module):
    """
    Attention-based pooling to get fixed-size representation from variable aircraft.
    
    This module uses a learnable query to attend to all aircraft representations
    and produce a single pooled representation.
    
    Example:
        >>> config = AttentionPoolingConfig(hidden_dim=256, num_heads=8)
        >>> pooling = AttentionPooling(config)
        >>> x = torch.randn(2, 20, 256)  # batch_size=2, max_aircraft=20, hidden_dim=256
        >>> mask = torch.ones(2, 20, dtype=torch.bool)  # all aircraft valid
        >>> output = pooling(x, mask)
        >>> print(output.shape)  # torch.Size([2, 256])
    """
    
    def __init__(self, config: AttentionPoolingConfig):
        """
        Initialize attention pooling.
        
        Args:
            config: Attention pooling configuration
        """
        super().__init__()
        
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.dropout = config.dropout
        
        # Learnable query
        self.query = nn.Parameter(torch.randn(1, 1, self.hidden_dim))
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            batch_first=True
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize network weights."""
        nn.init.xavier_uniform_(self.query)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through attention pooling.
        
        Args:
            x: Input tensor of shape (batch_size, max_aircraft, hidden_dim)
            mask: Boolean mask of shape (batch_size, max_aircraft) where True indicates valid aircraft
            
        Returns:
            Pooled tensor of shape (batch_size, hidden_dim)
        """
        # Validate input shapes
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input tensor, got {x.dim()}D")
        
        if mask.dim() != 2:
            raise ValueError(f"Expected 2D mask tensor, got {mask.dim()}D")
        
        if x.size(0) != mask.size(0):
            raise ValueError(f"Batch size mismatch: x={x.size(0)}, mask={mask.size(0)}")
        
        if x.size(1) != mask.size(1):
            raise ValueError(f"Sequence length mismatch: x={x.size(1)}, mask={mask.size(1)}")
        
        batch_size = x.size(0)
        
        # Expand query to batch size
        query = self.query.expand(batch_size, -1, -1)
        
        # Create attention mask (True values are ignored)
        attn_mask = ~mask  # Invert: False for valid, True for padding
        
        # Apply attention
        pooled, _ = self.attention(
            query=query,
            key=x,
            value=x,
            key_padding_mask=attn_mask
        )
        
        return pooled.squeeze(1)  # Remove sequence dimension
    
    def get_output_dim(self) -> int:
        """Get output dimension of the pooling layer."""
        return self.hidden_dim
