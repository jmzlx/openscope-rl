"""
Trajectory Transformer for modeling full trajectories autoregressively.

This module implements a unified transformer that models the sequence:
s₀, a₀, r₀, s₁, a₁, r₁, ...

The model predicts all tokens (states, actions, rewards) with multi-head outputs
and can be used for both behavior cloning and world model planning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field
import math


@dataclass
class TrajectoryTransformerConfig:
    """Configuration for Trajectory Transformer."""

    # Input dimensions
    state_dim: int = 128  # Flattened state dimension
    action_dim: int = 65  # Total action space size (21+5+18+13+8)
    reward_dim: int = 1

    # Embedding dimensions
    embed_dim: int = 256

    # Transformer architecture
    num_layers: int = 6
    num_heads: int = 8
    ff_dim: int = 1024
    dropout: float = 0.1

    # Context length (in timesteps)
    context_length: int = 20

    # Training
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4

    # Loss weights
    state_loss_weight: float = 1.0
    action_loss_weight: float = 1.0
    reward_loss_weight: float = 0.1

    # Temperature for sampling
    temperature: float = 1.0

    def __post_init__(self):
        """Validate configuration."""
        if self.embed_dim % self.num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        if self.context_length <= 0:
            raise ValueError("context_length must be positive")


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Initialize positional encoding.

        Args:
            d_model: Embedding dimension
            max_len: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add batch dimension

        # Register as buffer (not a parameter)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)

        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TrajectoryTransformer(nn.Module):
    """
    Trajectory Transformer that models full trajectories autoregressively.

    Models the sequence: s₀, a₀, r₀, s₁, a₁, r₁, ...
    Uses a single transformer with multi-head outputs for states, actions, and rewards.

    Example:
        >>> config = TrajectoryTransformerConfig(state_dim=128, action_dim=65)
        >>> model = TrajectoryTransformer(config)
        >>>
        >>> # Forward pass
        >>> states = torch.randn(32, 10, 128)  # (batch, timesteps, state_dim)
        >>> actions = torch.randint(0, 65, (32, 10))  # (batch, timesteps)
        >>> rewards = torch.randn(32, 10, 1)  # (batch, timesteps, 1)
        >>>
        >>> outputs = model(states, actions, rewards)
        >>> pred_states, pred_actions, pred_rewards = outputs
    """

    def __init__(self, config: TrajectoryTransformerConfig):
        """
        Initialize Trajectory Transformer.

        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config

        # Token embeddings
        self.state_embed = nn.Linear(config.state_dim, config.embed_dim)
        self.action_embed = nn.Embedding(config.action_dim, config.embed_dim)
        self.reward_embed = nn.Linear(config.reward_dim, config.embed_dim)

        # Token type embeddings (to distinguish state/action/reward)
        self.token_type_embed = nn.Embedding(3, config.embed_dim)

        # Positional encoding
        max_seq_len = config.context_length * 3  # s, a, r for each timestep
        self.pos_encoding = PositionalEncoding(
            config.embed_dim, max_len=max_seq_len, dropout=config.dropout
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.num_heads,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=config.num_layers
        )

        # Prediction heads
        self.state_head = nn.Sequential(
            nn.Linear(config.embed_dim, config.ff_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ff_dim, config.state_dim),
        )

        self.action_head = nn.Sequential(
            nn.Linear(config.embed_dim, config.ff_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ff_dim, config.action_dim),
        )

        self.reward_head = nn.Sequential(
            nn.Linear(config.embed_dim, config.ff_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ff_dim // 2, config.reward_dim),
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(
        self,
        states: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        rewards: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the trajectory transformer.

        Args:
            states: State tensor of shape (batch, timesteps, state_dim)
            actions: Action tensor of shape (batch, timesteps)
            rewards: Reward tensor of shape (batch, timesteps, 1)
            attention_mask: Optional attention mask

        Returns:
            Tuple of (predicted_states, predicted_actions, predicted_rewards)
            - predicted_states: (batch, timesteps, state_dim)
            - predicted_actions: (batch, timesteps, action_dim)
            - predicted_rewards: (batch, timesteps, reward_dim)
        """
        batch_size, timesteps, _ = states.shape

        # Embed tokens
        state_embeds = self.state_embed(states)  # (batch, T, embed)

        if actions is None:
            # During planning, we may not have actions yet
            actions = torch.zeros(batch_size, timesteps, dtype=torch.long, device=states.device)
        action_embeds = self.action_embed(actions)  # (batch, T, embed)

        if rewards is None:
            rewards = torch.zeros(batch_size, timesteps, 1, device=states.device)
        reward_embeds = self.reward_embed(rewards)  # (batch, T, embed)

        # Create token type IDs
        state_type = torch.zeros(batch_size, timesteps, dtype=torch.long, device=states.device)
        action_type = torch.ones(batch_size, timesteps, dtype=torch.long, device=states.device)
        reward_type = torch.full((batch_size, timesteps), 2, dtype=torch.long, device=states.device)

        # Add token type embeddings
        state_embeds = state_embeds + self.token_type_embed(state_type)
        action_embeds = action_embeds + self.token_type_embed(action_type)
        reward_embeds = reward_embeds + self.token_type_embed(reward_type)

        # Interleave tokens: s₀, a₀, r₀, s₁, a₁, r₁, ...
        # Shape: (batch, timesteps * 3, embed)
        token_embeds = torch.stack(
            [state_embeds, action_embeds, reward_embeds], dim=2
        ).reshape(batch_size, timesteps * 3, -1)

        # Add positional encoding
        token_embeds = self.pos_encoding(token_embeds)

        # Create causal mask
        seq_len = timesteps * 3
        causal_mask = self._create_causal_mask(seq_len, states.device)

        # Apply transformer
        if attention_mask is not None:
            # Expand attention mask to match causal mask shape
            attention_mask = attention_mask.unsqueeze(1).expand(-1, seq_len, -1)
            attention_mask = attention_mask & causal_mask.unsqueeze(0)
        else:
            attention_mask = causal_mask

        hidden = self.transformer(token_embeds, mask=causal_mask, is_causal=True)

        # Extract state, action, reward tokens
        # State predictions come from previous action/reward tokens
        # Action predictions come from state tokens
        # Reward predictions come from action tokens
        state_hidden = hidden[:, 0::3, :]  # Every 3rd starting from 0
        action_hidden = hidden[:, 1::3, :]  # Every 3rd starting from 1
        reward_hidden = hidden[:, 2::3, :]  # Every 3rd starting from 2

        # Predict next tokens
        pred_states = self.state_head(action_hidden)  # Predict s_{t+1} from a_t
        pred_actions = self.action_head(state_hidden)  # Predict a_t from s_t
        pred_rewards = self.reward_head(action_hidden)  # Predict r_t from a_t

        return pred_states, pred_actions, pred_rewards

    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Create causal attention mask.

        Args:
            seq_len: Sequence length
            device: Device to create mask on

        Returns:
            Causal mask of shape (seq_len, seq_len)
        """
        mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
            diagonal=1
        )
        return ~mask  # Invert: True means attend, False means mask

    def compute_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training loss.

        Args:
            states: Current states (batch, timesteps, state_dim)
            actions: Actions taken (batch, timesteps)
            rewards: Rewards received (batch, timesteps, 1)
            next_states: Next states (batch, timesteps, state_dim)

        Returns:
            Dictionary with loss components
        """
        # Forward pass
        pred_states, pred_actions, pred_rewards = self.forward(states, actions, rewards)

        # State prediction loss (MSE)
        state_loss = F.mse_loss(pred_states, next_states)

        # Action prediction loss (cross-entropy)
        action_loss = F.cross_entropy(
            pred_actions.reshape(-1, self.config.action_dim),
            actions.reshape(-1)
        )

        # Reward prediction loss (MSE)
        reward_loss = F.mse_loss(pred_rewards, rewards)

        # Total loss
        total_loss = (
            self.config.state_loss_weight * state_loss +
            self.config.action_loss_weight * action_loss +
            self.config.reward_loss_weight * reward_loss
        )

        return {
            "total_loss": total_loss,
            "state_loss": state_loss,
            "action_loss": action_loss,
            "reward_loss": reward_loss,
        }

    def generate(
        self,
        states: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        rewards: Optional[torch.Tensor] = None,
        num_steps: int = 5,
        temperature: float = 1.0,
        sample: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate trajectory continuation autoregressively.

        Args:
            states: Initial state(s) (batch, timesteps, state_dim)
            actions: Optional initial actions (batch, timesteps)
            rewards: Optional initial rewards (batch, timesteps, 1)
            num_steps: Number of steps to generate
            temperature: Sampling temperature
            sample: Whether to sample or take argmax

        Returns:
            Dictionary with generated states, actions, rewards
        """
        batch_size, timesteps, _ = states.shape
        device = states.device

        # Initialize with provided data or zeros
        if actions is None:
            actions = torch.zeros(batch_size, timesteps, dtype=torch.long, device=device)
        if rewards is None:
            rewards = torch.zeros(batch_size, timesteps, 1, device=device)

        # Lists to store generated trajectory
        gen_states = [states]
        gen_actions = [actions]
        gen_rewards = [rewards]

        for step in range(num_steps):
            # Get current context (limit to context_length)
            ctx_states = torch.cat(gen_states, dim=1)[:, -self.config.context_length:, :]
            ctx_actions = torch.cat(gen_actions, dim=1)[:, -self.config.context_length:]
            ctx_rewards = torch.cat(gen_rewards, dim=1)[:, -self.config.context_length:, :]

            # Predict next step
            with torch.no_grad():
                pred_states, pred_actions, pred_rewards = self.forward(
                    ctx_states, ctx_actions, ctx_rewards
                )

            # Sample or select next action
            action_logits = pred_actions[:, -1, :] / temperature
            if sample:
                next_action = torch.multinomial(F.softmax(action_logits, dim=-1), num_samples=1)
            else:
                next_action = action_logits.argmax(dim=-1, keepdim=True)

            # Get predicted next state and reward
            next_state = pred_states[:, -1:, :]
            next_reward = pred_rewards[:, -1:, :]

            # Append to trajectory
            gen_actions.append(next_action.unsqueeze(1))
            gen_states.append(next_state)
            gen_rewards.append(next_reward)

        return {
            "states": torch.cat(gen_states, dim=1)[:, timesteps:, :],
            "actions": torch.cat(gen_actions, dim=1)[:, timesteps:],
            "rewards": torch.cat(gen_rewards, dim=1)[:, timesteps:, :],
        }

    def get_action(
        self,
        states: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        rewards: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        sample: bool = True,
    ) -> torch.Tensor:
        """
        Get next action given current trajectory context.

        Args:
            states: State sequence (batch, timesteps, state_dim)
            actions: Optional action history (batch, timesteps-1)
            rewards: Optional reward history (batch, timesteps-1, 1)
            temperature: Sampling temperature
            sample: Whether to sample or take argmax

        Returns:
            Next action (batch,)
        """
        batch_size, timesteps, _ = states.shape
        device = states.device

        # Pad actions and rewards if needed
        if actions is None:
            actions = torch.zeros(batch_size, timesteps, dtype=torch.long, device=device)
        elif actions.shape[1] < timesteps:
            pad = torch.zeros(batch_size, timesteps - actions.shape[1], dtype=torch.long, device=device)
            actions = torch.cat([actions, pad], dim=1)

        if rewards is None:
            rewards = torch.zeros(batch_size, timesteps, 1, device=device)
        elif rewards.shape[1] < timesteps:
            pad = torch.zeros(batch_size, timesteps - rewards.shape[1], 1, device=device)
            rewards = torch.cat([rewards, pad], dim=1)

        # Predict action
        with torch.no_grad():
            _, pred_actions, _ = self.forward(states, actions, rewards)

        # Sample or select action
        action_logits = pred_actions[:, -1, :] / temperature
        if sample:
            action = torch.multinomial(F.softmax(action_logits, dim=-1), num_samples=1).squeeze(1)
        else:
            action = action_logits.argmax(dim=-1)

        return action


def create_trajectory_transformer(
    state_dim: int = 128,
    action_dim: int = 65,
    **kwargs
) -> TrajectoryTransformer:
    """
    Create a Trajectory Transformer model with default or custom configuration.

    Args:
        state_dim: State dimension
        action_dim: Action dimension
        **kwargs: Additional configuration parameters

    Returns:
        Initialized TrajectoryTransformer model

    Example:
        >>> model = create_trajectory_transformer(
        ...     state_dim=128,
        ...     action_dim=65,
        ...     num_layers=4,
        ...     context_length=10
        ... )
    """
    config = TrajectoryTransformerConfig(state_dim=state_dim, action_dim=action_dim, **kwargs)
    return TrajectoryTransformer(config)
