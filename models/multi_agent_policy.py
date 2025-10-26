"""
Multi-agent policy architecture for cooperative ATC control.

This module implements a multi-agent reinforcement learning architecture where
each aircraft is an independent agent with a shared policy. Agents communicate
through attention mechanisms and are trained with centralized critic (MAPPO).

Key Features:
- Shared policy network across all agents
- Communication via self-attention between agents
- Centralized critic sees global state
- Decentralized actors see local observations + communication
- Handles variable number of agents (aircraft spawn/exit)
"""

import logging
from typing import Dict, Tuple, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import NetworkConfig, create_default_network_config, validate_network_config
from .encoders import ATCTransformerEncoder, GlobalStateEncoder, AttentionPooling


logger = logging.getLogger(__name__)


class CommunicationModule(nn.Module):
    """
    Communication module for multi-agent coordination using attention.

    This module allows agents to exchange information through self-attention,
    enabling emergent coordination patterns.

    Example:
        >>> comm = CommunicationModule(hidden_dim=256, num_heads=8)
        >>> agent_features = torch.randn(2, 10, 256)  # batch_size=2, num_agents=10
        >>> mask = torch.ones(2, 10, dtype=torch.bool)
        >>> communicated, attention_weights = comm(agent_features, mask)
        >>> print(communicated.shape)  # torch.Size([2, 10, 256])
        >>> print(attention_weights.shape)  # torch.Size([2, 8, 10, 10]) (batch, heads, agents, agents)
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        """
        Initialize communication module.

        Args:
            hidden_dim: Hidden dimension for agent features
            num_heads: Number of attention heads
            num_layers: Number of communication layers
            dropout: Dropout rate
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        # Multi-layer communication via self-attention
        self.comm_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(num_layers)
        ])

        # Layer norms for each communication layer
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])

        # Feed-forward networks after each attention layer
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.Dropout(dropout)
            )
            for _ in range(num_layers)
        ])

        # Final layer norm
        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        agent_features: torch.Tensor,
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass through communication module.

        Args:
            agent_features: Agent features of shape (batch_size, num_agents, hidden_dim)
            mask: Boolean mask of shape (batch_size, num_agents) where True = valid agent

        Returns:
            Tuple of (communicated_features, attention_weights_list)
        """
        # Invert mask for attention (True = ignore)
        attn_mask = ~mask

        x = agent_features
        all_attention_weights = []

        # Apply communication layers
        for i in range(self.num_layers):
            # Self-attention for communication
            attn_out, attn_weights = self.comm_layers[i](
                query=x,
                key=x,
                value=x,
                key_padding_mask=attn_mask,
                need_weights=True,
                average_attn_weights=False  # Get per-head weights
            )
            all_attention_weights.append(attn_weights)

            # Residual connection + layer norm
            x = self.layer_norms[i](x + attn_out)

            # Feed-forward network with residual
            x = x + self.ffns[i](x)

        # Final normalization
        x = self.final_norm(x)

        return x, all_attention_weights


class LocalObservationEncoder(nn.Module):
    """
    Encoder for local agent observations.

    Each agent has its own local view which includes:
    - Its own state (position, velocity, heading, altitude, etc.)
    - Relative positions/states of nearby aircraft
    - Local conflict information
    """

    def __init__(
        self,
        agent_feature_dim: int = 14,  # Single aircraft features
        hidden_dim: int = 256,
        num_nearby: int = 5,  # Consider N nearest neighbors
        dropout: float = 0.1
    ):
        """
        Initialize local observation encoder.

        Args:
            agent_feature_dim: Dimension of single agent features
            hidden_dim: Hidden dimension
            num_nearby: Number of nearby agents to consider
            dropout: Dropout rate
        """
        super().__init__()

        self.agent_feature_dim = agent_feature_dim
        self.hidden_dim = hidden_dim
        self.num_nearby = num_nearby

        # Encoder for own state
        self.own_state_encoder = nn.Sequential(
            nn.Linear(agent_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Encoder for relative observations of nearby agents
        self.neighbor_encoder = nn.Sequential(
            nn.Linear(agent_feature_dim * 2, hidden_dim),  # Own state + relative state
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Attention over neighbors
        self.neighbor_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )

        # Combine own state and neighbor information
        self.combiner = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(
        self,
        own_state: torch.Tensor,
        neighbor_states: torch.Tensor,
        neighbor_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through local observation encoder.

        Args:
            own_state: Own agent state (batch_size, num_agents, agent_feature_dim)
            neighbor_states: Relative neighbor states (batch_size, num_agents, num_nearby, agent_feature_dim)
            neighbor_mask: Neighbor validity mask (batch_size, num_agents, num_nearby)

        Returns:
            Local observation encoding (batch_size, num_agents, hidden_dim)
        """
        batch_size, num_agents, _ = own_state.shape

        # Encode own state
        own_encoded = self.own_state_encoder(own_state)  # (B, N, H)

        # Encode neighbors (if any)
        if neighbor_states is not None and neighbor_states.size(-2) > 0:
            # Expand own state to match neighbor dimensions
            own_expanded = own_state.unsqueeze(2).expand(-1, -1, self.num_nearby, -1)  # (B, N, K, F)

            # Concatenate own state with relative neighbor states
            neighbor_input = torch.cat([own_expanded, neighbor_states], dim=-1)  # (B, N, K, 2F)

            # Reshape for processing
            neighbor_input = neighbor_input.view(batch_size * num_agents, self.num_nearby, -1)
            neighbor_encoded = self.neighbor_encoder(neighbor_input)  # (B*N, K, H)

            # Attention over neighbors
            neighbor_mask_flat = neighbor_mask.view(batch_size * num_agents, self.num_nearby)
            attn_mask = ~neighbor_mask_flat  # Invert for PyTorch attention

            own_query = own_encoded.view(batch_size * num_agents, 1, -1)  # (B*N, 1, H)
            neighbor_pooled, _ = self.neighbor_attention(
                query=own_query,
                key=neighbor_encoded,
                value=neighbor_encoded,
                key_padding_mask=attn_mask
            )  # (B*N, 1, H)

            neighbor_pooled = neighbor_pooled.view(batch_size, num_agents, -1)  # (B, N, H)
        else:
            # No neighbors - use zero vector
            neighbor_pooled = torch.zeros_like(own_encoded)

        # Combine own state and neighbor information
        combined = torch.cat([own_encoded, neighbor_pooled], dim=-1)  # (B, N, 2H)
        local_encoding = self.combiner(combined)  # (B, N, H)

        return local_encoding


class DecentralizedActor(nn.Module):
    """
    Decentralized actor network for each agent.

    Each agent has its own local observation and receives communication from
    other agents, then decides its own action independently.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_command_types: int = 4,
        num_altitude_levels: int = 10,
        num_heading_changes: int = 37,
        num_speed_levels: int = 10,
        dropout: float = 0.1
    ):
        """
        Initialize decentralized actor.

        Args:
            hidden_dim: Hidden dimension
            num_command_types: Number of command types
            num_altitude_levels: Number of altitude levels
            num_heading_changes: Number of heading change options
            num_speed_levels: Number of speed levels
            dropout: Dropout rate
        """
        super().__init__()

        self.hidden_dim = hidden_dim

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Action heads (one per agent outputs its own action components)
        self.command_type_head = nn.Linear(hidden_dim, num_command_types)
        self.altitude_head = nn.Linear(hidden_dim, num_altitude_levels)
        self.heading_head = nn.Linear(hidden_dim, num_heading_changes)
        self.speed_head = nn.Linear(hidden_dim, num_speed_levels)

    def forward(self, agent_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through decentralized actor.

        Args:
            agent_features: Agent features (batch_size, num_agents, hidden_dim)

        Returns:
            Dictionary of action logits for each component
        """
        # Policy features
        policy_features = self.policy_net(agent_features)  # (B, N, H)

        # Action logits for each agent
        action_logits = {
            "command_type": self.command_type_head(policy_features),  # (B, N, num_commands)
            "altitude": self.altitude_head(policy_features),  # (B, N, num_altitudes)
            "heading": self.heading_head(policy_features),  # (B, N, num_headings)
            "speed": self.speed_head(policy_features),  # (B, N, num_speeds)
        }

        return action_logits


class CentralizedCritic(nn.Module):
    """
    Centralized critic network for MAPPO.

    The critic sees the full global state including all agents' observations
    and provides a value estimate for the team.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        global_feature_dim: int = 4,
        dropout: float = 0.1
    ):
        """
        Initialize centralized critic.

        Args:
            hidden_dim: Hidden dimension
            global_feature_dim: Global state feature dimension
            dropout: Dropout rate
        """
        super().__init__()

        self.hidden_dim = hidden_dim

        # Global state encoder
        self.global_encoder = GlobalStateEncoder(
            input_dim=global_feature_dim,
            hidden_dim=hidden_dim,
            num_layers=2,
            dropout=dropout
        )

        # Pooling over all agents
        self.agent_pooling = AttentionPooling(
            config=type('Config', (), {
                'hidden_dim': hidden_dim,
                'num_heads': 8,
                'dropout': dropout
            })()
        )

        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # Agent pool + global state
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(
        self,
        agent_features: torch.Tensor,
        agent_mask: torch.Tensor,
        global_state: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through centralized critic.

        Args:
            agent_features: All agent features (batch_size, num_agents, hidden_dim)
            agent_mask: Agent validity mask (batch_size, num_agents)
            global_state: Global state features (batch_size, global_feature_dim)

        Returns:
            Value estimate (batch_size, 1)
        """
        # Encode global state
        global_encoded = self.global_encoder(global_state)  # (B, H)

        # Pool agent features
        agent_pooled = self.agent_pooling(agent_features, agent_mask)  # (B, H)

        # Combine and compute value
        combined = torch.cat([agent_pooled, global_encoded], dim=-1)  # (B, 2H)
        value = self.value_net(combined)  # (B, 1)

        return value


class MultiAgentPolicy(nn.Module):
    """
    Multi-agent policy architecture for cooperative ATC control.

    This architecture implements:
    1. Shared encoder for all agents (transformer over aircraft)
    2. Communication module (self-attention between agents)
    3. Decentralized actors (each agent outputs its own action)
    4. Centralized critic (sees global state)

    Training uses MAPPO (Multi-Agent PPO):
    - Centralized training: critic sees global state
    - Decentralized execution: actors use local observations + communication

    Example:
        >>> config = create_default_network_config(max_aircraft=10)
        >>> policy = MultiAgentPolicy(config)
        >>> obs = {
        ...     "aircraft": torch.randn(2, 10, 14),
        ...     "aircraft_mask": torch.ones(2, 10, dtype=torch.bool),
        ...     "global_state": torch.randn(2, 4)
        ... }
        >>> action_logits, value, comm_attn = policy(obs)
        >>> print(action_logits["command_type"].shape)  # torch.Size([2, 10, 4])
        >>> print(value.shape)  # torch.Size([2, 1])
    """

    def __init__(self, config: Optional[NetworkConfig] = None):
        """
        Initialize multi-agent policy.

        Args:
            config: Network configuration (uses default if None)
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
        self.hidden_dim = config.encoder_config.hidden_dim

        # Shared encoder for all agents
        self.aircraft_encoder = ATCTransformerEncoder(config.encoder_config)

        # Communication module for agent coordination
        self.communication = CommunicationModule(
            hidden_dim=self.hidden_dim,
            num_heads=8,
            num_layers=2,
            dropout=config.encoder_config.dropout
        )

        # Decentralized actor (shared across all agents)
        self.actor = DecentralizedActor(
            hidden_dim=self.hidden_dim,
            num_command_types=4,
            num_altitude_levels=10,
            num_heading_changes=37,
            num_speed_levels=10,
            dropout=config.encoder_config.dropout
        )

        # Centralized critic
        self.critic = CentralizedCritic(
            hidden_dim=self.hidden_dim,
            global_feature_dim=self.global_feature_dim,
            dropout=config.encoder_config.dropout
        )

        # Initialize weights
        self._init_weights()

        logger.info(f"MultiAgentPolicy initialized: {self.max_aircraft} max agents, "
                   f"{self.hidden_dim} hidden dim")

    def _init_weights(self) -> None:
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        obs: Dict[str, torch.Tensor],
        return_communication: bool = True
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Forward pass through multi-agent policy.

        Args:
            obs: Observation dictionary containing:
                - aircraft: (batch_size, max_aircraft, aircraft_feature_dim)
                - aircraft_mask: (batch_size, max_aircraft)
                - global_state: (batch_size, global_feature_dim)
            return_communication: Whether to return communication attention weights

        Returns:
            Tuple of (action_logits, value, communication_attention)
            - action_logits: Dict with keys [command_type, altitude, heading, speed]
                            Each has shape (batch_size, num_agents, num_actions)
            - value: (batch_size, 1)
            - communication_attention: List of attention weight tensors (if return_communication=True)
        """
        aircraft = obs["aircraft"]
        aircraft_mask = obs["aircraft_mask"]
        global_state = obs["global_state"]

        # 1. Encode all agents through shared encoder
        agent_features = self.aircraft_encoder(aircraft, aircraft_mask)  # (B, N, H)

        # 2. Communication between agents
        communicated_features, comm_attention = self.communication(
            agent_features,
            aircraft_mask
        )  # (B, N, H), List[(B, heads, N, N)]

        # 3. Decentralized actor: each agent outputs its own action
        action_logits = self.actor(communicated_features)  # Dict of (B, N, A_i)

        # 4. Centralized critic: value function sees global state
        value = self.critic(
            communicated_features,
            aircraft_mask,
            global_state
        )  # (B, 1)

        if return_communication:
            return action_logits, value, comm_attention
        else:
            return action_logits, value, None

    def get_action_and_value(
        self,
        obs: Dict[str, torch.Tensor],
        action: Optional[Dict[str, torch.Tensor]] = None,
        agent_ids: Optional[torch.Tensor] = None
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample actions for all active agents or compute log prob of given actions.

        Args:
            obs: Observation dictionary
            action: Optional action dictionary for computing log probabilities.
                   Each component has shape (batch_size, num_agents)
            agent_ids: Optional tensor of active agent indices (batch_size, num_active_agents)
                      If None, uses all agents indicated by mask

        Returns:
            Tuple of (actions, log_probs, entropy, value)
            - actions: Dict with keys [command_type, altitude, heading, speed]
                      Each has shape (batch_size, num_agents)
            - log_probs: (batch_size, num_agents)
            - entropy: (batch_size, num_agents)
            - value: (batch_size, 1)
        """
        # Forward pass
        logits, value, _ = self(obs, return_communication=False)

        batch_size = obs["aircraft"].size(0)
        num_agents = obs["aircraft"].size(1)

        # Create distributions for each action component (per agent)
        command_dist = torch.distributions.Categorical(logits=logits["command_type"])
        altitude_dist = torch.distributions.Categorical(logits=logits["altitude"])
        heading_dist = torch.distributions.Categorical(logits=logits["heading"])
        speed_dist = torch.distributions.Categorical(logits=logits["speed"])

        # Sample or use provided actions
        if action is None:
            action = {
                "command_type": command_dist.sample(),  # (B, N)
                "altitude": altitude_dist.sample(),  # (B, N)
                "heading": heading_dist.sample(),  # (B, N)
                "speed": speed_dist.sample(),  # (B, N)
            }

        # Compute log probabilities (per agent)
        log_prob = (
            command_dist.log_prob(action["command_type"]) +
            altitude_dist.log_prob(action["altitude"]) +
            heading_dist.log_prob(action["heading"]) +
            speed_dist.log_prob(action["speed"])
        )  # (B, N)

        # Compute entropy (per agent)
        entropy = (
            command_dist.entropy() +
            altitude_dist.entropy() +
            heading_dist.entropy() +
            speed_dist.entropy()
        )  # (B, N)

        # Mask out invalid agents
        mask = obs["aircraft_mask"]  # (B, N)
        log_prob = log_prob * mask.float()
        entropy = entropy * mask.float()

        return action, log_prob, entropy, value.squeeze(-1)

    def get_value(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Get value estimate without sampling actions.

        Args:
            obs: Observation dictionary

        Returns:
            Value estimate (batch_size, 1)
        """
        _, value, _ = self(obs, return_communication=False)
        return value

    def count_parameters(self) -> Dict[str, int]:
        """
        Count model parameters.

        Returns:
            Dictionary with parameter counts
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # Count by component
        encoder_params = sum(p.numel() for p in self.aircraft_encoder.parameters())
        comm_params = sum(p.numel() for p in self.communication.parameters())
        actor_params = sum(p.numel() for p in self.actor.parameters())
        critic_params = sum(p.numel() for p in self.critic.parameters())

        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "encoder_parameters": encoder_params,
            "communication_parameters": comm_params,
            "actor_parameters": actor_params,
            "critic_parameters": critic_params,
        }


def create_multi_agent_policy(config: Optional[NetworkConfig] = None) -> MultiAgentPolicy:
    """
    Create multi-agent policy with given configuration.

    Args:
        config: Network configuration (uses default if None)

    Returns:
        MultiAgentPolicy instance
    """
    return MultiAgentPolicy(config)


def create_default_multi_agent_policy(**overrides) -> MultiAgentPolicy:
    """
    Create multi-agent policy with default configuration and optional overrides.

    Args:
        **overrides: Configuration values to override

    Returns:
        MultiAgentPolicy instance
    """
    config = create_default_network_config(**overrides)
    return MultiAgentPolicy(config)
