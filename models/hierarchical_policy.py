"""
Hierarchical policy architecture for ATC with two-level decision making.

This module implements a hierarchical reinforcement learning approach where:
- High-level policy selects which aircraft to command (every N steps)
- Low-level policy generates the specific command for the selected aircraft
- Separate critics evaluate each level of the hierarchy

This dramatically reduces the action space complexity:
- Flat policy: ~51,480 combinations (20 aircraft * 5 commands * 18 altitudes * 13 headings * 8 speeds)
- Hierarchical: ~100 total (20 aircraft at high-level + ~80 at low-level per aircraft)
"""

import logging
from typing import Dict, Tuple, Optional
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from .config import (
    EncoderConfig,
    AttentionPoolingConfig,
    ValueHeadConfig,
    ActivationType,
    create_default_network_config,
)
from .encoders import ATCTransformerEncoder, GlobalStateEncoder, AttentionPooling


logger = logging.getLogger(__name__)


@dataclass
class HighLevelPolicyConfig:
    """Configuration for high-level policy (aircraft selection)."""

    # Architecture
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.1

    # Aircraft selection
    max_aircraft: int = 20
    use_attention: bool = True  # Use attention for aircraft selection

    # Activation
    activation: ActivationType = ActivationType.RELU


@dataclass
class LowLevelPolicyConfig:
    """Configuration for low-level policy (command generation)."""

    # Architecture
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.1

    # Action space sizes (excludes aircraft selection)
    command_type_size: int = 5
    altitude_size: int = 18
    heading_size: int = 13
    speed_size: int = 8

    # Activation
    activation: ActivationType = ActivationType.RELU


@dataclass
class HierarchicalPolicyConfig:
    """Main configuration for hierarchical policy."""

    # Model dimensions
    aircraft_feature_dim: int = 14
    global_feature_dim: int = 4
    max_aircraft: int = 20

    # Temporal abstraction
    option_length: int = 5  # High-level acts every N steps

    # Component configurations
    encoder_config: EncoderConfig = field(default_factory=EncoderConfig)
    attention_pooling_config: AttentionPoolingConfig = field(
        default_factory=AttentionPoolingConfig
    )
    high_level_config: HighLevelPolicyConfig = field(default_factory=HighLevelPolicyConfig)
    low_level_config: LowLevelPolicyConfig = field(default_factory=LowLevelPolicyConfig)
    high_level_value_config: ValueHeadConfig = field(default_factory=ValueHeadConfig)
    low_level_value_config: ValueHeadConfig = field(default_factory=ValueHeadConfig)

    # Intrinsic reward
    use_intrinsic_reward: bool = True
    intrinsic_reward_scale: float = 0.1

    def __post_init__(self):
        """Validate configuration consistency."""
        # Update max_aircraft in high-level config
        self.high_level_config.max_aircraft = self.max_aircraft

        # Validate dimensions
        if self.aircraft_feature_dim <= 0:
            raise ValueError("aircraft_feature_dim must be positive")

        if self.global_feature_dim <= 0:
            raise ValueError("global_feature_dim must be positive")

        if self.max_aircraft <= 0:
            raise ValueError("max_aircraft must be positive")

        if self.option_length <= 0:
            raise ValueError("option_length must be positive")


class HighLevelPolicy(nn.Module):
    """
    High-level policy for aircraft selection.

    This policy selects which aircraft to command based on the current state.
    It uses attention to attend to all aircraft and outputs a distribution over aircraft.
    """

    def __init__(self, config: HighLevelPolicyConfig):
        """
        Initialize high-level policy.

        Args:
            config: High-level policy configuration
        """
        super().__init__()

        self.config = config
        self.hidden_dim = config.hidden_dim
        self.max_aircraft = config.max_aircraft
        self.use_attention = config.use_attention

        # Build shared layers
        shared_layers = []
        for i in range(config.num_layers):
            shared_layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))

            if config.activation == ActivationType.RELU:
                shared_layers.append(nn.ReLU())
            elif config.activation == ActivationType.GELU:
                shared_layers.append(nn.GELU())
            elif config.activation == ActivationType.TANH:
                shared_layers.append(nn.Tanh())

            if config.dropout > 0:
                shared_layers.append(nn.Dropout(config.dropout))

        self.shared_layers = nn.Sequential(*shared_layers)

        if self.use_attention:
            # Aircraft selection via attention
            self.aircraft_attention = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.aircraft_key = nn.Linear(self.hidden_dim, self.hidden_dim)
        else:
            # Direct aircraft selection logits
            self.aircraft_logits = nn.Linear(self.hidden_dim, self.max_aircraft + 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        shared_features: torch.Tensor,
        aircraft_encoded: torch.Tensor,
        aircraft_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through high-level policy.

        Args:
            shared_features: Shared features of shape (batch_size, hidden_dim)
            aircraft_encoded: Aircraft encodings of shape (batch_size, max_aircraft, hidden_dim)
            aircraft_mask: Boolean mask of shape (batch_size, max_aircraft)

        Returns:
            Tuple of (aircraft_logits, attention_weights)
        """
        batch_size = aircraft_mask.size(0)

        # Process shared features
        processed_features = self.shared_layers(shared_features)

        if self.use_attention:
            # Aircraft selection via attention
            query = self.aircraft_attention(processed_features).unsqueeze(
                1
            )  # (batch, 1, hidden)
            keys = self.aircraft_key(aircraft_encoded)  # (batch, max_aircraft, hidden)

            # Attention scores for aircraft selection
            attention_scores = torch.bmm(
                query, keys.transpose(1, 2)
            ).squeeze(1)  # (batch, max_aircraft)

            # Store attention weights for visualization (before masking)
            attention_weights = torch.softmax(
                attention_scores.masked_fill(~aircraft_mask, float("-inf")), dim=-1
            )

            # Add "no action" option
            no_action_logit = torch.zeros(batch_size, 1, device=attention_scores.device)
            aircraft_logits = torch.cat([attention_scores, no_action_logit], dim=1)

            # Mask out invalid aircraft
            mask_expanded = torch.cat(
                [
                    aircraft_mask,
                    torch.ones(batch_size, 1, dtype=torch.bool, device=aircraft_mask.device),
                ],
                dim=1,
            )
            aircraft_logits = aircraft_logits.masked_fill(~mask_expanded, float("-inf"))
        else:
            # Direct logits
            aircraft_logits = self.aircraft_logits(processed_features)

            # Mask out invalid aircraft
            mask_expanded = torch.cat(
                [
                    aircraft_mask,
                    torch.ones(batch_size, 1, dtype=torch.bool, device=aircraft_mask.device),
                ],
                dim=1,
            )
            aircraft_logits = aircraft_logits.masked_fill(~mask_expanded, float("-inf"))

            # Compute attention weights for visualization
            attention_weights = torch.softmax(aircraft_logits[:, :-1], dim=-1)

        return aircraft_logits, attention_weights


class LowLevelPolicy(nn.Module):
    """
    Low-level policy for command generation.

    This policy generates specific commands (altitude, heading, speed) for
    the selected aircraft. It is conditioned on both the global state and
    the specific aircraft state.
    """

    def __init__(self, config: LowLevelPolicyConfig):
        """
        Initialize low-level policy.

        Args:
            config: Low-level policy configuration
        """
        super().__init__()

        self.config = config
        self.hidden_dim = config.hidden_dim

        # Build shared layers
        shared_layers = []
        for i in range(config.num_layers):
            shared_layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))

            if config.activation == ActivationType.RELU:
                shared_layers.append(nn.ReLU())
            elif config.activation == ActivationType.GELU:
                shared_layers.append(nn.GELU())
            elif config.activation == ActivationType.TANH:
                shared_layers.append(nn.Tanh())

            if config.dropout > 0:
                shared_layers.append(nn.Dropout(config.dropout))

        self.shared_layers = nn.Sequential(*shared_layers)

        # Command output heads
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

    def forward(
        self, shared_features: torch.Tensor, selected_aircraft_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through low-level policy.

        Args:
            shared_features: Shared features of shape (batch_size, hidden_dim)
            selected_aircraft_features: Features of selected aircraft (batch_size, hidden_dim)

        Returns:
            Dictionary of command logits
        """
        # Combine shared and aircraft-specific features
        combined_features = shared_features + selected_aircraft_features

        # Process features
        processed_features = self.shared_layers(combined_features)

        # Generate command logits
        command_logits = self.command_head(processed_features)
        altitude_logits = self.altitude_head(processed_features)
        heading_logits = self.heading_head(processed_features)
        speed_logits = self.speed_head(processed_features)

        return {
            "command_type": command_logits,
            "altitude": altitude_logits,
            "heading": heading_logits,
            "speed": speed_logits,
        }


class HierarchicalValueFunction(nn.Module):
    """
    Value function for a single level of the hierarchy.

    Separate value functions for high-level and low-level policies.
    """

    def __init__(self, config: ValueHeadConfig):
        """
        Initialize value function.

        Args:
            config: Value head configuration
        """
        super().__init__()

        self.config = config
        self.hidden_dim = config.hidden_dim

        # Build layers
        layers = []
        for i in range(config.num_layers):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))

            if config.activation == ActivationType.RELU:
                layers.append(nn.ReLU())
            elif config.activation == ActivationType.GELU:
                layers.append(nn.GELU())
            elif config.activation == ActivationType.TANH:
                layers.append(nn.Tanh())

            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))

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
        Forward pass through value function.

        Args:
            x: Input features of shape (batch_size, hidden_dim)

        Returns:
            Value estimates of shape (batch_size, 1)
        """
        return self.value_network(x)


class HierarchicalPolicy(nn.Module):
    """
    Two-level hierarchical policy for ATC.

    Architecture:
    1. Shared encoder processes observations
    2. High-level policy selects aircraft every N steps
    3. Low-level policy generates commands for selected aircraft
    4. Separate critics for each level

    Benefits:
    - Reduced action space complexity
    - Temporal abstraction through options
    - Improved interpretability (can visualize which aircraft and why)
    """

    def __init__(self, config: Optional[HierarchicalPolicyConfig] = None):
        """
        Initialize hierarchical policy.

        Args:
            config: Hierarchical policy configuration (uses default if None)
        """
        super().__init__()

        if config is None:
            config = HierarchicalPolicyConfig()

        self.config = config
        self.max_aircraft = config.max_aircraft
        self.aircraft_feature_dim = config.aircraft_feature_dim
        self.global_feature_dim = config.global_feature_dim
        self.option_length = config.option_length

        # Shared encoders
        self.aircraft_encoder = ATCTransformerEncoder(config.encoder_config)
        self.global_encoder = GlobalStateEncoder(
            input_dim=self.global_feature_dim,
            hidden_dim=config.encoder_config.hidden_dim,
            num_layers=2,
            dropout=config.encoder_config.dropout,
            activation=config.encoder_config.activation.value,
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

        # High-level and low-level policies
        self.high_level_policy = HighLevelPolicy(config.high_level_config)
        self.low_level_policy = LowLevelPolicy(config.low_level_config)

        # Value functions
        self.high_level_value = HierarchicalValueFunction(config.high_level_value_config)
        self.low_level_value = HierarchicalValueFunction(config.low_level_value_config)

        # Option tracking (not saved in state_dict)
        self.register_buffer("current_option", torch.tensor(-1), persistent=False)
        self.register_buffer("option_steps_remaining", torch.tensor(0), persistent=False)

        # Initialize weights
        self._init_weights()

        logger.info(
            f"HierarchicalPolicy initialized: {self.max_aircraft} aircraft, "
            f"{config.encoder_config.hidden_dim} hidden dim, "
            f"option length {self.option_length}"
        )

    def _init_weights(self) -> None:
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _encode_state(
        self, obs: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode observation through shared encoder.

        Args:
            obs: Observation dictionary

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

    def forward(
        self,
        obs: Dict[str, torch.Tensor],
        high_level_only: bool = False,
        selected_aircraft_id: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Forward pass through hierarchical policy.

        Args:
            obs: Observation dictionary
            high_level_only: If True, only compute high-level policy
            selected_aircraft_id: Pre-selected aircraft IDs (batch_size,) for low-level only

        Returns:
            Tuple of (high_level_output, low_level_output, attention_info)
        """
        # Encode state
        shared_features, aircraft_encoded = self._encode_state(obs)

        # High-level policy
        aircraft_logits, attention_weights = self.high_level_policy(
            shared_features, aircraft_encoded, obs["aircraft_mask"]
        )
        high_level_value = self.high_level_value(shared_features)

        high_level_output = {
            "aircraft_logits": aircraft_logits,
            "value": high_level_value,
        }

        attention_info = {"aircraft_attention": attention_weights}

        if high_level_only:
            return high_level_output, {}, attention_info

        # Low-level policy
        batch_size = aircraft_encoded.size(0)

        # Get selected aircraft features
        if selected_aircraft_id is None:
            # Sample from high-level policy
            aircraft_dist = torch.distributions.Categorical(logits=aircraft_logits)
            selected_aircraft_id = aircraft_dist.sample()

        # Extract features for selected aircraft
        selected_aircraft_features = []
        for b in range(batch_size):
            aircraft_id = selected_aircraft_id[b].item()
            if aircraft_id < self.max_aircraft:
                selected_aircraft_features.append(aircraft_encoded[b, aircraft_id])
            else:
                # "No action" selected - use zero features
                selected_aircraft_features.append(
                    torch.zeros_like(aircraft_encoded[b, 0])
                )

        selected_aircraft_features = torch.stack(selected_aircraft_features)

        # Generate low-level commands
        command_logits = self.low_level_policy(shared_features, selected_aircraft_features)
        low_level_value = self.low_level_value(
            shared_features + selected_aircraft_features
        )

        low_level_output = {
            "command_logits": command_logits,
            "value": low_level_value,
            "selected_aircraft_id": selected_aircraft_id,
        }

        return high_level_output, low_level_output, attention_info

    def get_action_and_value(
        self, obs: Dict[str, torch.Tensor], action: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample hierarchical action or compute log prob of given action.

        This method is compatible with the standard actor-critic interface.

        Args:
            obs: Observation dictionary
            action: Optional action dictionary for computing log probabilities

        Returns:
            Tuple of (action, log_prob, entropy, value)
        """
        # Get high-level and low-level outputs
        high_level_out, low_level_out, _ = self.forward(obs)

        # Create distributions
        aircraft_dist = torch.distributions.Categorical(
            logits=high_level_out["aircraft_logits"]
        )

        command_dist = torch.distributions.Categorical(
            logits=low_level_out["command_logits"]["command_type"]
        )
        altitude_dist = torch.distributions.Categorical(
            logits=low_level_out["command_logits"]["altitude"]
        )
        heading_dist = torch.distributions.Categorical(
            logits=low_level_out["command_logits"]["heading"]
        )
        speed_dist = torch.distributions.Categorical(
            logits=low_level_out["command_logits"]["speed"]
        )

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
            aircraft_dist.log_prob(action["aircraft_id"])
            + command_dist.log_prob(action["command_type"])
            + altitude_dist.log_prob(action["altitude"])
            + heading_dist.log_prob(action["heading"])
            + speed_dist.log_prob(action["speed"])
        )

        # Compute entropy
        entropy = (
            aircraft_dist.entropy()
            + command_dist.entropy()
            + altitude_dist.entropy()
            + heading_dist.entropy()
            + speed_dist.entropy()
        )

        # Use high-level value for overall value estimate
        value = high_level_out["value"].squeeze(-1)

        return action, log_prob, entropy, value

    def get_hierarchical_action_and_value(
        self,
        obs: Dict[str, torch.Tensor],
        level: str = "both",
        action: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, any]:
        """
        Get action and value with hierarchical decomposition.

        Args:
            obs: Observation dictionary
            level: Which level to compute ("high", "low", or "both")
            action: Optional action for log prob computation

        Returns:
            Dictionary with hierarchical outputs
        """
        if level not in ["high", "low", "both"]:
            raise ValueError(f"Invalid level: {level}")

        # Encode state
        shared_features, aircraft_encoded = self._encode_state(obs)

        result = {}

        if level in ["high", "both"]:
            # High-level policy
            aircraft_logits, attention_weights = self.high_level_policy(
                shared_features, aircraft_encoded, obs["aircraft_mask"]
            )
            high_level_value = self.high_level_value(shared_features)

            aircraft_dist = torch.distributions.Categorical(logits=aircraft_logits)

            if action is None:
                aircraft_id = aircraft_dist.sample()
            else:
                aircraft_id = action["aircraft_id"]

            result["high_level"] = {
                "action": {"aircraft_id": aircraft_id},
                "log_prob": aircraft_dist.log_prob(aircraft_id),
                "entropy": aircraft_dist.entropy(),
                "value": high_level_value.squeeze(-1),
                "attention_weights": attention_weights,
            }

        if level in ["low", "both"]:
            # Get selected aircraft
            if action is not None:
                aircraft_id = action["aircraft_id"]
            elif "high_level" in result:
                aircraft_id = result["high_level"]["action"]["aircraft_id"]
            else:
                raise ValueError("Need aircraft_id for low-level policy")

            # Extract selected aircraft features
            batch_size = aircraft_encoded.size(0)
            selected_aircraft_features = []
            for b in range(batch_size):
                aid = aircraft_id[b].item()
                if aid < self.max_aircraft:
                    selected_aircraft_features.append(aircraft_encoded[b, aid])
                else:
                    selected_aircraft_features.append(
                        torch.zeros_like(aircraft_encoded[b, 0])
                    )
            selected_aircraft_features = torch.stack(selected_aircraft_features)

            # Low-level policy
            command_logits = self.low_level_policy(
                shared_features, selected_aircraft_features
            )
            low_level_value = self.low_level_value(
                shared_features + selected_aircraft_features
            )

            # Create distributions
            command_dist = torch.distributions.Categorical(
                logits=command_logits["command_type"]
            )
            altitude_dist = torch.distributions.Categorical(logits=command_logits["altitude"])
            heading_dist = torch.distributions.Categorical(logits=command_logits["heading"])
            speed_dist = torch.distributions.Categorical(logits=command_logits["speed"])

            # Sample or use provided action
            if action is None:
                low_level_action = {
                    "command_type": command_dist.sample(),
                    "altitude": altitude_dist.sample(),
                    "heading": heading_dist.sample(),
                    "speed": speed_dist.sample(),
                }
            else:
                low_level_action = {
                    "command_type": action["command_type"],
                    "altitude": action["altitude"],
                    "heading": action["heading"],
                    "speed": action["speed"],
                }

            # Compute log prob and entropy
            log_prob = (
                command_dist.log_prob(low_level_action["command_type"])
                + altitude_dist.log_prob(low_level_action["altitude"])
                + heading_dist.log_prob(low_level_action["heading"])
                + speed_dist.log_prob(low_level_action["speed"])
            )

            entropy = (
                command_dist.entropy()
                + altitude_dist.entropy()
                + heading_dist.entropy()
                + speed_dist.entropy()
            )

            result["low_level"] = {
                "action": low_level_action,
                "log_prob": log_prob,
                "entropy": entropy,
                "value": low_level_value.squeeze(-1),
            }

        return result

    def count_parameters(self) -> Dict[str, int]:
        """
        Count model parameters.

        Returns:
            Dictionary with parameter counts
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        high_level_params = sum(p.numel() for p in self.high_level_policy.parameters())
        low_level_params = sum(p.numel() for p in self.low_level_policy.parameters())
        shared_params = total_params - high_level_params - low_level_params

        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "high_level_parameters": high_level_params,
            "low_level_parameters": low_level_params,
            "shared_parameters": shared_params,
        }


def create_hierarchical_policy(
    config: Optional[HierarchicalPolicyConfig] = None,
) -> HierarchicalPolicy:
    """
    Create hierarchical policy with given configuration.

    Args:
        config: Hierarchical policy configuration (uses default if None)

    Returns:
        HierarchicalPolicy instance
    """
    return HierarchicalPolicy(config)
