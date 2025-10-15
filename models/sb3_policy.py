"""
Stable-Baselines3 compatible policy wrapper for our custom Transformer network
"""

from typing import Optional

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.distributions import (
    CategoricalDistribution,
    Distribution,
)
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule

from models.networks import ATCActorCritic


class ATCTransformerPolicy(ActorCriticPolicy):
    """
    Custom policy for SB3 that uses our Transformer-based actor-critic network

    This bridges our custom ATCActorCritic network with SB3's training infrastructure
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[list] = None,
        activation_fn: type[nn.Module] = nn.ReLU,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        sde_net_arch: Optional[list] = None,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class=None,
        features_extractor_kwargs: Optional[dict] = None,
        normalize_images: bool = True,
        optimizer_class: type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[dict] = None,
        # Custom parameters for our network
        aircraft_feature_dim: int = 14,
        global_feature_dim: int = 4,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        max_aircraft: int = 20,
    ):
        self.aircraft_feature_dim = aircraft_feature_dim
        self.global_feature_dim = global_feature_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_aircraft = max_aircraft

        # Extract action space sizes from the Dict action space
        if isinstance(action_space, spaces.Dict):
            self.action_space_sizes = {key: space.n for key, space in action_space.spaces.items()}
        else:
            raise ValueError("ATCTransformerPolicy requires Dict action space")

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            sde_net_arch,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        """
        Build our custom Transformer network instead of the default MLP
        """
        self.mlp_extractor = ATCActorCritic(
            aircraft_feature_dim=self.aircraft_feature_dim,
            global_feature_dim=self.global_feature_dim,
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            max_aircraft=self.max_aircraft,
            action_space_sizes=self.action_space_sizes,
        )

    def forward(
        self, obs: dict[str, torch.Tensor], deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network

        Args:
            obs: Dictionary observation
            deterministic: Whether to sample or use mode

        Returns:
            actions: Sampled actions
            values: State values
            log_prob: Log probability of actions
        """
        # Get action logits and values from our network
        action_logits, values = self.mlp_extractor(obs)

        # Sample actions
        actions, log_prob, entropy = self._sample_actions(action_logits, deterministic)

        return actions, values.squeeze(-1), log_prob

    def _sample_actions(
        self, action_logits: dict[str, torch.Tensor], deterministic: bool = False
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Sample actions from the policy distribution

        Args:
            action_logits: Dictionary of logits for each action component
            deterministic: Whether to sample or use mode

        Returns:
            actions: Sampled actions (dict)
            log_prob: Total log probability
            entropy: Total entropy
        """
        actions = {}
        log_probs = []
        entropies = []

        # Sample each action component
        for key, logits in action_logits.items():
            dist = CategoricalDistribution(logits.shape[-1])
            dist.proba_distribution(action_logits=logits)

            if deterministic:
                action = dist.mode()
            else:
                action = dist.sample()

            actions[key] = action
            log_probs.append(dist.log_prob(action))
            entropies.append(dist.entropy())

        # Sum log probs and entropies (independent actions)
        total_log_prob = torch.stack(log_probs).sum(dim=0)
        total_entropy = torch.stack(entropies).sum(dim=0)

        return actions, total_log_prob, total_entropy

    def evaluate_actions(
        self, obs: dict[str, torch.Tensor], actions: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions according to the current policy

        Args:
            obs: Observations
            actions: Actions to evaluate

        Returns:
            values: State values
            log_prob: Log probability of actions
            entropy: Entropy of distribution
        """
        # Forward pass
        action_logits, values = self.mlp_extractor(obs)

        log_probs = []
        entropies = []

        # Evaluate each action component
        for key, logits in action_logits.items():
            dist = CategoricalDistribution(logits.shape[-1])
            dist.proba_distribution(action_logits=logits)

            log_probs.append(dist.log_prob(actions[key]))
            entropies.append(dist.entropy())

        total_log_prob = torch.stack(log_probs).sum(dim=0)
        total_entropy = torch.stack(entropies).sum(dim=0)

        return values.squeeze(-1), total_log_prob, total_entropy

    def predict_values(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Get value predictions for observations

        Args:
            obs: Observations

        Returns:
            values: Predicted state values
        """
        with torch.no_grad():
            _, values = self.mlp_extractor(obs)
        return values.squeeze(-1)

    def get_distribution(self, obs: dict[str, torch.Tensor]) -> Distribution:
        """
        Get the action distribution (required by SB3 but not used for Dict actions)
        """
        raise NotImplementedError(
            "ATCTransformerPolicy uses Dict actions, not a single distribution"
        )

    def _predict(
        self, observation: dict[str, torch.Tensor], deterministic: bool = False
    ) -> dict[str, torch.Tensor]:
        """
        Get the action according to the policy

        Args:
            observation: Observation
            deterministic: Whether to sample or use mode

        Returns:
            actions: Predicted actions
        """
        actions, _, _ = self.forward(observation, deterministic)
        return actions
