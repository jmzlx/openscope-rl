"""
Beam search planner for Trajectory Transformer.

This module implements beam search planning using the trained trajectory transformer
as a world model. It performs n-step lookahead by simulating trajectories and scoring
them based on predicted rewards.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
import logging

from models.trajectory_transformer import TrajectoryTransformer
from environment.utils import get_device


logger = logging.getLogger(__name__)


@dataclass
class PlannerConfig:
    """Configuration for beam search planner."""

    # Beam search
    beam_width: int = 5
    lookahead_steps: int = 5

    # Action selection
    temperature: float = 1.0
    top_k: int = 10  # Consider top-k actions at each step

    # Scoring
    discount_factor: float = 0.99
    reward_weight: float = 1.0
    state_quality_weight: float = 0.1  # Weight for predicted state quality

    # Diversity
    diversity_penalty: float = 0.0  # Penalty for similar actions

    # Device
    device: str = field(default_factory=lambda: get_device())


class BeamSearchPlanner:
    """
    Beam search planner using Trajectory Transformer as world model.

    This planner uses the trained trajectory transformer to simulate future trajectories
    and select actions based on predicted cumulative rewards.

    Example:
        >>> from models import TrajectoryTransformer, TrajectoryTransformerConfig
        >>> model_config = TrajectoryTransformerConfig(state_dim=128, action_dim=65)
        >>> model = TrajectoryTransformer(model_config)
        >>>
        >>> planner_config = PlannerConfig(beam_width=5, lookahead_steps=5)
        >>> planner = BeamSearchPlanner(model, planner_config)
        >>>
        >>> # Get best action
        >>> state = torch.randn(1, 1, 128)
        >>> action = planner.plan(state)
    """

    def __init__(
        self,
        model: TrajectoryTransformer,
        config: PlannerConfig,
    ):
        """
        Initialize beam search planner.

        Args:
            model: Trained trajectory transformer model
            config: Planner configuration
        """
        self.model = model
        self.config = config

        # Move model to device and set to eval mode
        self.model.to(self.config.device)
        self.model.eval()

        # Statistics
        self.num_plans = 0
        self.avg_beam_scores = []

    def plan(
        self,
        states: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        rewards: Optional[torch.Tensor] = None,
        return_all_beams: bool = False,
    ) -> torch.Tensor:
        """
        Plan best action using beam search.

        Args:
            states: Current state sequence (batch, timesteps, state_dim)
            actions: Optional action history (batch, timesteps)
            rewards: Optional reward history (batch, timesteps, 1)
            return_all_beams: Whether to return all beam candidates

        Returns:
            Best action (batch,) or all beam actions if return_all_beams=True
        """
        batch_size = states.shape[0]

        if batch_size > 1:
            # Process each batch item separately
            best_actions = []
            for i in range(batch_size):
                state_i = states[i:i+1]
                action_i = actions[i:i+1] if actions is not None else None
                reward_i = rewards[i:i+1] if rewards is not None else None

                action = self._plan_single(state_i, action_i, reward_i)
                best_actions.append(action)

            return torch.cat(best_actions)
        else:
            return self._plan_single(states, actions, rewards, return_all_beams)

    def _plan_single(
        self,
        states: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        rewards: Optional[torch.Tensor] = None,
        return_all_beams: bool = False,
    ) -> torch.Tensor:
        """
        Plan for a single trajectory using beam search.

        Args:
            states: State sequence (1, timesteps, state_dim)
            actions: Optional action history (1, timesteps)
            rewards: Optional reward history (1, timesteps, 1)
            return_all_beams: Whether to return all beam candidates

        Returns:
            Best action or all beam actions
        """
        # Move to device
        states = states.to(self.config.device)
        if actions is not None:
            actions = actions.to(self.config.device)
        if rewards is not None:
            rewards = rewards.to(self.config.device)

        # Initialize beams
        # Each beam: (state_seq, action_seq, reward_seq, score)
        initial_beam = {
            "states": states,
            "actions": actions if actions is not None else torch.zeros(
                1, states.shape[1], dtype=torch.long, device=self.config.device
            ),
            "rewards": rewards if rewards is not None else torch.zeros(
                1, states.shape[1], 1, device=self.config.device
            ),
            "score": 0.0,
        }

        beams = [initial_beam]

        # Beam search
        with torch.no_grad():
            for step in range(self.config.lookahead_steps):
                new_beams = []

                for beam in beams:
                    # Get next action distribution
                    _, pred_actions, pred_rewards = self.model.forward(
                        beam["states"], beam["actions"], beam["rewards"]
                    )

                    # Get action logits for last timestep
                    action_logits = pred_actions[:, -1, :] / self.config.temperature

                    # Get top-k actions
                    top_k = min(self.config.top_k, action_logits.shape[-1])
                    top_probs, top_actions = torch.topk(
                        F.softmax(action_logits, dim=-1), k=top_k
                    )

                    # Expand beam with each top-k action
                    for k in range(top_k):
                        action = top_actions[0, k]
                        action_prob = top_probs[0, k]

                        # Predict next state and reward with this action
                        next_action = action.unsqueeze(0).unsqueeze(0)

                        # Get predictions for this action
                        temp_actions = torch.cat([beam["actions"], next_action], dim=1)
                        next_state_pred, _, next_reward_pred = self.model.forward(
                            beam["states"], temp_actions, beam["rewards"]
                        )

                        # Use predicted state and reward
                        next_state = next_state_pred[:, -1:, :]
                        next_reward = next_reward_pred[:, -1:, :]

                        # Update sequences
                        new_states = torch.cat([beam["states"], next_state], dim=1)
                        new_actions = torch.cat([beam["actions"], next_action], dim=1)
                        new_rewards = torch.cat([beam["rewards"], next_reward], dim=1)

                        # Limit context length
                        if new_states.shape[1] > self.model.config.context_length:
                            new_states = new_states[:, -self.model.config.context_length:, :]
                            new_actions = new_actions[:, -self.model.config.context_length:]
                            new_rewards = new_rewards[:, -self.model.config.context_length:, :]

                        # Compute score
                        discount = self.config.discount_factor ** step
                        reward_score = next_reward.item() * discount * self.config.reward_weight

                        # Action probability score
                        prob_score = torch.log(action_prob).item()

                        # State quality score (negative MSE of state prediction)
                        # This encourages trajectories where the model is confident
                        state_quality = -torch.mean((next_state_pred[:, -1, :] - next_state.squeeze()) ** 2).item()
                        state_quality *= self.config.state_quality_weight

                        # Total score
                        new_score = beam["score"] + reward_score + prob_score + state_quality

                        new_beams.append({
                            "states": new_states,
                            "actions": new_actions,
                            "rewards": new_rewards,
                            "score": new_score,
                        })

                # Keep top beam_width beams
                new_beams.sort(key=lambda x: x["score"], reverse=True)
                beams = new_beams[:self.config.beam_width]

        # Get best beam
        best_beam = beams[0]

        # Statistics
        self.num_plans += 1
        self.avg_beam_scores.append(best_beam["score"])

        if return_all_beams:
            # Return first action from each beam
            all_actions = torch.tensor([
                beam["actions"][0, states.shape[1]].item()
                for beam in beams
            ], device=self.config.device)
            return all_actions
        else:
            # Return first action from best beam (after initial context)
            best_action = best_beam["actions"][0, states.shape[1]]
            return best_action.unsqueeze(0)

    def evaluate_trajectory(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Evaluate a trajectory using the world model.

        Args:
            states: State sequence (batch, timesteps, state_dim)
            actions: Action sequence (batch, timesteps)
            rewards: Reward sequence (batch, timesteps, 1)

        Returns:
            Dictionary with evaluation metrics
        """
        # Move to device
        states = states.to(self.config.device)
        actions = actions.to(self.config.device)
        rewards = rewards.to(self.config.device)

        with torch.no_grad():
            # Get predictions
            pred_states, pred_actions, pred_rewards = self.model.forward(
                states, actions, rewards
            )

            # Compute metrics
            # State prediction error
            next_states = torch.cat([states[:, 1:, :], torch.zeros_like(states[:, :1, :])], dim=1)
            state_mse = F.mse_loss(pred_states, next_states).item()

            # Action prediction accuracy
            action_acc = (pred_actions.argmax(dim=-1) == actions).float().mean().item()

            # Reward prediction error
            reward_mse = F.mse_loss(pred_rewards, rewards).item()

            # Total predicted return
            predicted_return = pred_rewards.sum().item()
            actual_return = rewards.sum().item()

        return {
            "state_mse": state_mse,
            "action_accuracy": action_acc,
            "reward_mse": reward_mse,
            "predicted_return": predicted_return,
            "actual_return": actual_return,
            "return_error": abs(predicted_return - actual_return),
        }

    def get_statistics(self) -> Dict[str, float]:
        """
        Get planner statistics.

        Returns:
            Dictionary of statistics
        """
        return {
            "num_plans": self.num_plans,
            "avg_beam_score": np.mean(self.avg_beam_scores) if self.avg_beam_scores else 0.0,
            "beam_scores": self.avg_beam_scores,
        }

    def reset_statistics(self):
        """Reset planner statistics."""
        self.num_plans = 0
        self.avg_beam_scores = []


def create_planner(
    model: TrajectoryTransformer,
    beam_width: int = 5,
    lookahead_steps: int = 5,
    **kwargs
) -> BeamSearchPlanner:
    """
    Create a beam search planner with default or custom configuration.

    Args:
        model: Trained trajectory transformer model
        beam_width: Number of beams to maintain
        lookahead_steps: Number of steps to look ahead
        **kwargs: Additional configuration parameters

    Returns:
        Initialized BeamSearchPlanner

    Example:
        >>> planner = create_planner(
        ...     model,
        ...     beam_width=10,
        ...     lookahead_steps=5,
        ...     temperature=0.8
        ... )
    """
    config = PlannerConfig(
        beam_width=beam_width,
        lookahead_steps=lookahead_steps,
        **kwargs
    )
    return BeamSearchPlanner(model, config)
