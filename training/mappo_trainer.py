"""
MAPPO (Multi-Agent PPO) trainer for cooperative ATC control.

This module implements the MAPPO algorithm for training multi-agent policies
with centralized training and decentralized execution (CTDE).

Key Features:
- Centralized value function (critic sees global state)
- Decentralized policy (actors see local observations + communication)
- Handles variable number of agents (aircraft spawn/exit)
- Vectorized advantage computation per agent
- Communication visualization and analysis

References:
    Yu et al. "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games"
    https://arxiv.org/abs/2103.01955
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from models import MultiAgentPolicy, create_default_network_config


logger = logging.getLogger(__name__)


@dataclass
class MAPPOConfig:
    """Configuration for MAPPO training."""

    # Environment settings
    max_aircraft: int = 10
    aircraft_feature_dim: int = 14
    global_feature_dim: int = 4

    # Network settings
    hidden_dim: int = 256
    num_encoder_layers: int = 4
    num_attention_heads: int = 8

    # Training hyperparameters
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5

    # PPO settings
    num_epochs: int = 10
    num_minibatches: int = 4
    batch_size: int = 2048
    steps_per_rollout: int = 2048

    # Multi-agent specific
    use_value_clipping: bool = True
    use_huber_loss: bool = False
    huber_delta: float = 10.0

    # Training settings
    total_timesteps: int = 1_000_000
    save_interval: int = 10_000
    log_interval: int = 100

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Logging
    log_dir: str = "logs/mappo"
    save_dir: str = "checkpoints/mappo"


class MAPPOTrainer:
    """
    MAPPO trainer for multi-agent cooperative control.

    This trainer implements the Multi-Agent PPO algorithm with:
    - Centralized critic (sees global state)
    - Decentralized actors (see local observations + communication)
    - Per-agent advantage computation
    - Vectorized training for variable number of agents
    """

    def __init__(
        self,
        env,
        config: Optional[MAPPOConfig] = None,
        policy: Optional[MultiAgentPolicy] = None
    ):
        """
        Initialize MAPPO trainer.

        Args:
            env: Gymnasium environment
            config: Training configuration
            policy: Multi-agent policy (creates default if None)
        """
        self.env = env
        self.config = config if config is not None else MAPPOConfig()

        # Create policy if not provided
        if policy is None:
            network_config = create_default_network_config(
                max_aircraft=self.config.max_aircraft,
                aircraft_feature_dim=self.config.aircraft_feature_dim,
                global_feature_dim=self.config.global_feature_dim,
                hidden_dim=self.config.hidden_dim,
                num_encoder_layers=self.config.num_encoder_layers,
                num_attention_heads=self.config.num_attention_heads,
            )
            self.policy = MultiAgentPolicy(network_config).to(self.config.device)
        else:
            self.policy = policy.to(self.config.device)

        # Optimizer
        self.optimizer = optim.Adam(
            self.policy.parameters(),
            lr=self.config.learning_rate
        )

        # Logging
        self.writer = SummaryWriter(log_dir=self.config.log_dir)

        # Statistics
        self.global_step = 0
        self.num_updates = 0

        logger.info(f"MAPPO Trainer initialized with config: {self.config}")
        logger.info(f"Policy parameters: {self.policy.count_parameters()}")

    def collect_rollout(
        self,
        num_steps: int
    ) -> Dict[str, torch.Tensor]:
        """
        Collect rollout data from environment.

        Args:
            num_steps: Number of steps to collect

        Returns:
            Dictionary containing rollout data
        """
        # Storage for rollout
        obs_list = []
        action_list = []
        log_prob_list = []
        reward_list = []
        done_list = []
        value_list = []
        mask_list = []

        obs, info = self.env.reset()

        for step in range(num_steps):
            # Convert observation to tensor
            obs_tensor = self._obs_to_tensor(obs)
            mask_list.append(obs_tensor["aircraft_mask"])

            # Get action and value
            with torch.no_grad():
                action, log_prob, _, value = self.policy.get_action_and_value(obs_tensor)

            # Convert action to environment format
            env_action = self._action_to_env(action, obs_tensor["aircraft_mask"])

            # Step environment
            next_obs, reward, terminated, truncated, info = self.env.step(env_action)
            done = terminated or truncated

            # Store data
            obs_list.append(obs_tensor)
            action_list.append(action)
            log_prob_list.append(log_prob)
            reward_list.append(torch.tensor([reward], dtype=torch.float32))
            done_list.append(torch.tensor([done], dtype=torch.float32))
            value_list.append(value)

            obs = next_obs
            self.global_step += 1

            if done:
                obs, info = self.env.reset()

        # Stack all data
        rollout = {
            "observations": self._stack_obs_list(obs_list),
            "actions": self._stack_action_dict(action_list),
            "log_probs": torch.stack(log_prob_list),  # (T, B, N)
            "rewards": torch.stack(reward_list),  # (T, B)
            "dones": torch.stack(done_list),  # (T, B)
            "values": torch.stack(value_list),  # (T, B)
            "masks": torch.stack(mask_list),  # (T, B, N)
        }

        return rollout

    def compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        next_value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute GAE advantages and returns.

        Args:
            rewards: Reward tensor (T, B)
            values: Value estimates (T, B)
            dones: Done flags (T, B)
            next_value: Next state value (B,)

        Returns:
            Tuple of (advantages, returns)
        """
        T, B = rewards.shape
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)

        # Compute advantages using GAE
        gae = 0
        for t in reversed(range(T)):
            if t == T - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]

            # TD error
            delta = rewards[t] + self.config.gamma * next_val * (1 - dones[t]) - values[t]

            # GAE
            gae = delta + self.config.gamma * self.config.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae

        # Returns = advantages + values
        returns = advantages + values

        return advantages, returns

    def update_policy(
        self,
        rollout: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Update policy using PPO.

        Args:
            rollout: Rollout data

        Returns:
            Dictionary of training metrics
        """
        # Move data to device
        obs = {k: v.to(self.config.device) for k, v in rollout["observations"].items()}
        actions = {k: v.to(self.config.device) for k, v in rollout["actions"].items()}
        old_log_probs = rollout["log_probs"].to(self.config.device)
        rewards = rollout["rewards"].to(self.config.device)
        dones = rollout["dones"].to(self.config.device)
        old_values = rollout["values"].to(self.config.device)
        masks = rollout["masks"].to(self.config.device)

        # Compute next value for advantage calculation
        with torch.no_grad():
            # Get last observation
            last_obs = {k: v[-1:] for k, v in obs.items()}
            next_value = self.policy.get_value(last_obs).squeeze(0)

        # Compute advantages and returns
        advantages, returns = self.compute_advantages(
            rewards, old_values, dones, next_value
        )

        # Flatten time and batch dimensions
        T, B, N = old_log_probs.shape
        num_samples = T * B

        # Reshape data
        obs_flat = {k: v.view(T * B, *v.shape[2:]) for k, v in obs.items()}
        actions_flat = {k: v.view(T * B, *v.shape[2:]) for k, v in actions.items()}
        old_log_probs_flat = old_log_probs.view(T * B, N)
        advantages_flat = advantages.view(T * B, 1).expand(T * B, N)
        returns_flat = returns.view(T * B, 1).expand(T * B, N)
        old_values_flat = old_values.view(T * B, 1).expand(T * B, N)
        masks_flat = masks.view(T * B, N)

        # Normalize advantages per agent (across active agents only)
        active_mask = masks_flat.bool()
        for i in range(T * B):
            if active_mask[i].any():
                agent_advantages = advantages_flat[i, active_mask[i]]
                advantages_flat[i, active_mask[i]] = (
                    agent_advantages - agent_advantages.mean()
                ) / (agent_advantages.std() + 1e-8)

        # Training metrics
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_approx_kl = 0.0
        total_clipfrac = 0.0
        num_updates_this_epoch = 0

        # PPO epochs
        for epoch in range(self.config.num_epochs):
            # Shuffle data
            indices = torch.randperm(num_samples)

            # Mini-batch updates
            minibatch_size = num_samples // self.config.num_minibatches

            for start in range(0, num_samples, minibatch_size):
                end = start + minibatch_size
                mb_indices = indices[start:end]

                # Mini-batch data
                mb_obs = {k: v[mb_indices] for k, v in obs_flat.items()}
                mb_actions = {k: v[mb_indices] for k, v in actions_flat.items()}
                mb_old_log_probs = old_log_probs_flat[mb_indices]
                mb_advantages = advantages_flat[mb_indices]
                mb_returns = returns_flat[mb_indices]
                mb_old_values = old_values_flat[mb_indices]
                mb_masks = masks_flat[mb_indices]

                # Forward pass
                _, new_log_probs, entropy, new_values = self.policy.get_action_and_value(
                    mb_obs, mb_actions
                )

                # Expand new_values to match per-agent shape
                new_values = new_values.unsqueeze(-1).expand(-1, N)

                # Compute losses (per agent, then average over active agents)
                ratio = torch.exp(new_log_probs - mb_old_log_probs)

                # Policy loss (clipped)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(
                    ratio,
                    1 - self.config.clip_epsilon,
                    1 + self.config.clip_epsilon
                ) * mb_advantages

                # Only consider active agents
                active_mask = mb_masks.bool()
                policy_loss_per_agent = -torch.min(surr1, surr2)
                policy_loss = (policy_loss_per_agent * mb_masks).sum() / mb_masks.sum()

                # Value loss
                if self.config.use_value_clipping:
                    value_pred_clipped = mb_old_values + torch.clamp(
                        new_values - mb_old_values,
                        -self.config.clip_epsilon,
                        self.config.clip_epsilon
                    )
                    value_loss_1 = (new_values - mb_returns) ** 2
                    value_loss_2 = (value_pred_clipped - mb_returns) ** 2
                    value_loss_per_agent = torch.max(value_loss_1, value_loss_2)
                else:
                    if self.config.use_huber_loss:
                        value_loss_per_agent = nn.functional.huber_loss(
                            new_values,
                            mb_returns,
                            delta=self.config.huber_delta,
                            reduction='none'
                        )
                    else:
                        value_loss_per_agent = (new_values - mb_returns) ** 2

                value_loss = (value_loss_per_agent * mb_masks).sum() / mb_masks.sum()

                # Entropy (for exploration)
                entropy_loss = -(entropy * mb_masks).sum() / mb_masks.sum()

                # Total loss
                loss = (
                    policy_loss +
                    self.config.value_loss_coef * value_loss +
                    self.config.entropy_coef * entropy_loss
                )

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                # Track metrics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += -entropy_loss.item()

                # Approximate KL divergence
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - torch.log(ratio)).mean()
                    clipfrac = ((ratio - 1).abs() > self.config.clip_epsilon).float().mean()
                    total_approx_kl += approx_kl.item()
                    total_clipfrac += clipfrac.item()

                num_updates_this_epoch += 1

        self.num_updates += 1

        # Return average metrics
        return {
            "policy_loss": total_policy_loss / num_updates_this_epoch,
            "value_loss": total_value_loss / num_updates_this_epoch,
            "entropy": total_entropy / num_updates_this_epoch,
            "approx_kl": total_approx_kl / num_updates_this_epoch,
            "clipfrac": total_clipfrac / num_updates_this_epoch,
        }

    def train(self) -> None:
        """Main training loop."""
        logger.info("Starting MAPPO training...")

        start_time = time.time()
        num_rollouts = self.config.total_timesteps // self.config.steps_per_rollout

        for rollout_idx in range(num_rollouts):
            # Collect rollout
            rollout = self.collect_rollout(self.config.steps_per_rollout)

            # Update policy
            metrics = self.update_policy(rollout)

            # Logging
            if rollout_idx % self.config.log_interval == 0:
                elapsed_time = time.time() - start_time
                steps_per_sec = self.global_step / elapsed_time

                logger.info(
                    f"Rollout {rollout_idx}/{num_rollouts} | "
                    f"Steps: {self.global_step} | "
                    f"Policy Loss: {metrics['policy_loss']:.4f} | "
                    f"Value Loss: {metrics['value_loss']:.4f} | "
                    f"Entropy: {metrics['entropy']:.4f} | "
                    f"KL: {metrics['approx_kl']:.4f} | "
                    f"Steps/sec: {steps_per_sec:.1f}"
                )

                # TensorBoard logging
                self.writer.add_scalar("train/policy_loss", metrics["policy_loss"], self.global_step)
                self.writer.add_scalar("train/value_loss", metrics["value_loss"], self.global_step)
                self.writer.add_scalar("train/entropy", metrics["entropy"], self.global_step)
                self.writer.add_scalar("train/approx_kl", metrics["approx_kl"], self.global_step)
                self.writer.add_scalar("train/clipfrac", metrics["clipfrac"], self.global_step)
                self.writer.add_scalar("train/steps_per_sec", steps_per_sec, self.global_step)

            # Save checkpoint
            if rollout_idx % self.config.save_interval == 0:
                self.save_checkpoint(rollout_idx)

        logger.info("Training complete!")
        self.writer.close()

    def save_checkpoint(self, rollout_idx: int) -> None:
        """Save model checkpoint."""
        import os
        os.makedirs(self.config.save_dir, exist_ok=True)

        checkpoint_path = os.path.join(
            self.config.save_dir,
            f"checkpoint_{rollout_idx}.pt"
        )

        torch.save({
            "rollout_idx": rollout_idx,
            "global_step": self.global_step,
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
        }, checkpoint_path)

        logger.info(f"Saved checkpoint to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device)

        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint["global_step"]

        logger.info(f"Loaded checkpoint from {checkpoint_path}")

    def _obs_to_tensor(self, obs: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """Convert observation to tensor."""
        return {
            "aircraft": torch.from_numpy(obs["aircraft"]).unsqueeze(0).float(),
            "aircraft_mask": torch.from_numpy(obs["aircraft_mask"]).unsqueeze(0).bool(),
            "global_state": torch.from_numpy(obs["global_state"]).unsqueeze(0).float(),
        }

    def _action_to_env(
        self,
        action: Dict[str, torch.Tensor],
        mask: torch.Tensor
    ) -> Dict[str, int]:
        """
        Convert multi-agent action to environment format.

        For now, we select the first active agent's action.
        In future, we could implement proper multi-agent action selection.
        """
        # Find first active agent
        active_agents = mask[0].nonzero(as_tuple=True)[0]

        if len(active_agents) == 0:
            # No active agents - return no-op
            return {
                "aircraft_id": 0,
                "command_type": 0,
                "altitude": 0,
                "heading": 0,
                "speed": 0,
            }

        # Select first active agent
        agent_id = active_agents[0].item()

        return {
            "aircraft_id": agent_id + 1,  # +1 because 0 is no-op
            "command_type": action["command_type"][0, agent_id].item(),
            "altitude": action["altitude"][0, agent_id].item(),
            "heading": action["heading"][0, agent_id].item(),
            "speed": action["speed"][0, agent_id].item(),
        }

    def _stack_obs_list(self, obs_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Stack list of observations into single tensor dict."""
        return {
            "aircraft": torch.stack([obs["aircraft"] for obs in obs_list]),
            "aircraft_mask": torch.stack([obs["aircraft_mask"] for obs in obs_list]),
            "global_state": torch.stack([obs["global_state"] for obs in obs_list]),
        }

    def _stack_action_dict(self, action_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Stack list of action dicts into single tensor dict."""
        return {
            "command_type": torch.stack([a["command_type"] for a in action_list]),
            "altitude": torch.stack([a["altitude"] for a in action_list]),
            "heading": torch.stack([a["heading"] for a in action_list]),
            "speed": torch.stack([a["speed"] for a in action_list]),
        }


def train_mappo(
    env,
    config: Optional[MAPPOConfig] = None,
    policy: Optional[MultiAgentPolicy] = None
) -> MAPPOTrainer:
    """
    Train multi-agent policy using MAPPO.

    Args:
        env: Gymnasium environment
        config: Training configuration
        policy: Multi-agent policy (creates default if None)

    Returns:
        Trained MAPPOTrainer instance
    """
    trainer = MAPPOTrainer(env, config, policy)
    trainer.train()
    return trainer
