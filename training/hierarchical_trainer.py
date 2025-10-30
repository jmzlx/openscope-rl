"""
Hierarchical PPO trainer with options framework.

This module implements training for hierarchical policies with:
- Temporal abstraction (options framework)
- Separate PPO updates for high-level and low-level policies
- Intrinsic rewards for high-level exploration
- WandB logging for monitoring and visualization
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available, logging disabled")


logger = logging.getLogger(__name__)


@dataclass
class HierarchicalPPOConfig:
    """Configuration for hierarchical PPO training."""

    # Training
    total_timesteps: int = 1_000_000
    num_envs: int = 4
    num_steps: int = 128  # Steps per rollout
    num_minibatches: int = 4
    num_epochs: int = 4

    # Learning rates
    high_level_lr: float = 3e-4
    low_level_lr: float = 3e-4
    lr_anneal: bool = True

    # PPO parameters
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Intrinsic reward
    use_intrinsic_reward: bool = True
    intrinsic_reward_scale: float = 0.1
    intrinsic_reward_type: str = "option_change"  # "option_change" or "diversity"

    # Options
    option_length: int = 5  # High-level acts every N steps

    # Logging
    log_interval: int = 10
    save_interval: int = 100
    eval_interval: int = 50
    use_wandb: bool = False
    wandb_project: str = "openscope-hierarchical-rl"
    wandb_entity: Optional[str] = None

    # Checkpoint
    checkpoint_dir: str = "checkpoints/hierarchical"


class HierarchicalRolloutBuffer:
    """
    Rollout buffer for hierarchical policies.

    Stores separate trajectories for high-level and low-level policies.
    """

    def __init__(
        self,
        buffer_size: int,
        num_envs: int,
        obs_space_shape: Dict[str, Tuple],
        device: str = "cpu",
    ):
        """
        Initialize rollout buffer.

        Args:
            buffer_size: Number of steps to store per environment
            num_envs: Number of parallel environments
            obs_space_shape: Observation space shape dictionary
            device: Device to store tensors
        """
        self.buffer_size = buffer_size
        self.num_envs = num_envs
        self.device = device

        # Observations
        self.obs = {
            key: np.zeros((buffer_size, num_envs, *shape), dtype=np.float32)
            for key, shape in obs_space_shape.items()
        }

        # High-level data
        self.high_level_actions = np.zeros((buffer_size, num_envs), dtype=np.int64)
        self.high_level_logprobs = np.zeros((buffer_size, num_envs), dtype=np.float32)
        self.high_level_values = np.zeros((buffer_size, num_envs), dtype=np.float32)
        self.high_level_rewards = np.zeros((buffer_size, num_envs), dtype=np.float32)

        # Low-level data
        self.low_level_actions = {
            "command_type": np.zeros((buffer_size, num_envs), dtype=np.int64),
            "altitude": np.zeros((buffer_size, num_envs), dtype=np.int64),
            "heading": np.zeros((buffer_size, num_envs), dtype=np.int64),
            "speed": np.zeros((buffer_size, num_envs), dtype=np.int64),
        }
        self.low_level_logprobs = np.zeros((buffer_size, num_envs), dtype=np.float32)
        self.low_level_values = np.zeros((buffer_size, num_envs), dtype=np.float32)
        self.low_level_rewards = np.zeros((buffer_size, num_envs), dtype=np.float32)

        # Episode info
        self.dones = np.zeros((buffer_size, num_envs), dtype=np.bool_)
        self.option_mask = np.zeros(
            (buffer_size, num_envs), dtype=np.bool_
        )  # True when high-level acts

        # Attention weights for visualization
        self.attention_weights = np.zeros(
            (buffer_size, num_envs, 20), dtype=np.float32
        )  # Assuming max 20 aircraft

        self.pos = 0
        self.full = False

    def add(
        self,
        obs: Dict[str, np.ndarray],
        high_level_action: np.ndarray,
        high_level_logprob: np.ndarray,
        high_level_value: np.ndarray,
        low_level_actions: Dict[str, np.ndarray],
        low_level_logprob: np.ndarray,
        low_level_value: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        is_option_step: np.ndarray,
        attention_weights: np.ndarray,
    ):
        """Add experience to buffer."""
        for key in self.obs.keys():
            self.obs[key][self.pos] = obs[key]

        self.high_level_actions[self.pos] = high_level_action
        self.high_level_logprobs[self.pos] = high_level_logprob
        self.high_level_values[self.pos] = high_level_value

        for key in self.low_level_actions.keys():
            self.low_level_actions[key][self.pos] = low_level_actions[key]
        self.low_level_logprobs[self.pos] = low_level_logprob
        self.low_level_values[self.pos] = low_level_value

        # Both levels receive the same environmental reward
        self.high_level_rewards[self.pos] = reward
        self.low_level_rewards[self.pos] = reward

        self.dones[self.pos] = done
        self.option_mask[self.pos] = is_option_step
        self.attention_weights[self.pos] = attention_weights

        self.pos = (self.pos + 1) % self.buffer_size
        if self.pos == 0:
            self.full = True

    def get(self, level: str = "both") -> Dict[str, torch.Tensor]:
        """
        Get buffer contents.

        Args:
            level: Which level to get ("high", "low", or "both")

        Returns:
            Dictionary of tensors
        """
        size = self.buffer_size if self.full else self.pos

        result = {
            "obs": {
                key: torch.from_numpy(self.obs[key][:size]).to(self.device)
                for key in self.obs.keys()
            },
            "dones": torch.from_numpy(self.dones[:size]).to(self.device),
        }

        if level in ["high", "both"]:
            result["high_level"] = {
                "actions": torch.from_numpy(self.high_level_actions[:size]).to(
                    self.device
                ),
                "logprobs": torch.from_numpy(self.high_level_logprobs[:size]).to(
                    self.device
                ),
                "values": torch.from_numpy(self.high_level_values[:size]).to(self.device),
                "rewards": torch.from_numpy(self.high_level_rewards[:size]).to(
                    self.device
                ),
                "mask": torch.from_numpy(self.option_mask[:size]).to(self.device),
            }

        if level in ["low", "both"]:
            result["low_level"] = {
                "actions": {
                    key: torch.from_numpy(self.low_level_actions[key][:size]).to(
                        self.device
                    )
                    for key in self.low_level_actions.keys()
                },
                "logprobs": torch.from_numpy(self.low_level_logprobs[:size]).to(
                    self.device
                ),
                "values": torch.from_numpy(self.low_level_values[:size]).to(self.device),
                "rewards": torch.from_numpy(self.low_level_rewards[:size]).to(
                    self.device
                ),
            }

        return result

    def reset(self):
        """Reset buffer."""
        self.pos = 0
        self.full = False


class HierarchicalPPOTrainer:
    """
    Hierarchical PPO trainer with options framework.

    Trains high-level and low-level policies separately with PPO.
    """

    def __init__(
        self,
        policy,
        env,
        config: Optional[HierarchicalPPOConfig] = None,
        device: str = "cpu",
    ):
        """
        Initialize trainer.

        Args:
            policy: HierarchicalPolicy instance
            env: Vectorized environment
            config: Training configuration
            device: Device to use
        """
        self.policy = policy.to(device)
        self.env = env
        self.config = config or HierarchicalPPOConfig()
        self.device = device

        # Optimizers for each level
        self.high_level_optimizer = optim.Adam(
            list(policy.high_level_policy.parameters())
            + list(policy.high_level_value.parameters()),
            lr=self.config.high_level_lr,
        )

        self.low_level_optimizer = optim.Adam(
            list(policy.low_level_policy.parameters())
            + list(policy.low_level_value.parameters()),
            lr=self.config.low_level_lr,
        )

        # Get observation space shape
        obs_space_shape = {
            "aircraft": env.single_observation_space["aircraft"].shape,
            "aircraft_mask": env.single_observation_space["aircraft_mask"].shape,
            "global_state": env.single_observation_space["global_state"].shape,
            "conflict_matrix": env.single_observation_space["conflict_matrix"].shape,
        }

        # Rollout buffer
        self.buffer = HierarchicalRolloutBuffer(
            buffer_size=self.config.num_steps,
            num_envs=self.config.num_envs,
            obs_space_shape=obs_space_shape,
            device=device,
        )

        # Option tracking
        self.current_options = np.full(self.config.num_envs, -1, dtype=np.int64)
        self.option_steps_remaining = np.zeros(self.config.num_envs, dtype=np.int64)

        # Stats tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.global_step = 0
        self.update_step = 0

        # WandB
        if self.config.use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                config=self.config.__dict__,
            )

        logger.info(
            f"HierarchicalPPOTrainer initialized: "
            f"{self.config.num_envs} envs, "
            f"option length {self.config.option_length}"
        )

    def compute_intrinsic_reward(
        self, new_option: np.ndarray, old_option: np.ndarray
    ) -> np.ndarray:
        """
        Compute intrinsic reward for high-level policy.

        Args:
            new_option: Newly selected options
            old_option: Previous options

        Returns:
            Intrinsic rewards
        """
        if self.config.intrinsic_reward_type == "option_change":
            # Reward for changing options (encourages exploration)
            return (new_option != old_option).astype(np.float32)
        elif self.config.intrinsic_reward_type == "diversity":
            # Reward for selecting diverse options
            # Count unique options in batch
            unique_count = len(np.unique(new_option))
            return np.full_like(new_option, unique_count / len(new_option), dtype=np.float32)
        else:
            return np.zeros_like(new_option, dtype=np.float32)

    def collect_rollouts(self) -> Dict[str, float]:
        """
        Collect rollouts from environment.

        Returns:
            Dictionary with rollout statistics
        """
        self.buffer.reset()
        self.policy.eval()

        episode_rewards = []
        episode_lengths = []

        obs = self.env.reset()

        for step in range(self.config.num_steps):
            # Determine if this is an option step (high-level acts)
            is_option_step = self.option_steps_remaining <= 0

            with torch.no_grad():
                # Convert obs to tensors
                obs_tensor = {
                    key: torch.from_numpy(val).to(self.device) for key, val in obs.items()
                }

                # Get hierarchical action
                result = self.policy.get_hierarchical_action_and_value(
                    obs_tensor, level="both"
                )

                high_level_action = result["high_level"]["action"]["aircraft_id"].cpu().numpy()
                high_level_logprob = result["high_level"]["log_prob"].cpu().numpy()
                high_level_value = result["high_level"]["value"].cpu().numpy()
                attention_weights = result["high_level"]["attention_weights"].cpu().numpy()

                low_level_action = {
                    key: val.cpu().numpy()
                    for key, val in result["low_level"]["action"].items()
                }
                low_level_logprob = result["low_level"]["log_prob"].cpu().numpy()
                low_level_value = result["low_level"]["value"].cpu().numpy()

                # Update options
                if is_option_step.any():
                    old_options = self.current_options.copy()
                    self.current_options[is_option_step] = high_level_action[is_option_step]
                    self.option_steps_remaining[is_option_step] = self.config.option_length

                    # Add intrinsic reward
                    if self.config.use_intrinsic_reward:
                        intrinsic_reward = self.compute_intrinsic_reward(
                            self.current_options, old_options
                        )
                        intrinsic_reward *= self.config.intrinsic_reward_scale
                    else:
                        intrinsic_reward = np.zeros(self.config.num_envs)
                else:
                    intrinsic_reward = np.zeros(self.config.num_envs)

            # Execute action in environment
            action = {
                "aircraft_id": high_level_action,
                **low_level_action,
            }

            next_obs, rewards, dones, infos = self.env.step(action)

            # Add intrinsic reward to high-level
            high_level_rewards = rewards + intrinsic_reward

            # Store transition
            self.buffer.add(
                obs=obs,
                high_level_action=high_level_action,
                high_level_logprob=high_level_logprob,
                high_level_value=high_level_value,
                low_level_actions=low_level_action,
                low_level_logprob=low_level_logprob,
                low_level_value=low_level_value,
                reward=rewards,
                done=dones,
                is_option_step=is_option_step,
                attention_weights=attention_weights,
            )

            # Update for next step
            obs = next_obs
            self.option_steps_remaining -= 1
            self.global_step += self.config.num_envs

            # Reset options for done environments
            if dones.any():
                self.current_options[dones] = -1
                self.option_steps_remaining[dones] = 0

            # Track episode stats
            for info in infos:
                if "episode" in info:
                    episode_rewards.append(info["episode"]["r"])
                    episode_lengths.append(info["episode"]["l"])

        stats = {
            "mean_reward": np.mean(episode_rewards) if episode_rewards else 0.0,
            "mean_length": np.mean(episode_lengths) if episode_lengths else 0.0,
            "num_episodes": len(episode_rewards),
        }

        return stats

    def compute_gae(
        self, rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation.

        Args:
            rewards: Rewards tensor
            values: Value estimates
            dones: Done flags

        Returns:
            Tuple of (advantages, returns)
        """
        advantages = torch.zeros_like(rewards)
        last_gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.config.gamma * next_value * (1 - dones[t]) - values[t]
            last_gae = delta + self.config.gamma * self.config.gae_lambda * (1 - dones[t]) * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        return advantages, returns

    def update_policy(self) -> Dict[str, float]:
        """
        Update both high-level and low-level policies.

        Returns:
            Dictionary with update statistics
        """
        self.policy.train()

        data = self.buffer.get("both")

        # Flatten batch dimensions
        b_obs = {key: val.flatten(0, 1) for key, val in data["obs"].items()}
        b_dones = data["dones"].flatten(0, 1)

        # High-level data
        b_high_actions = data["high_level"]["actions"].flatten(0, 1)
        b_high_logprobs = data["high_level"]["logprobs"].flatten(0, 1)
        b_high_values = data["high_level"]["values"].flatten(0, 1)
        b_high_rewards = data["high_level"]["rewards"].flatten(0, 1)
        b_high_mask = data["high_level"]["mask"].flatten(0, 1)

        # Low-level data
        b_low_actions = {key: val.flatten(0, 1) for key, val in data["low_level"]["actions"].items()}
        b_low_logprobs = data["low_level"]["logprobs"].flatten(0, 1)
        b_low_values = data["low_level"]["values"].flatten(0, 1)
        b_low_rewards = data["low_level"]["rewards"].flatten(0, 1)

        # Compute advantages and returns
        high_advantages, high_returns = self.compute_gae(b_high_rewards, b_high_values, b_dones)
        low_advantages, low_returns = self.compute_gae(b_low_rewards, b_low_values, b_dones)

        # Normalize advantages
        high_advantages = (high_advantages - high_advantages.mean()) / (
            high_advantages.std() + 1e-8
        )
        low_advantages = (low_advantages - low_advantages.mean()) / (
            low_advantages.std() + 1e-8
        )

        # Training stats
        stats = {
            "high_level_loss": 0.0,
            "low_level_loss": 0.0,
            "high_level_value_loss": 0.0,
            "low_level_value_loss": 0.0,
            "high_level_policy_loss": 0.0,
            "low_level_policy_loss": 0.0,
        }

        # Update for multiple epochs
        batch_size = len(b_high_actions)
        minibatch_size = batch_size // self.config.num_minibatches
        indices = np.arange(batch_size)

        for epoch in range(self.config.num_epochs):
            np.random.shuffle(indices)

            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_indices = indices[start:end]

                # Get minibatch
                mb_obs = {key: val[mb_indices] for key, val in b_obs.items()}

                # Update high-level policy (only on option steps)
                mb_high_mask = b_high_mask[mb_indices]
                if mb_high_mask.any():
                    result = self.policy.get_hierarchical_action_and_value(
                        mb_obs, level="high", action={"aircraft_id": b_high_actions[mb_indices]}
                    )

                    new_high_logprob = result["high_level"]["log_prob"]
                    new_high_value = result["high_level"]["value"]
                    new_high_entropy = result["high_level"]["entropy"]

                    old_high_logprob = b_high_logprobs[mb_indices]
                    high_adv = high_advantages[mb_indices]
                    high_ret = high_returns[mb_indices]

                    # Apply mask
                    new_high_logprob = new_high_logprob[mb_high_mask]
                    old_high_logprob = old_high_logprob[mb_high_mask]
                    new_high_value = new_high_value[mb_high_mask]
                    high_adv = high_adv[mb_high_mask]
                    high_ret = high_ret[mb_high_mask]
                    new_high_entropy = new_high_entropy[mb_high_mask]

                    # Policy loss
                    ratio = torch.exp(new_high_logprob - old_high_logprob)
                    pg_loss1 = -high_adv * ratio
                    pg_loss2 = -high_adv * torch.clamp(
                        ratio, 1 - self.config.clip_coef, 1 + self.config.clip_coef
                    )
                    high_pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    if self.config.clip_vloss:
                        v_loss_unclipped = (new_high_value - high_ret) ** 2
                        v_clipped = b_high_values[mb_indices][mb_high_mask] + torch.clamp(
                            new_high_value - b_high_values[mb_indices][mb_high_mask],
                            -self.config.clip_coef,
                            self.config.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - high_ret) ** 2
                        high_v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                    else:
                        high_v_loss = 0.5 * ((new_high_value - high_ret) ** 2).mean()

                    # Total loss
                    high_entropy_loss = new_high_entropy.mean()
                    high_loss = (
                        high_pg_loss
                        + self.config.vf_coef * high_v_loss
                        - self.config.ent_coef * high_entropy_loss
                    )

                    # Update
                    self.high_level_optimizer.zero_grad()
                    high_loss.backward()
                    nn.utils.clip_grad_norm_(
                        list(self.policy.high_level_policy.parameters())
                        + list(self.policy.high_level_value.parameters()),
                        self.config.max_grad_norm,
                    )
                    self.high_level_optimizer.step()

                    stats["high_level_loss"] += high_loss.item()
                    stats["high_level_value_loss"] += high_v_loss.item()
                    stats["high_level_policy_loss"] += high_pg_loss.item()

                # Update low-level policy
                full_action = {
                    "aircraft_id": b_high_actions[mb_indices],
                    **{key: val[mb_indices] for key, val in b_low_actions.items()},
                }

                result = self.policy.get_hierarchical_action_and_value(
                    mb_obs, level="low", action=full_action
                )

                new_low_logprob = result["low_level"]["log_prob"]
                new_low_value = result["low_level"]["value"]
                new_low_entropy = result["low_level"]["entropy"]

                old_low_logprob = b_low_logprobs[mb_indices]
                low_adv = low_advantages[mb_indices]
                low_ret = low_returns[mb_indices]

                # Policy loss
                ratio = torch.exp(new_low_logprob - old_low_logprob)
                pg_loss1 = -low_adv * ratio
                pg_loss2 = -low_adv * torch.clamp(
                    ratio, 1 - self.config.clip_coef, 1 + self.config.clip_coef
                )
                low_pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                if self.config.clip_vloss:
                    v_loss_unclipped = (new_low_value - low_ret) ** 2
                    v_clipped = b_low_values[mb_indices] + torch.clamp(
                        new_low_value - b_low_values[mb_indices],
                        -self.config.clip_coef,
                        self.config.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - low_ret) ** 2
                    low_v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    low_v_loss = 0.5 * ((new_low_value - low_ret) ** 2).mean()

                # Total loss
                low_entropy_loss = new_low_entropy.mean()
                low_loss = (
                    low_pg_loss
                    + self.config.vf_coef * low_v_loss
                    - self.config.ent_coef * low_entropy_loss
                )

                # Update
                self.low_level_optimizer.zero_grad()
                low_loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.policy.low_level_policy.parameters())
                    + list(self.policy.low_level_value.parameters()),
                    self.config.max_grad_norm,
                )
                self.low_level_optimizer.step()

                stats["low_level_loss"] += low_loss.item()
                stats["low_level_value_loss"] += low_v_loss.item()
                stats["low_level_policy_loss"] += low_pg_loss.item()

        # Average stats
        num_updates = self.config.num_epochs * self.config.num_minibatches
        for key in stats:
            stats[key] /= num_updates

        self.update_step += 1
        return stats

    def train(self):
        """Run training loop."""
        logger.info("Starting training...")
        start_time = time.time()

        num_updates = self.config.total_timesteps // (
            self.config.num_steps * self.config.num_envs
        )

        for update in range(1, num_updates + 1):
            # Anneal learning rate
            if self.config.lr_anneal:
                frac = 1.0 - (update - 1.0) / num_updates
                self.high_level_optimizer.param_groups[0]["lr"] = (
                    self.config.high_level_lr * frac
                )
                self.low_level_optimizer.param_groups[0]["lr"] = (
                    self.config.low_level_lr * frac
                )

            # Collect rollouts
            rollout_stats = self.collect_rollouts()

            # Update policy
            update_stats = self.update_policy()

            # Log
            if update % self.config.log_interval == 0:
                elapsed_time = time.time() - start_time
                fps = self.global_step / elapsed_time

                logger.info(
                    f"Update {update}/{num_updates} | "
                    f"Steps {self.global_step}/{self.config.total_timesteps} | "
                    f"FPS {fps:.0f} | "
                    f"Reward {rollout_stats['mean_reward']:.2f} | "
                    f"Length {rollout_stats['mean_length']:.1f}"
                )

                if self.config.use_wandb and WANDB_AVAILABLE:
                    wandb.log(
                        {
                            "global_step": self.global_step,
                            "fps": fps,
                            **rollout_stats,
                            **update_stats,
                            "high_level_lr": self.high_level_optimizer.param_groups[0]["lr"],
                            "low_level_lr": self.low_level_optimizer.param_groups[0]["lr"],
                        }
                    )

            # Save checkpoint
            if update % self.config.save_interval == 0:
                self.save_checkpoint(update)

        logger.info("Training completed!")

    def save_checkpoint(self, update: int):
        """Save checkpoint."""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = checkpoint_dir / f"checkpoint_{update}.pt"

        torch.save(
            {
                "update": update,
                "global_step": self.global_step,
                "policy_state_dict": self.policy.state_dict(),
                "high_level_optimizer_state_dict": self.high_level_optimizer.state_dict(),
                "low_level_optimizer_state_dict": self.low_level_optimizer.state_dict(),
                "config": self.config,
            },
            checkpoint_path,
        )

        logger.info(f"Checkpoint saved: {checkpoint_path}")
