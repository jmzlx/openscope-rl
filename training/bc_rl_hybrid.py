"""
Hybrid BC+RL training module.

This module implements a hybrid training approach that combines behavioral cloning
with reinforcement learning. The training process:
1. Pre-trains the policy using BC loss from demonstrations
2. Fine-tunes using PPO with a mixed loss: BC loss + RL loss
3. Progressively reduces BC weight over time

This approach significantly improves sample efficiency compared to pure RL.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Callable
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from collections import deque

from poc.atc_rl import Realistic3DATCEnv
from models.networks import ATCActorCritic
from models.config import create_default_network_config, NetworkConfig
from training.rule_based_expert import Demonstration, load_demonstrations
from training.behavioral_cloning import BehavioralCloningTrainer, DemonstrationDataset


logger = logging.getLogger(__name__)


class BCRLBuffer:
    """
    Experience replay buffer for BC+RL hybrid training.

    Stores both RL transitions and expert demonstrations for mixed training.
    """

    def __init__(self, capacity: int = 10000, device: str = "cpu"):
        """
        Initialize buffer.

        Args:
            capacity: Maximum buffer size
            device: Device to place tensors on
        """
        self.capacity = capacity
        self.device = device

        self.observations = deque(maxlen=capacity)
        self.actions = deque(maxlen=capacity)
        self.rewards = deque(maxlen=capacity)
        self.next_observations = deque(maxlen=capacity)
        self.dones = deque(maxlen=capacity)
        self.log_probs = deque(maxlen=capacity)
        self.values = deque(maxlen=capacity)

    def add(
        self,
        obs: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: float,
        next_obs: Dict[str, np.ndarray],
        done: bool,
        log_prob: float,
        value: float,
    ) -> None:
        """Add a transition to the buffer."""
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_observations.append(next_obs)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def get_batch(self, batch_size: int) -> Tuple[Dict, ...]:
        """Sample a batch from the buffer."""
        indices = np.random.choice(len(self.observations), batch_size, replace=False)

        obs_batch = {}
        for key in ['aircraft', 'aircraft_mask', 'global_state', 'conflict_matrix']:
            obs_batch[key] = torch.stack([
                torch.from_numpy(self.observations[i][key]).float()
                for i in indices
            ]).to(self.device)

        actions_batch = torch.stack([
            torch.from_numpy(self.actions[i]).long()
            for i in indices
        ]).to(self.device)

        rewards_batch = torch.tensor([self.rewards[i] for i in indices]).float().to(self.device)
        dones_batch = torch.tensor([self.dones[i] for i in indices]).float().to(self.device)
        log_probs_batch = torch.tensor([self.log_probs[i] for i in indices]).float().to(self.device)
        values_batch = torch.tensor([self.values[i] for i in indices]).float().to(self.device)

        return obs_batch, actions_batch, rewards_batch, dones_batch, log_probs_batch, values_batch

    def __len__(self) -> int:
        return len(self.observations)


class BCRLHybridTrainer:
    """
    Hybrid trainer combining behavioral cloning and reinforcement learning.

    This trainer uses a mixed objective:
    - BC loss: Cross-entropy loss on expert demonstrations
    - RL loss: PPO loss on environment interactions
    - Weight scheduling: Progressively reduce BC weight over training

    Example:
        >>> # Load demonstrations and pre-trained model
        >>> demos = load_demonstrations("data/expert_demonstrations.pkl")
        >>>
        >>> # Create environment
        >>> env = Realistic3DATCEnv(max_aircraft=5)
        >>>
        >>> # Create trainer
        >>> trainer = BCRLHybridTrainer(
        ...     env=env,
        ...     demonstrations=demos,
        ...     pretrained_model_path="checkpoints/bc_pretrained.pth"
        ... )
        >>>
        >>> # Train with mixed BC+RL
        >>> trainer.train(num_iterations=1000, bc_weight_start=1.0, bc_weight_end=0.1)
    """

    def __init__(
        self,
        env: Realistic3DATCEnv,
        demonstrations: Optional[List[Demonstration]] = None,
        network_config: Optional[NetworkConfig] = None,
        pretrained_model_path: Optional[str] = None,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize hybrid BC+RL trainer.

        Args:
            env: RL environment
            demonstrations: Expert demonstrations (optional if using pretrained model)
            network_config: Network configuration
            pretrained_model_path: Path to pretrained BC model
            learning_rate: Learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_epsilon: PPO clip epsilon
            value_loss_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            device: Device to train on
        """
        self.env = env
        self.device = device
        self.demonstrations = demonstrations

        # Hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        # Create network
        if network_config is None:
            network_config = create_default_network_config(max_aircraft=env.max_aircraft)

        self.model = ATCActorCritic(network_config).to(device)

        # Load pretrained model if provided
        if pretrained_model_path is not None:
            self._load_pretrained(pretrained_model_path)

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Experience buffer
        self.buffer = BCRLBuffer(capacity=10000, device=device)

        # BC dataset for expert demonstrations
        if demonstrations is not None:
            self.bc_dataset = DemonstrationDataset(demonstrations, device=device)
        else:
            self.bc_dataset = None

        # Loss function for BC
        self.bc_loss_fn = nn.CrossEntropyLoss()

        # Training statistics
        self.episode_rewards = []
        self.bc_losses = []
        self.rl_losses = []
        self.total_losses = []

        logger.info(f"Initialized BC+RL hybrid trainer on {device}")

    def _load_pretrained(self, model_path: str) -> None:
        """Load pretrained BC model."""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded pretrained model from {model_path}")

    def train(
        self,
        num_iterations: int = 1000,
        episodes_per_iteration: int = 4,
        ppo_epochs: int = 4,
        bc_batch_size: int = 64,
        bc_weight_start: float = 1.0,
        bc_weight_end: float = 0.0,
        bc_decay_schedule: str = "linear",
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Train using hybrid BC+RL approach.

        Args:
            num_iterations: Number of training iterations
            episodes_per_iteration: Episodes to collect per iteration
            ppo_epochs: PPO update epochs per iteration
            bc_batch_size: Batch size for BC loss
            bc_weight_start: Initial BC loss weight
            bc_weight_end: Final BC loss weight
            bc_decay_schedule: Weight decay schedule ("linear" or "exponential")
            verbose: Whether to print progress

        Returns:
            Dictionary with training statistics
        """
        for iteration in range(num_iterations):
            # Calculate BC weight based on schedule
            bc_weight = self._get_bc_weight(
                iteration, num_iterations, bc_weight_start, bc_weight_end, bc_decay_schedule
            )

            # Collect episodes
            episode_rewards = []
            for _ in range(episodes_per_iteration):
                episode_reward = self._collect_episode()
                episode_rewards.append(episode_reward)

            avg_episode_reward = np.mean(episode_rewards)
            self.episode_rewards.append(avg_episode_reward)

            # Update policy with PPO + BC
            if len(self.buffer) >= bc_batch_size:
                for _ in range(ppo_epochs):
                    bc_loss, rl_loss = self._update_policy(bc_batch_size, bc_weight)
                    self.bc_losses.append(bc_loss)
                    self.rl_losses.append(rl_loss)
                    self.total_losses.append(bc_loss * bc_weight + rl_loss)

            # Logging
            if verbose and (iteration + 1) % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                avg_bc_loss = np.mean(self.bc_losses[-10:]) if self.bc_losses else 0
                avg_rl_loss = np.mean(self.rl_losses[-10:]) if self.rl_losses else 0

                logger.info(
                    f"Iteration {iteration + 1}/{num_iterations} - "
                    f"Reward: {avg_reward:.2f}, BC Weight: {bc_weight:.3f}, "
                    f"BC Loss: {avg_bc_loss:.4f}, RL Loss: {avg_rl_loss:.4f}"
                )

        return {
            'episode_rewards': self.episode_rewards,
            'bc_losses': self.bc_losses,
            'rl_losses': self.rl_losses,
            'total_losses': self.total_losses,
        }

    def _get_bc_weight(
        self,
        iteration: int,
        total_iterations: int,
        start_weight: float,
        end_weight: float,
        schedule: str,
    ) -> float:
        """Calculate BC weight based on decay schedule."""
        progress = iteration / total_iterations

        if schedule == "linear":
            return start_weight + (end_weight - start_weight) * progress
        elif schedule == "exponential":
            return start_weight * (end_weight / start_weight) ** progress
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

    def _collect_episode(self) -> float:
        """Collect one episode of experience."""
        obs, info = self.env.reset()
        episode_reward = 0.0
        done = False

        while not done:
            # Convert observation to tensor
            obs_tensor = {
                'aircraft': torch.from_numpy(obs['aircraft']).float().unsqueeze(0).to(self.device),
                'aircraft_mask': torch.from_numpy(obs['aircraft_mask']).bool().unsqueeze(0).to(self.device),
                'global_state': torch.from_numpy(obs['global_state']).float().unsqueeze(0).to(self.device),
                'conflict_matrix': torch.from_numpy(obs['conflict_matrix']).float().unsqueeze(0).to(self.device),
            }

            # Get action from policy
            with torch.no_grad():
                action_logits, value = self.model(obs_tensor)

                # Sample action
                action = self._sample_action(action_logits)
                log_prob = self._compute_log_prob(action_logits, action)

            # Execute action
            action_np = action.cpu().numpy()
            next_obs, reward, terminated, truncated, info = self.env.step(action_np)
            done = terminated or truncated

            # Store transition
            self.buffer.add(
                obs=obs,
                action=action_np,
                reward=reward,
                next_obs=next_obs,
                done=done,
                log_prob=log_prob.item(),
                value=value.item(),
            )

            obs = next_obs
            episode_reward += reward

        return episode_reward

    def _sample_action(self, action_logits: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Sample action from policy logits."""
        action_components = []
        action_names = ['aircraft_id', 'command_type', 'altitude', 'heading', 'speed']

        for name in action_names:
            if name in action_logits:
                dist = torch.distributions.Categorical(logits=action_logits[name])
                action_components.append(dist.sample())

        return torch.stack(action_components, dim=1).squeeze(0)

    def _compute_log_prob(
        self,
        action_logits: Dict[str, torch.Tensor],
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Compute log probability of action."""
        log_prob = 0.0
        action_names = ['aircraft_id', 'command_type', 'altitude', 'heading', 'speed']

        for i, name in enumerate(action_names):
            if name in action_logits:
                dist = torch.distributions.Categorical(logits=action_logits[name])
                log_prob += dist.log_prob(action[i])

        return log_prob

    def _update_policy(self, batch_size: int, bc_weight: float) -> Tuple[float, float]:
        """Update policy with mixed BC+RL loss."""
        # Sample batch from RL buffer
        obs_batch, actions_batch, rewards_batch, dones_batch, old_log_probs_batch, old_values_batch = \
            self.buffer.get_batch(min(batch_size, len(self.buffer)))

        # Compute RL loss (PPO)
        action_logits, values = self.model(obs_batch)

        # Compute log probs for actions
        log_probs = []
        for i in range(len(actions_batch)):
            log_prob = self._compute_log_prob(
                {k: v[i:i+1] for k, v in action_logits.items()},
                actions_batch[i]
            )
            log_probs.append(log_prob)

        log_probs = torch.stack(log_probs)

        # Compute advantages (simplified - using rewards)
        advantages = rewards_batch - old_values_batch

        # PPO clipped loss
        ratio = torch.exp(log_probs - old_log_probs_batch)
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

        # Value loss
        value_loss = 0.5 * (values.squeeze() - rewards_batch).pow(2).mean()

        # Entropy bonus
        entropy = -log_probs.mean()

        # Total RL loss
        rl_loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy

        # Compute BC loss (if demonstrations available)
        bc_loss = 0.0
        if self.bc_dataset is not None and bc_weight > 0:
            # Sample from expert demonstrations
            expert_indices = np.random.choice(len(self.bc_dataset), batch_size, replace=True)

            expert_obs_list = []
            expert_actions_list = []

            for idx in expert_indices:
                obs, action = self.bc_dataset[idx]
                expert_obs_list.append(obs)
                expert_actions_list.append(action)

            # Stack observations
            expert_obs_batch = {
                key: torch.stack([obs[key] for obs in expert_obs_list])
                for key in expert_obs_list[0].keys()
            }
            expert_actions_batch = torch.stack(expert_actions_list)

            # Forward pass
            expert_action_logits, _ = self.model(expert_obs_batch)

            # BC loss (cross-entropy)
            action_names = ['aircraft_id', 'command_type', 'altitude', 'heading', 'speed']
            for i, name in enumerate(action_names):
                if name in expert_action_logits:
                    bc_loss += self.bc_loss_fn(
                        expert_action_logits[name],
                        expert_actions_batch[:, i]
                    )

        # Combined loss
        total_loss = bc_weight * bc_loss + rl_loss

        # Optimization step
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()

        return bc_loss.item() if isinstance(bc_loss, torch.Tensor) else bc_loss, rl_loss.item()

    def save_model(self, save_path: str) -> None:
        """Save trained model."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode_rewards': self.episode_rewards,
            'bc_losses': self.bc_losses,
            'rl_losses': self.rl_losses,
            'total_losses': self.total_losses,
            'config': self.model.config,
        }

        torch.save(checkpoint, save_path)
        logger.info(f"Model saved to {save_path}")

    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """
        Evaluate trained policy.

        Args:
            num_episodes: Number of episodes to evaluate

        Returns:
            Evaluation metrics
        """
        self.model.eval()
        episode_rewards = []
        success_count = 0

        for _ in range(num_episodes):
            obs, info = self.env.reset()
            episode_reward = 0.0
            done = False

            while not done:
                obs_tensor = {
                    'aircraft': torch.from_numpy(obs['aircraft']).float().unsqueeze(0).to(self.device),
                    'aircraft_mask': torch.from_numpy(obs['aircraft_mask']).bool().unsqueeze(0).to(self.device),
                    'global_state': torch.from_numpy(obs['global_state']).float().unsqueeze(0).to(self.device),
                    'conflict_matrix': torch.from_numpy(obs['conflict_matrix']).float().unsqueeze(0).to(self.device),
                }

                with torch.no_grad():
                    action_logits, _ = self.model(obs_tensor)
                    action = self._sample_action(action_logits)

                action_np = action.cpu().numpy()
                obs, reward, terminated, truncated, info = self.env.step(action_np)
                done = terminated or truncated
                episode_reward += reward

            episode_rewards.append(episode_reward)
            if info.get('successful_landings', 0) > 0:
                success_count += 1

        self.model.train()

        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'success_rate': success_count / num_episodes,
        }


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create environment
    env = Realistic3DATCEnv(max_aircraft=5)

    # Load demonstrations
    demonstrations = load_demonstrations("data/expert_demonstrations.pkl")

    # Create trainer with pretrained BC model
    trainer = BCRLHybridTrainer(
        env=env,
        demonstrations=demonstrations,
        pretrained_model_path="checkpoints/bc_pretrained.pth",
    )

    # Train with hybrid BC+RL
    logger.info("Starting hybrid BC+RL training...")
    history = trainer.train(
        num_iterations=500,
        bc_weight_start=1.0,
        bc_weight_end=0.1,
        bc_decay_schedule="linear",
    )

    # Save trained model
    trainer.save_model("checkpoints/bc_rl_hybrid.pth")

    # Evaluate
    metrics = trainer.evaluate(num_episodes=20)
    logger.info(f"\nEvaluation Results:")
    logger.info(f"  Mean Reward: {metrics['mean_reward']:.2f} +/- {metrics['std_reward']:.2f}")
    logger.info(f"  Success Rate: {metrics['success_rate']:.2%}")
