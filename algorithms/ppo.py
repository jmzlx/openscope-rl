"""
Proximal Policy Optimization (PPO) implementation for OpenScope
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import time


@dataclass
class RolloutBuffer:
    """Buffer for storing rollout data"""
    
    def __init__(self, buffer_size: int, device: torch.device):
        self.buffer_size = buffer_size
        self.device = device
        self.reset()
    
    def reset(self):
        """Clear the buffer"""
        self.observations = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.log_probs = []
        self.pos = 0
        self.full = False
    
    def add(
        self,
        obs: Dict[str, np.ndarray],
        action: Dict[str, np.ndarray],
        reward: float,
        done: bool,
        value: float,
        log_prob: float
    ):
        """Add a transition to the buffer"""
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.log_probs.append(log_prob)
        
        self.pos += 1
        if self.pos >= self.buffer_size:
            self.full = True
    
    def get(self) -> Tuple:
        """Get all data from buffer"""
        return (
            self.observations,
            self.actions,
            self.rewards,
            self.dones,
            self.values,
            self.log_probs
        )
    
    def size(self) -> int:
        """Current buffer size"""
        return len(self.observations)


class PPO:
    """
    Proximal Policy Optimization algorithm
    """
    
    def __init__(
        self,
        policy: nn.Module,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        n_steps: int = 2048,
        n_epochs: int = 10,
        batch_size: int = 64,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.policy = policy.to(device)
        self.device = torch.device(device)
        
        # Hyperparameters
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        # Rollout buffer
        self.rollout_buffer = RolloutBuffer(n_steps, device)
        
        # Statistics
        self.training_stats = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "total_loss": [],
            "approx_kl": [],
            "clip_fraction": []
        }
    
    def collect_rollouts(
        self,
        env,
        n_steps: int
    ) -> Dict:
        """
        Collect rollouts from the environment
        
        Args:
            env: Environment to collect from
            n_steps: Number of steps to collect
        
        Returns:
            stats: Dictionary of statistics
        """
        self.rollout_buffer.reset()
        self.policy.eval()
        
        obs, info = env.reset()
        episode_rewards = []
        episode_lengths = []
        current_episode_reward = 0
        current_episode_length = 0
        
        for step in range(n_steps):
            # Convert observation to tensors
            obs_tensor = self._obs_to_tensor(obs)
            
            # Get action from policy
            with torch.no_grad():
                action, log_prob, entropy, value = self.policy.get_action_and_value(obs_tensor)
            
            # Convert action to numpy for environment
            action_np = self._action_to_numpy(action)
            
            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated
            
            # Store transition
            self.rollout_buffer.add(
                obs=obs,
                action=action_np,
                reward=reward,
                done=done,
                value=value.cpu().item(),
                log_prob=log_prob.cpu().item()
            )
            
            current_episode_reward += reward
            current_episode_length += 1
            
            if done:
                episode_rewards.append(current_episode_reward)
                episode_lengths.append(current_episode_length)
                current_episode_reward = 0
                current_episode_length = 0
                obs, info = env.reset()
            else:
                obs = next_obs
        
        # Compute advantages and returns
        self._compute_returns_and_advantages()
        
        return {
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths,
            "mean_reward": np.mean(episode_rewards) if episode_rewards else 0,
            "mean_length": np.mean(episode_lengths) if episode_lengths else 0
        }
    
    def _compute_returns_and_advantages(self):
        """Compute GAE advantages and returns"""
        observations, actions, rewards, dones, values, log_probs = self.rollout_buffer.get()
        
        advantages = []
        returns = []
        
        last_gae_lam = 0
        last_value = 0
        
        for step in reversed(range(len(rewards))):
            if step == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[step]
                next_value = last_value
            else:
                next_non_terminal = 1.0 - dones[step]
                next_value = values[step + 1]
            
            delta = rewards[step] + self.gamma * next_value * next_non_terminal - values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            
            advantages.insert(0, last_gae_lam)
            returns.insert(0, last_gae_lam + values[step])
        
        # Store in buffer
        self.rollout_buffer.advantages = np.array(advantages, dtype=np.float32)
        self.rollout_buffer.returns = np.array(returns, dtype=np.float32)
    
    def train(self) -> Dict:
        """
        Train the policy using collected rollouts
        
        Returns:
            stats: Training statistics
        """
        self.policy.train()
        
        observations, actions, rewards, dones, values, log_probs = self.rollout_buffer.get()
        advantages = self.rollout_buffer.advantages
        returns = self.rollout_buffer.returns
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert to tensors
        old_log_probs = torch.tensor(log_probs, dtype=torch.float32, device=self.device)
        old_values = torch.tensor(values, dtype=torch.float32, device=self.device)
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=self.device)
        
        # Training epochs
        for epoch in range(self.n_epochs):
            # Generate random indices for minibatches
            indices = np.arange(len(observations))
            np.random.shuffle(indices)
            
            # Minibatch training
            for start_idx in range(0, len(observations), self.batch_size):
                end_idx = start_idx + self.batch_size
                mb_indices = indices[start_idx:end_idx]
                
                # Get minibatch data
                mb_obs = self._batch_obs([observations[i] for i in mb_indices])
                mb_actions = self._batch_actions([actions[i] for i in mb_indices])
                mb_advantages = advantages_tensor[mb_indices]
                mb_returns = returns_tensor[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_old_values = old_values[mb_indices]
                
                # Forward pass
                _, new_log_probs, entropy, new_values = self.policy.get_action_and_value(
                    mb_obs, mb_actions
                )
                
                # Policy loss (PPO clip objective)
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss (clipped)
                value_pred_clipped = mb_old_values + torch.clamp(
                    new_values - mb_old_values,
                    -self.clip_epsilon,
                    self.clip_epsilon
                )
                value_loss_unclipped = (new_values - mb_returns).pow(2)
                value_loss_clipped = (value_pred_clipped - mb_returns).pow(2)
                value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
                
                # Entropy bonus
                entropy_loss = entropy.mean()
                
                # Total loss
                loss = (
                    policy_loss +
                    self.value_coef * value_loss -
                    self.entropy_coef * entropy_loss
                )
                
                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Statistics
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - (new_log_probs - mb_old_log_probs)).mean()
                    clip_fraction = ((ratio - 1.0).abs() > self.clip_epsilon).float().mean()
                
                self.training_stats["policy_loss"].append(policy_loss.item())
                self.training_stats["value_loss"].append(value_loss.item())
                self.training_stats["entropy"].append(entropy_loss.item())
                self.training_stats["total_loss"].append(loss.item())
                self.training_stats["approx_kl"].append(approx_kl.item())
                self.training_stats["clip_fraction"].append(clip_fraction.item())
        
        # Return average statistics
        return {
            "policy_loss": np.mean(self.training_stats["policy_loss"][-10:]),
            "value_loss": np.mean(self.training_stats["value_loss"][-10:]),
            "entropy": np.mean(self.training_stats["entropy"][-10:]),
            "total_loss": np.mean(self.training_stats["total_loss"][-10:]),
            "approx_kl": np.mean(self.training_stats["approx_kl"][-10:]),
            "clip_fraction": np.mean(self.training_stats["clip_fraction"][-10:])
        }
    
    def _obs_to_tensor(self, obs: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """Convert observation dict to tensor dict"""
        return {
            key: torch.from_numpy(value).unsqueeze(0).to(self.device)
            for key, value in obs.items()
        }
    
    def _action_to_numpy(self, action: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
        """Convert action tensor dict to numpy dict"""
        return {
            key: value.cpu().numpy().item() if value.numel() == 1 else value.cpu().numpy()
            for key, value in action.items()
        }
    
    def _batch_obs(self, obs_list: List[Dict]) -> Dict[str, torch.Tensor]:
        """Batch a list of observations"""
        batched = {}
        keys = obs_list[0].keys()
        
        for key in keys:
            batched[key] = torch.from_numpy(
                np.stack([obs[key] for obs in obs_list])
            ).to(self.device)
        
        return batched
    
    def _batch_actions(self, action_list: List[Dict]) -> Dict[str, torch.Tensor]:
        """Batch a list of actions"""
        batched = {}
        keys = action_list[0].keys()
        
        for key in keys:
            batched[key] = torch.from_numpy(
                np.array([action[key] for action in action_list])
            ).to(self.device)
        
        return batched
    
    def save(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_stats': self.training_stats
        }, path)
    
    def load(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_stats = checkpoint['training_stats']

