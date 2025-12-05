"""
MPC Planner for TD-MPC 2.

This module implements Model Predictive Control (MPC) planning using
Cross-Entropy Method (CEM) for action optimization.

Key features:
- CEM-based action sampling
- Model rollouts for trajectory evaluation
- Q-function for terminal value estimation
- Configurable planning horizon
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from models.tdmpc2 import TDMPC2Model, TDMPC2Config
from environment.utils import get_device

logger = logging.getLogger(__name__)


@dataclass
class MPCPlannerConfig:
    """Configuration for MPC planner."""
    
    # Planning
    planning_horizon: int = 5
    num_samples: int = 512
    num_elites: int = 64
    num_iterations: int = 6
    
    # Action sampling
    action_std_init: float = 0.3
    action_std_min: float = 0.05
    
    # CEM parameters
    momentum: float = 0.1
    
    # Discount factor
    gamma: float = 0.99
    
    # Device
    device: str = field(default_factory=lambda: get_device())
    
    def __post_init__(self):
        """Validate configuration."""
        if self.planning_horizon <= 0:
            raise ValueError("planning_horizon must be positive")
        if self.num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if self.num_elites <= 0 or self.num_elites > self.num_samples:
            raise ValueError(
                f"num_elites ({self.num_elites}) must be positive and <= num_samples ({self.num_samples})"
            )
        if self.num_iterations <= 0:
            raise ValueError("num_iterations must be positive")
        if not (0.0 <= self.momentum <= 1.0):
            raise ValueError(f"momentum ({self.momentum}) must be in [0.0, 1.0]")
        if not (0.0 < self.gamma <= 1.0):
            raise ValueError(f"gamma ({self.gamma}) must be in (0.0, 1.0]")
        if self.action_std_init <= 0:
            raise ValueError("action_std_init must be positive")
        if self.action_std_min <= 0:
            raise ValueError("action_std_min must be positive")
        if self.action_std_min > self.action_std_init:
            raise ValueError(
                f"action_std_min ({self.action_std_min}) must be <= action_std_init ({self.action_std_init})"
            )
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for logging."""
        return {
            "planning_horizon": self.planning_horizon,
            "num_samples": self.num_samples,
            "num_elites": self.num_elites,
            "num_iterations": self.num_iterations,
            "gamma": self.gamma,
        }


class MPCPlanner:
    """
    Model Predictive Control planner using Cross-Entropy Method.
    
    Plans actions by:
    1. Sampling candidate action sequences
    2. Rolling out trajectories using learned model
    3. Evaluating with rewards + Q-function terminal value
    4. Selecting best actions via CEM
    
    Example:
        >>> model_config = TDMPC2Config()
        >>> model = TDMPC2Model(model_config)
        >>> planner_config = MPCPlannerConfig(planning_horizon=5)
        >>> planner = MPCPlanner(model, planner_config)
        >>> 
        >>> # Plan action
        >>> aircraft = torch.randn(1, 20, 14)
        >>> mask = torch.ones(1, 20, dtype=torch.bool)
        >>> global_state = torch.randn(1, 4)
        >>> action = planner.plan(aircraft, mask, global_state)
    """
    
    def __init__(
        self,
        model: TDMPC2Model,
        config: MPCPlannerConfig,
    ):
        """
        Initialize MPC planner.
        
        Args:
            model: TD-MPC 2 model
            config: Planner configuration
            
        Raises:
            ValueError: If model and config are incompatible
        """
        if not isinstance(model, TDMPC2Model):
            raise TypeError(f"model must be TDMPC2Model, got {type(model)}")
        
        self.model = model
        self.config = config
        
        # Move model to device and set to eval mode
        self.model.to(config.device)
        self.model.eval()
        
        # Action space bounds (normalized)
        # For discrete actions, we'll sample in continuous space and discretize
        self.action_dim = model.config.action_dim
        self.action_low = -1.0
        self.action_high = 1.0
        
        # Statistics
        self.num_plans = 0
        self.avg_plan_time = 0.0
        
        logger.info(f"Initialized MPC planner: horizon={config.planning_horizon}, "
                   f"samples={config.num_samples}, elites={config.num_elites}")
    
    def plan(
        self,
        aircraft: torch.Tensor,
        aircraft_mask: torch.Tensor,
        global_state: torch.Tensor,
        return_all_actions: bool = False,
    ) -> torch.Tensor:
        """
        Plan action using MPC.
        
        Args:
            aircraft: Aircraft features (batch_size, max_aircraft, aircraft_feature_dim)
            aircraft_mask: Boolean mask (batch_size, max_aircraft)
            global_state: Global state (batch_size, global_feature_dim)
            return_all_actions: If True, return all planned actions (batch_size, horizon, action_dim)
            
        Returns:
            Best action (batch_size, action_dim) or all actions if return_all_actions
            
        Raises:
            ValueError: If input shapes are invalid
        """
        # Validate inputs
        if aircraft.dim() != 3:
            raise ValueError(
                f"Expected 3D aircraft tensor, got {aircraft.dim()}D with shape {aircraft.shape}"
            )
        if aircraft_mask.dim() != 2:
            raise ValueError(
                f"Expected 2D mask tensor, got {aircraft_mask.dim()}D with shape {aircraft_mask.shape}"
            )
        if global_state.dim() != 2:
            raise ValueError(
                f"Expected 2D global_state tensor, got {global_state.dim()}D with shape {global_state.shape}"
            )
        
        batch_size = aircraft.size(0)
        if batch_size != aircraft_mask.size(0) or batch_size != global_state.size(0):
            raise ValueError(
                f"Batch size mismatch: aircraft={aircraft.size(0)}, "
                f"mask={aircraft_mask.size(0)}, global_state={global_state.size(0)}"
            )
        
        self.num_plans += 1
        device = aircraft.device
        
        # Encode current observation
        with torch.no_grad():
            latent = self.model.encode(aircraft, aircraft_mask, global_state)
        
        # Initialize action distribution
        action_mean = torch.zeros(
            batch_size,
            self.config.planning_horizon,
            self.action_dim,
            device=device,
        )
        action_std = torch.ones(
            batch_size,
            self.config.planning_horizon,
            self.action_dim,
            device=device,
        ) * self.config.action_std_init
        
        # CEM iterations
        for iteration in range(self.config.num_iterations):
            # Sample actions
            actions = self._sample_actions(action_mean, action_std)
            
            # Evaluate trajectories
            returns = self._evaluate_trajectories(latent, actions)
            
            # Select elites
            elite_indices = self._select_elites(returns)
            
            # Update distribution
            action_mean, action_std = self._update_distribution(
                actions, elite_indices, action_mean, action_std
            )
        
        # Return first action of best sequence
        if return_all_actions:
            return action_mean
        else:
            return action_mean[:, 0, :]
    
    def _sample_actions(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sample actions from distribution.
        
        Args:
            mean: Action mean (batch_size, horizon, action_dim)
            std: Action std (batch_size, horizon, action_dim)
            
        Returns:
            Sampled actions (num_samples, batch_size, horizon, action_dim)
        """
        # Sample from normal distribution
        noise = torch.randn(
            self.config.num_samples,
            mean.size(0),
            mean.size(1),
            mean.size(2),
            device=mean.device,
        )
        
        actions = mean.unsqueeze(0) + std.unsqueeze(0) * noise
        
        # Clip to bounds
        actions = torch.clamp(actions, self.action_low, self.action_high)
        
        return actions
    
    def _evaluate_trajectories(
        self,
        initial_latent: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Evaluate trajectories using model rollouts.
        
        Args:
            initial_latent: Initial latent state (batch_size, latent_dim)
            actions: Action sequences (num_samples, batch_size, horizon, action_dim)
            
        Returns:
            Returns for each trajectory (num_samples, batch_size)
        """
        num_samples = actions.size(0)
        batch_size = actions.size(1)
        horizon = actions.size(2)
        device = actions.device
        
        # Expand initial latent for all samples and flatten for batch processing
        # initial_latent: (batch_size, latent_dim)
        # We need: (num_samples * batch_size, latent_dim)
        latent = initial_latent.unsqueeze(0).expand(num_samples, -1, -1)  # (num_samples, batch_size, latent_dim)
        latent = latent.reshape(num_samples * batch_size, -1)  # (num_samples * batch_size, latent_dim)
        
        total_returns = torch.zeros(num_samples, batch_size, device=device)
        discount = 1.0
        
        with torch.no_grad():
            for t in range(horizon):
                # Get actions for this timestep and flatten
                action = actions[:, :, t, :]  # (num_samples, batch_size, action_dim)
                action = action.reshape(num_samples * batch_size, -1)  # (num_samples * batch_size, action_dim)
                
                # Predict next latent
                next_latent = self.model.dynamics(latent, action)  # (num_samples * batch_size, latent_dim)
                
                # Get reward from model
                reward = self.model.reward(next_latent)  # (num_samples * batch_size, 1)
                reward = reward.reshape(num_samples, batch_size)  # (num_samples, batch_size)
                
                # Accumulate discounted return
                total_returns += discount * reward
                discount *= self.config.gamma
                
                # Update latent for next step
                latent = next_latent
            
            # Add terminal value from Q-function
            # Use last action and last latent
            last_action = actions[:, :, -1, :]  # (num_samples, batch_size, action_dim)
            last_action = last_action.reshape(num_samples * batch_size, -1)  # (num_samples * batch_size, action_dim)
            q_value = self.model.q_network(latent, last_action)  # (num_samples * batch_size, 1)
            q_value = q_value.reshape(num_samples, batch_size)  # (num_samples, batch_size)
            
            total_returns += discount * q_value
        
        return total_returns
    
    def _select_elites(
        self,
        returns: torch.Tensor,
    ) -> torch.Tensor:
        """
        Select elite trajectories based on returns.
        
        Args:
            returns: Returns for each trajectory (num_samples, batch_size)
            
        Returns:
            Elite indices (num_elites, batch_size)
        """
        # Get top-k returns for each batch
        _, indices = torch.topk(
            returns,
            self.config.num_elites,
            dim=0,
        )  # (num_elites, batch_size)
        
        return indices
    
    def _update_distribution(
        self,
        actions: torch.Tensor,
        elite_indices: torch.Tensor,
        old_mean: torch.Tensor,
        old_std: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update action distribution from elites.
        
        Args:
            actions: All sampled actions (num_samples, batch_size, horizon, action_dim)
            elite_indices: Elite indices (num_elites, batch_size)
            old_mean: Previous mean (batch_size, horizon, action_dim)
            old_std: Previous std (batch_size, horizon, action_dim)
            
        Returns:
            Updated (mean, std)
        """
        batch_size = actions.size(1)
        horizon = actions.size(2)
        action_dim = actions.size(3)
        
        # Select elite actions
        # elite_indices: (num_elites, batch_size)
        # actions: (num_samples, batch_size, horizon, action_dim)
        # We need to gather: (num_elites, batch_size, horizon, action_dim)
        
        elite_actions = torch.zeros(
            self.config.num_elites,
            batch_size,
            horizon,
            action_dim,
            device=actions.device,
        )
        
        for b in range(batch_size):
            elite_actions[:, b, :, :] = actions[elite_indices[:, b], b, :, :]
        
        # Compute new mean and std
        new_mean = elite_actions.mean(dim=0)  # (batch_size, horizon, action_dim)
        new_std = elite_actions.std(dim=0)  # (batch_size, horizon, action_dim)
        
        # Clip std to prevent collapse
        new_std = torch.clamp(new_std, min=self.config.action_std_min)
        
        # Momentum update
        mean = (1.0 - self.config.momentum) * old_mean + self.config.momentum * new_mean
        std = (1.0 - self.config.momentum) * old_std + self.config.momentum * new_std
        
        return mean, std
    
    def get_statistics(self) -> Dict[str, float]:
        """
        Get planner statistics.
        
        Returns:
            Dictionary of statistics
        """
        return {
            "num_plans": self.num_plans,
            "avg_plan_time": self.avg_plan_time,
        }

