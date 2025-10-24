"""
Reward calculation module for OpenScope RL environment.

This module provides the RewardCalculator class for computing rewards
based on game state changes and configurable reward shaping.
"""

import logging
from typing import Any, Dict, List, Optional
from abc import ABC, abstractmethod

from .config import RewardConfig
from .exceptions import RewardCalculationError


logger = logging.getLogger(__name__)


class RewardStrategy(ABC):
    """Abstract base class for reward calculation strategies."""
    
    @abstractmethod
    def calculate_reward(self, state: Dict[str, Any], prev_state: Dict[str, Any], 
                        action: Optional[Dict[str, int]] = None) -> float:
        """
        Calculate reward based on state change.
        
        Args:
            state: Current game state
            prev_state: Previous game state
            action: Action taken (optional)
            
        Returns:
            Calculated reward
        """
        pass


class DefaultRewardStrategy(RewardStrategy):
    """Default reward calculation strategy."""
    
    def __init__(self, config: RewardConfig):
        """
        Initialize default reward strategy.
        
        Args:
            config: Reward configuration
        """
        self.config = config
    
    def calculate_reward(self, state: Dict[str, Any], prev_state: Dict[str, Any], 
                        action: Optional[Dict[str, int]] = None) -> float:
        """
        Calculate reward using default strategy.
        
        Args:
            state: Current game state
            prev_state: Previous game state
            action: Action taken (optional)
            
        Returns:
            Calculated reward
        """
        reward = 0.0
        
        # Base score change
        score_change = state.get("score", 0) - prev_state.get("score", 0)
        reward += score_change
        
        # Time penalty
        reward += self.config.timestep_penalty
        
        # Action reward
        if action is not None:
            reward += self.config.action_reward
        
        # Conflict and violation penalties
        conflicts = state.get("conflicts", [])
        for conflict in conflicts:
            if conflict.get("hasViolation"):
                reward += self.config.separation_loss / 10.0
            elif conflict.get("hasConflict"):
                reward += self.config.conflict_warning
        
        # Safe separation bonus
        aircraft = state.get("aircraft", [])
        if len(conflicts) == 0 and len(aircraft) > 0:
            reward += self.config.safe_separation_bonus
        
        return reward


class SafetyFocusedRewardStrategy(RewardStrategy):
    """Reward strategy focused on safety (minimizing conflicts)."""
    
    def __init__(self, config: RewardConfig):
        """
        Initialize safety-focused reward strategy.
        
        Args:
            config: Reward configuration
        """
        self.config = config
    
    def calculate_reward(self, state: Dict[str, Any], prev_state: Dict[str, Any], 
                        action: Optional[Dict[str, int]] = None) -> float:
        """
        Calculate reward with focus on safety.
        
        Args:
            state: Current game state
            prev_state: Previous game state
            action: Action taken (optional)
            
        Returns:
            Calculated reward
        """
        reward = 0.0
        
        # Heavy penalty for violations
        conflicts = state.get("conflicts", [])
        violation_count = sum(1 for c in conflicts if c.get("hasViolation"))
        conflict_count = sum(1 for c in conflicts if c.get("hasConflict"))
        
        reward += violation_count * self.config.separation_loss
        reward += conflict_count * self.config.conflict_warning * 2
        
        # Bonus for no conflicts
        if len(conflicts) == 0:
            reward += self.config.safe_separation_bonus * 2
        
        # Small time penalty
        reward += self.config.timestep_penalty
        
        return reward


class EfficiencyFocusedRewardStrategy(RewardStrategy):
    """Reward strategy focused on efficiency (maximizing throughput)."""
    
    def __init__(self, config: RewardConfig):
        """
        Initialize efficiency-focused reward strategy.
        
        Args:
            config: Reward configuration
        """
        self.config = config
    
    def calculate_reward(self, state: Dict[str, Any], prev_state: Dict[str, Any], 
                        action: Optional[Dict[str, int]] = None) -> float:
        """
        Calculate reward with focus on efficiency.
        
        Args:
            state: Current game state
            prev_state: Previous game state
            action: Action taken (optional)
            
        Returns:
            Calculated reward
        """
        reward = 0.0
        
        # Reward for taking actions
        if action is not None:
            reward += self.config.action_reward * 2
        
        # Reward based on aircraft count (throughput)
        aircraft = state.get("aircraft", [])
        reward += len(aircraft) * 0.1
        
        # Moderate penalty for violations
        conflicts = state.get("conflicts", [])
        violation_count = sum(1 for c in conflicts if c.get("hasViolation"))
        reward += violation_count * self.config.separation_loss * 0.5
        
        # Small time penalty
        reward += self.config.timestep_penalty
        
        return reward


class RewardCalculator:
    """
    Calculates rewards based on game state changes.
    
    This class provides configurable reward calculation using different
    strategies and can be extended with custom reward functions.
    """
    
    def __init__(self, config: RewardConfig, strategy: Optional[RewardStrategy] = None):
        """
        Initialize reward calculator.
        
        Args:
            config: Reward configuration
            strategy: Reward calculation strategy (default: DefaultRewardStrategy)
        """
        self.config = config
        self.strategy = strategy or DefaultRewardStrategy(config)
        self._episode_rewards = []
    
    def calculate_reward(self, state: Dict[str, Any], prev_state: Dict[str, Any], 
                        action: Optional[Dict[str, int]] = None) -> float:
        """
        Calculate reward based on state change.
        
        Args:
            state: Current game state
            prev_state: Previous game state
            action: Action taken (optional)
            
        Returns:
            Calculated reward
            
        Raises:
            RewardCalculationError: If reward calculation fails
        """
        try:
            reward = self.strategy.calculate_reward(state, prev_state, action)
            
            # Store reward for episode tracking
            self._episode_rewards.append(reward)
            
            logger.debug(f"Calculated reward: {reward:.3f}")
            return reward
            
        except Exception as e:
            logger.error(f"Failed to calculate reward: {e}")
            raise RewardCalculationError(f"Reward calculation failed: {e}") from e
    
    def calculate_episode_bonus(self, episode_metrics: Dict[str, Any]) -> float:
        """
        Calculate episode completion bonus.
        
        Args:
            episode_metrics: Episode metrics dictionary
            
        Returns:
            Episode bonus reward
        """
        try:
            successful_exits = episode_metrics.get("successful_exits", 0)
            total_aircraft = episode_metrics.get("total_aircraft_spawned", 1)
            violations = episode_metrics.get("violations", 0)
            
            # Calculate success rate
            success_rate = successful_exits / max(total_aircraft, 1)
            
            bonus = 0.0
            
            # Success rate bonuses
            if success_rate >= self.config.high_success_threshold:
                bonus += self.config.high_success_rate_bonus
            elif success_rate >= self.config.medium_success_threshold:
                bonus += self.config.medium_success_rate_bonus
            elif success_rate >= self.config.low_success_threshold:
                bonus += self.config.low_success_rate_bonus
            
            # Penalty for violations
            if violations > 0:
                bonus += violations * self.config.separation_loss * 0.1
            
            logger.debug(f"Episode bonus: {bonus:.3f} (success_rate: {success_rate:.3f})")
            return bonus
            
        except Exception as e:
            logger.error(f"Failed to calculate episode bonus: {e}")
            raise RewardCalculationError(f"Episode bonus calculation failed: {e}") from e
    
    def set_strategy(self, strategy: RewardStrategy) -> None:
        """
        Set reward calculation strategy.
        
        Args:
            strategy: New reward strategy
        """
        self.strategy = strategy
        logger.info(f"Reward strategy changed to {strategy.__class__.__name__}")
    
    def get_episode_rewards(self) -> List[float]:
        """Get rewards for current episode."""
        return self._episode_rewards.copy()
    
    def reset_episode(self) -> None:
        """Reset episode reward tracking."""
        self._episode_rewards = []
        logger.debug("Episode rewards reset")
    
    def get_episode_total(self) -> float:
        """Get total reward for current episode."""
        return sum(self._episode_rewards)
    
    def get_episode_average(self) -> float:
        """Get average reward for current episode."""
        if not self._episode_rewards:
            return 0.0
        return sum(self._episode_rewards) / len(self._episode_rewards)
    
    def get_reward_info(self) -> Dict[str, Any]:
        """
        Get information about reward calculation.
        
        Returns:
            Dictionary with reward information
        """
        return {
            "strategy": self.strategy.__class__.__name__,
            "config": {
                "timestep_penalty": self.config.timestep_penalty,
                "action_reward": self.config.action_reward,
                "separation_loss": self.config.separation_loss,
                "conflict_warning": self.config.conflict_warning,
                "safe_separation_bonus": self.config.safe_separation_bonus,
            },
            "episode_rewards_count": len(self._episode_rewards),
            "episode_total": self.get_episode_total(),
            "episode_average": self.get_episode_average(),
        }


def create_reward_calculator(config: RewardConfig, strategy_name: str = "default") -> RewardCalculator:
    """
    Create reward calculator with specified strategy.
    
    Args:
        config: Reward configuration
        strategy_name: Name of strategy ("default", "safety", "efficiency")
        
    Returns:
        RewardCalculator instance
        
    Raises:
        RewardCalculationError: If strategy name is invalid
    """
    strategies = {
        "default": DefaultRewardStrategy,
        "safety": SafetyFocusedRewardStrategy,
        "efficiency": EfficiencyFocusedRewardStrategy,
    }
    
    if strategy_name not in strategies:
        raise RewardCalculationError(f"Unknown strategy: {strategy_name}. "
                                   f"Available: {list(strategies.keys())}")
    
    strategy_class = strategies[strategy_name]
    strategy = strategy_class(config)
    
    return RewardCalculator(config, strategy)
