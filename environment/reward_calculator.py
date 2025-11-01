"""
Reward calculation module for OpenScope RL environment.

This module provides the RewardCalculator class for computing rewards
based on game state changes and configurable reward shaping.

Enhanced with:
- Trajectory prediction for immediate feedback
- Dense separation tracking
- Command-consequence attribution
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple
from abc import ABC, abstractmethod

from .config import RewardConfig
from .exceptions import RewardCalculationError


logger = logging.getLogger(__name__)


# ============================================================================
# TRAJECTORY PREDICTION UTILITIES
# ============================================================================

def predict_position(aircraft: Dict[str, Any], time_seconds: float) -> Tuple[float, float, float]:
    """
    Predict aircraft position after given time.
    
    Simple kinematic prediction assuming constant heading and speed.
    
    Args:
        aircraft: Aircraft state dictionary
        time_seconds: Time to predict ahead (seconds)
        
    Returns:
        Tuple of (x, y, altitude) in nm and feet
    """
    pos = aircraft.get("position", [0, 0])
    heading = aircraft.get("heading", 0)
    speed = aircraft.get("speed", 0)  # knots
    altitude = aircraft.get("altitude", 0)  # hundreds of feet
    
    # Convert speed from knots to nm/second
    speed_nm_per_sec = speed / 3600.0
    
    # Calculate displacement
    heading_rad = math.radians(heading)
    dx = speed_nm_per_sec * time_seconds * math.sin(heading_rad)
    dy = speed_nm_per_sec * time_seconds * math.cos(heading_rad)
    
    # Predict position
    future_x = pos[0] + dx
    future_y = pos[1] + dy
    future_alt = altitude  # Assume constant altitude for short predictions
    
    return (future_x, future_y, future_alt)


def calculate_distance_3d(pos1: Tuple[float, float, float], 
                          pos2: Tuple[float, float, float]) -> float:
    """
    Calculate 3D distance between two positions.
    
    Args:
        pos1: (x, y, altitude) in nm and hundreds of feet
        pos2: (x, y, altitude) in nm and hundreds of feet
        
    Returns:
        Distance in nautical miles (horizontal + vertical component)
    """
    # Horizontal distance
    dx = pos1[0] - pos2[0]
    dy = pos1[1] - pos2[1]
    horizontal_dist = math.sqrt(dx*dx + dy*dy)
    
    # Vertical distance (convert hundreds of feet to nm: 1nm = 6076ft)
    dalt_feet = abs(pos1[2] - pos2[2]) * 100
    vertical_dist_nm = dalt_feet / 6076.0
    
    # Combined 3D distance
    return math.sqrt(horizontal_dist**2 + vertical_dist_nm**2)


def predict_future_conflicts(aircraft_data: List[Dict[str, Any]], 
                             prediction_horizons: List[float] = [30, 60]) -> List[Dict[str, Any]]:
    """
    Predict future conflicts by simulating aircraft trajectories.
    
    Args:
        aircraft_data: List of aircraft state dictionaries
        prediction_horizons: Time horizons to check (seconds)
        
    Returns:
        List of predicted conflict dictionaries with fields:
        - ac1_id, ac2_id: Aircraft IDs
        - time_to_conflict: Seconds until conflict
        - predicted_distance: Predicted closest approach (nm)
        - severity: 0.0-1.0 conflict severity
    """
    predicted_conflicts = []
    
    for horizon in prediction_horizons:
        # Predict all aircraft positions
        future_positions = {}
        for ac in aircraft_data:
            ac_id = ac.get("callsign", "")
            if ac_id:
                future_positions[ac_id] = predict_position(ac, horizon)
        
        # Check all pairs for conflicts
        aircraft_ids = list(future_positions.keys())
        for i in range(len(aircraft_ids)):
            for j in range(i + 1, len(aircraft_ids)):
                ac1_id = aircraft_ids[i]
                ac2_id = aircraft_ids[j]
                
                pos1 = future_positions[ac1_id]
                pos2 = future_positions[ac2_id]
                
                predicted_dist = calculate_distance_3d(pos1, pos2)
                
                # Conflict if predicted distance < 5nm
                if predicted_dist < 5.0:
                    severity = (5.0 - predicted_dist) / 5.0
                    
                    predicted_conflicts.append({
                        'ac1_id': ac1_id,
                        'ac2_id': ac2_id,
                        'time_to_conflict': horizon,
                        'predicted_distance': predicted_dist,
                        'severity': severity,
                    })
    
    return predicted_conflicts


def calculate_separation_changes(state: Dict[str, Any], 
                                 prev_state: Dict[str, Any],
                                 action: Optional[Dict[str, int]] = None) -> Dict[str, float]:
    """
    Calculate separation changes for all aircraft pairs.
    
    If an action was taken, specifically tracks the commanded aircraft's
    separation changes to attribute consequences to the command.
    
    Args:
        state: Current game state
        prev_state: Previous game state  
        action: Action taken (if any)
        
    Returns:
        Dictionary with separation change metrics
    """
    curr_aircraft = state.get("aircraft", [])
    prev_aircraft = prev_state.get("aircraft", [])
    
    # Build lookup by callsign
    curr_by_id = {ac.get("callsign"): ac for ac in curr_aircraft}
    prev_by_id = {ac.get("callsign"): ac for ac in prev_aircraft}
    
    # Track separation changes
    improvements = []
    deteriorations = []
    commanded_aircraft_id = None
    
    if action:
        # Extract commanded aircraft ID from action
        # This would need to be passed from the environment
        commanded_aircraft_id = action.get("aircraft_id")
    
    # Check all pairs that exist in both states
    common_ids = set(curr_by_id.keys()) & set(prev_by_id.keys())
    ids_list = list(common_ids)
    
    for i in range(len(ids_list)):
        for j in range(i + 1, len(ids_list)):
            id1, id2 = ids_list[i], ids_list[j]
            
            # Current separation
            curr_pos1 = curr_by_id[id1].get("position", [0, 0])
            curr_pos2 = curr_by_id[id2].get("position", [0, 0])
            curr_dist = math.sqrt(
                (curr_pos1[0] - curr_pos2[0])**2 + 
                (curr_pos1[1] - curr_pos2[1])**2
            )
            
            # Previous separation
            prev_pos1 = prev_by_id[id1].get("position", [0, 0])
            prev_pos2 = prev_by_id[id2].get("position", [0, 0])
            prev_dist = math.sqrt(
                (prev_pos1[0] - prev_pos2[0])**2 + 
                (prev_pos1[1] - prev_pos2[1])**2
            )
            
            # Calculate change
            sep_change = curr_dist - prev_dist
            
            # Track if commanded aircraft was involved
            involves_commanded = commanded_aircraft_id in [id1, id2]
            
            if sep_change > 0.1:  # Improved by >0.1nm
                improvements.append({
                    'pair': (id1, id2),
                    'change': sep_change,
                    'current_dist': curr_dist,
                    'attributed': involves_commanded
                })
            elif sep_change < -0.1:  # Worsened by >0.1nm
                deteriorations.append({
                    'pair': (id1, id2),
                    'change': sep_change,
                    'current_dist': curr_dist,
                    'attributed': involves_commanded
                })
    
    return {
        'improvements': improvements,
        'deteriorations': deteriorations,
        'commanded_aircraft': commanded_aircraft_id,
    }


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
        Calculate reward using default strategy with dense feedback.
        
        Enhanced with:
        - Graduated conflict penalties (not binary)
        - Aircraft exit bonuses
        - Increased safe separation rewards
        - Progress tracking
        
        Args:
            state: Current game state
            prev_state: Previous game state
            action: Action taken (optional)
            
        Returns:
            Calculated reward
        """
        reward = 0.0
        
        # Base score change from OpenScope
        score_change = state.get("score", 0) - prev_state.get("score", 0)
        reward += score_change
        
        # Time penalty (increased from -0.01 to -0.1 for efficiency pressure)
        reward += self.config.timestep_penalty
        
        # Action reward
        if action is not None:
            reward += self.config.action_reward
        
        # Graduated conflict and violation penalties (not binary)
        conflicts = state.get("conflicts", [])
        CRITICAL_DISTANCE = 5.0  # nm
        
        for conflict in conflicts:
            distance = conflict.get("distance", float('inf'))
            
            # Graduated penalty based on proximity
            if distance < CRITICAL_DISTANCE:
                severity = (CRITICAL_DISTANCE - distance) / CRITICAL_DISTANCE
                reward += self.config.conflict_warning * severity * 5  # -2 to -10 based on proximity
            
            # Heavy penalty for actual violations
            if conflict.get("hasViolation"):
                reward += self.config.separation_loss  # -200
        
        # Aircraft exit bonus (track aircraft count changes)
        curr_aircraft = state.get("aircraft", [])
        prev_aircraft = prev_state.get("aircraft", [])
        aircraft_exited = len(prev_aircraft) - len(curr_aircraft)
        
        if aircraft_exited > 0:
            reward += self.config.successful_exit_bonus * aircraft_exited  # 50 per aircraft
        
        # Maintaining safe separation bonus (increased from 0.02 to 0.1)
        if len(conflicts) == 0 and len(curr_aircraft) > 0:
            reward += self.config.safe_separation_bonus * len(curr_aircraft)  # Scale with aircraft count
        
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


class PredictiveRewardStrategy(RewardStrategy):
    """
    Enhanced reward strategy with trajectory prediction and credit assignment.
    
    Key features:
    1. Predicts future conflicts (30s, 60s ahead) for immediate feedback
    2. Tracks separation changes for all aircraft pairs
    3. Attributes consequences to commanded aircraft
    4. Dense rewards for maintaining/improving separation
    
    This dramatically improves credit assignment by giving IMMEDIATE feedback
    about command quality, rather than waiting 50+ steps for conflicts to manifest.
    """
    
    def __init__(self, config: RewardConfig):
        """
        Initialize predictive reward strategy.
        
        Args:
            config: Reward configuration
        """
        self.config = config
    
    def calculate_reward(self, state: Dict[str, Any], prev_state: Dict[str, Any], 
                        action: Optional[Dict[str, int]] = None) -> float:
        """
        Calculate reward with predictive feedback and credit assignment.
        
        Args:
            state: Current game state
            prev_state: Previous game state
            action: Action taken (optional, should include 'aircraft_id')
            
        Returns:
            Calculated reward
        """
        reward = 0.0
        
        # ====================================================================
        # 1. BASE REWARDS (from game state)
        # ====================================================================
        
        # Score change from OpenScope
        score_change = state.get("score", 0) - prev_state.get("score", 0)
        reward += score_change
        
        # Time penalty
        reward += self.config.timestep_penalty
        
        # Action reward
        if action is not None:
            reward += self.config.action_reward
        
        # ====================================================================
        # 2. CURRENT CONFLICTS (graduated penalties)
        # ====================================================================
        
        conflicts = state.get("conflicts", [])
        CRITICAL_DISTANCE = 5.0
        
        for conflict in conflicts:
            distance = conflict.get("distance", float('inf'))
            
            if distance < CRITICAL_DISTANCE:
                severity = (CRITICAL_DISTANCE - distance) / CRITICAL_DISTANCE
                reward += self.config.conflict_warning * severity * 5
            
            if conflict.get("hasViolation"):
                reward += self.config.separation_loss
        
        # ====================================================================
        # 3. PREDICTIVE CONFLICT DETECTION (IMMEDIATE FEEDBACK)
        # ====================================================================
        
        curr_aircraft = state.get("aircraft", [])
        
        if len(curr_aircraft) > 1:
            # Predict conflicts 30s and 60s ahead
            predicted_conflicts = predict_future_conflicts(
                curr_aircraft, 
                prediction_horizons=[30, 60]
            )
            
            for pred_conflict in predicted_conflicts:
                severity = pred_conflict['severity']
                time_to_conflict = pred_conflict['time_to_conflict']
                
                # IMMEDIATE penalty scaled by severity and time
                # Closer conflicts = higher penalty
                time_factor = 2.0 if time_to_conflict <= 30 else 1.0
                
                # Give immediate feedback about this command's future consequences
                penalty = -5.0 * severity * time_factor
                reward += penalty
                
                # Extra penalty if commanded aircraft is involved
                if action:
                    commanded_id = action.get("aircraft_id")
                    if commanded_id in [pred_conflict['ac1_id'], pred_conflict['ac2_id']]:
                        reward -= 5.0 * severity  # Double penalty for commanded aircraft
        
        # ====================================================================
        # 4. DENSE SEPARATION TRACKING (command-consequence attribution)
        # ====================================================================
        
        sep_changes = calculate_separation_changes(state, prev_state, action)
        
        # Reward improvements in separation
        for improvement in sep_changes['improvements']:
            change = improvement['change']
            attributed = improvement['attributed']
            
            # Base reward for any improvement
            reward += 0.5 * min(change, 2.0)  # Cap at 2nm improvement
            
            # BONUS if this was due to our command
            if attributed:
                reward += 1.5 * min(change, 2.0)
        
        # Penalize deteriorations in separation
        for deterioration in sep_changes['deteriorations']:
            change = abs(deterioration['change'])
            current_dist = deterioration['current_dist']
            attributed = deterioration['attributed']
            
            # Penalty scaled by how close they are getting
            if current_dist < 7.0:  # Within 7nm buffer
                proximity_factor = (7.0 - current_dist) / 7.0
                penalty = -2.0 * change * (1.0 + proximity_factor)
                reward += penalty
                
                # EXTRA penalty if this was due to our command
                if attributed:
                    reward += penalty * 2.0  # Triple total penalty
        
        # ====================================================================
        # 5. SAFE SEPARATION MAINTENANCE
        # ====================================================================
        
        # Reward maintaining safe separation (no conflicts)
        if len(conflicts) == 0 and len(curr_aircraft) > 0:
            reward += self.config.safe_separation_bonus * len(curr_aircraft)
        
        # ====================================================================
        # 6. AIRCRAFT EXITS
        # ====================================================================
        
        prev_aircraft = prev_state.get("aircraft", [])
        aircraft_exited = len(prev_aircraft) - len(curr_aircraft)
        
        if aircraft_exited > 0:
            reward += self.config.successful_exit_bonus * aircraft_exited
        
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


def create_reward_calculator(config: RewardConfig, strategy_name: str = "predictive") -> RewardCalculator:
    """
    Create reward calculator with specified strategy.
    
    Args:
        config: Reward configuration
        strategy_name: Name of strategy:
            - "default": Basic reward shaping with graduated penalties
            - "predictive": (RECOMMENDED) Trajectory prediction + credit assignment
            - "safety": Focus on minimizing conflicts
            - "efficiency": Focus on maximizing throughput
        
    Returns:
        RewardCalculator instance
        
    Raises:
        RewardCalculationError: If strategy name is invalid
    """
    strategies = {
        "default": DefaultRewardStrategy,
        "predictive": PredictiveRewardStrategy,  # NEW: Recommended for curriculum learning
        "safety": SafetyFocusedRewardStrategy,
        "efficiency": EfficiencyFocusedRewardStrategy,
    }
    
    if strategy_name not in strategies:
        raise RewardCalculationError(f"Unknown strategy: {strategy_name}. "
                                   f"Available: {list(strategies.keys())}")
    
    strategy_class = strategies[strategy_name]
    strategy = strategy_class(config)
    
    logger.info(f"Created reward calculator with '{strategy_name}' strategy")
    
    return RewardCalculator(config, strategy)
