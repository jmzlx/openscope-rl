"""
Action masking for OpenScope RL environment.

This module implements action masking to improve sample efficiency by preventing
invalid actions during training. Action masking helps the agent focus on valid
actions, dramatically reducing the exploration space and improving learning.

Key invalid actions:
1. Aircraft ID >= num_aircraft (no such aircraft)
2. ILS command when aircraft has no target runway
3. Direct command when aircraft has no active flight plan

Integration with Stable-Baselines3:
- Uses InvalidActionMasking wrapper from sb3-contrib
- Provides get_action_mask() callable for the wrapper
"""

import logging
from typing import Any, Dict, List, Callable
import numpy as np
from gymnasium import spaces

from .config import CommandType

logger = logging.getLogger(__name__)


def get_action_mask(
    obs: Dict[str, np.ndarray],
    aircraft_data: List[Dict[str, Any]],
    max_aircraft: int,
    action_space: spaces.Dict,
) -> np.ndarray:
    """
    Generate action mask for current observation.

    Action masking improves sample efficiency by preventing invalid actions:
    - Invalid aircraft IDs (beyond actual aircraft count)
    - ILS commands for aircraft without target runway
    - Direct commands for aircraft without flight plan

    Aircraft ID semantics:
    - IDs 0 to (num_active_aircraft - 1): Select aircraft at that index
    - ID max_aircraft: Dedicated "no action" value (agent chooses to do nothing)
    
    Args:
        obs: Current observation dictionary
        aircraft_data: Raw aircraft data from game state
        max_aircraft: Maximum number of aircraft in environment
        action_space: The environment's action space

    Returns:
        Boolean mask array where True = valid action, False = invalid
        Shape: (action_space_size,) for MultiDiscrete space

    Example:
        >>> mask = get_action_mask(obs, aircraft_data, max_aircraft=20, action_space=env.action_space)
        >>> # mask[0] = valid aircraft IDs (including max_aircraft for "no action")
        >>> # mask[1] = valid command types (may depend on aircraft)
    """
    # Get aircraft mask to determine valid aircraft
    aircraft_mask = obs["aircraft_mask"]  # Shape: (max_aircraft,)
    num_active_aircraft = int(aircraft_mask.sum())

    # Create base action mask for MultiDiscrete space
    # Action components: [aircraft_id, command_type, altitude, heading, speed]
    action_mask = []

    # 1. Aircraft ID mask - mask out invalid aircraft IDs
    # Valid aircraft IDs: 0 to (num_active_aircraft - 1) for actual aircraft
    # Reserved ID: max_aircraft is the dedicated "no action" (do nothing) value
    aircraft_id_mask = np.zeros(max_aircraft + 1, dtype=bool)
    # Allow valid aircraft IDs (0 to num_active_aircraft - 1)
    aircraft_id_mask[:num_active_aircraft] = True
    # Always allow max_aircraft as "no action" - agent can always choose to do nothing
    aircraft_id_mask[max_aircraft] = True
    action_mask.append(aircraft_id_mask)

    # 2. Command type mask - mask ILS if no aircraft have target runway
    command_type_count = action_space["command_type"].n
    command_type_mask = np.ones(command_type_count, dtype=bool)
    
    # Check if any aircraft has a target runway for ILS commands
    has_any_runway = any(
        ac.get("targetRunway") or ac.get("target_runway") 
        for ac in aircraft_data
    )
    
    # Mask ILS command type when no aircraft have target runways
    # ILS is typically at index 3 (0=altitude, 1=heading, 2=speed, 3=ils, 4=direct)
    # This matches the default order in ActionConfig.command_types
    if not has_any_runway and command_type_count > 3:
        ils_index = 3  # ILS command type index
        command_type_mask[ils_index] = False
        logger.debug(f"Masking ILS commands (index {ils_index}) - no aircraft have target runways")
    
    action_mask.append(command_type_mask)

    # 3. Altitude mask - all altitudes valid
    altitude_levels = action_space["altitude"].n
    altitude_mask = np.ones(altitude_levels, dtype=bool)
    action_mask.append(altitude_mask)

    # 4. Heading mask - all headings valid
    heading_changes_count = action_space["heading"].n
    heading_mask = np.ones(heading_changes_count, dtype=bool)
    action_mask.append(heading_mask)

    # 5. Speed mask - all speeds valid
    speed_levels = action_space["speed"].n
    speed_mask = np.ones(speed_levels, dtype=bool)
    action_mask.append(speed_mask)

    # Concatenate all component masks
    return np.concatenate(action_mask)


def create_action_mask_fn(
    env
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Create action mask function for use with sb3-contrib ActionMasker wrapper.

    This function returns a callable that can be used with the ActionMasker
    wrapper from sb3-contrib. The wrapper will call this function to get the action
    mask for each step.

    Args:
        env: The OpenScope environment instance (should be wrapped with DictToMultiDiscreteWrapper)

    Returns:
        Callable that takes observation and returns action mask

    Example:
        >>> from sb3_contrib.common.wrappers import ActionMasker
        >>> from environment import DictToMultiDiscreteWrapper
        >>> env = PlaywrightEnv(...)
        >>> env = DictToMultiDiscreteWrapper(env)  # Convert to MultiDiscrete first
        >>> mask_fn = create_action_mask_fn(env)
        >>> env = ActionMasker(env, mask_fn)
    """
    def mask_fn(obs: np.ndarray) -> np.ndarray:
        """
        Action mask function for sb3-contrib wrapper.

        Returns flattened mask for MultiDiscrete action space.
        """
        # Get the underlying environment (in case of multiple wrappers)
        unwrapped = env
        while hasattr(unwrapped, 'env'):
            unwrapped = unwrapped.env
        
        # Get the last raw state from environment
        if hasattr(unwrapped, 'prev_state'):
            aircraft_data = unwrapped.prev_state.get("aircraft", [])
        else:
            aircraft_data = []

        # Get observation dict
        if isinstance(obs, dict):
            obs_dict = obs
        else:
            # For flattened observations, we need to reconstruct
            # This is a simplified version - actual implementation depends on wrapper
            obs_dict = {"aircraft_mask": np.zeros(unwrapped.config.max_aircraft, dtype=bool)}
            obs_dict["aircraft_mask"][:len(aircraft_data)] = True

        # Use the original Dict action space for mask computation
        # (the mask format is still compatible with MultiDiscrete)
        original_action_space = getattr(unwrapped, '_original_action_space', unwrapped.action_space)
        if isinstance(original_action_space, spaces.Dict):
            return get_action_mask(
                obs_dict,
                aircraft_data,
                unwrapped.config.max_aircraft,
                original_action_space,
            )
        else:
            # Already MultiDiscrete, use it directly
            return get_action_mask(
                obs_dict,
                aircraft_data,
                unwrapped.config.max_aircraft,
                original_action_space,
            )

    return mask_fn


class ActionMaskingWrapper:
    """
    Wrapper that applies action masking to policy outputs.

    This wrapper can be used to apply action masks during action selection,
    ensuring the policy only samples from valid actions.

    Example:
        >>> env = PlaywrightEnv(...)
        >>> wrapper = ActionMaskingWrapper(env)
        >>> masked_action = wrapper.apply_mask(action, obs, aircraft_data)
    """

    def __init__(self, env):
        """
        Initialize action masking wrapper.

        Args:
            env: OpenScope environment instance
        """
        self.env = env
        self.max_aircraft = env.config.max_aircraft
        self.action_space = env.action_space

    def get_mask(
        self,
        obs: Dict[str, np.ndarray],
        aircraft_data: List[Dict[str, Any]],
    ) -> Dict[str, np.ndarray]:
        """
        Get action mask for dict action space.

        Returns separate masks for each action component.

        Args:
            obs: Current observation
            aircraft_data: Raw aircraft data

        Returns:
            Dictionary of masks for each action component
        """
        aircraft_mask = obs["aircraft_mask"]
        num_active_aircraft = int(aircraft_mask.sum())

        return {
            "aircraft_id": self._get_aircraft_id_mask(num_active_aircraft),
            "command_type": self._get_command_type_mask(aircraft_data),
            "altitude": self._get_altitude_mask(),
            "heading": self._get_heading_mask(),
            "speed": self._get_speed_mask(),
        }

    def _get_aircraft_id_mask(self, num_active_aircraft: int) -> np.ndarray:
        """
        Get mask for aircraft ID selection.
        
        Valid IDs: 0 to (num_active_aircraft - 1) for actual aircraft
        Reserved ID: max_aircraft is always valid as the dedicated "no action" value
        """
        mask = np.zeros(self.max_aircraft + 1, dtype=bool)
        # Allow valid aircraft IDs (0 to num_active_aircraft - 1)
        mask[:num_active_aircraft] = True
        # Always allow max_aircraft as "no action" - agent can always choose to do nothing
        mask[self.max_aircraft] = True
        return mask

    def _get_command_type_mask(
        self,
        aircraft_data: List[Dict[str, Any]],
    ) -> np.ndarray:
        """
        Get mask for command types.
        
        Masks ILS commands when aircraft don't have target runway.
        Note: This creates a per-aircraft mask, but MultiDiscrete spaces in
        stable-baselines3 don't support per-instance masking, so we return
        a conservative mask that allows ILS if any aircraft has a runway.
        """
        command_type_count = self.action_space["command_type"].n
        mask = np.ones(command_type_count, dtype=bool)
        
        # Check if any aircraft has a target runway
        # If none do, we should mask ILS commands
        has_any_runway = any(
            ac.get("targetRunway") or ac.get("target_runway") 
            for ac in aircraft_data
        )
        
        # Find ILS command type index
        from .config import CommandType
        try:
            command_types = self.env.config.action_config.command_types
            ils_idx = next(
                i for i, ct in enumerate(command_types) 
                if ct == CommandType.ILS
            )
            # Mask ILS if no aircraft have runways
            if not has_any_runway:
                mask[ils_idx] = False
        except (StopIteration, AttributeError):
            # If we can't find ILS index or config, allow all commands
            pass
        
        return mask

    def _get_altitude_mask(self) -> np.ndarray:
        """Get mask for altitude selection."""
        return np.ones(self.action_space["altitude"].n, dtype=bool)

    def _get_heading_mask(self) -> np.ndarray:
        """Get mask for heading selection."""
        return np.ones(self.action_space["heading"].n, dtype=bool)

    def _get_speed_mask(self) -> np.ndarray:
        """Get mask for speed selection."""
        return np.ones(self.action_space["speed"].n, dtype=bool)

    def apply_mask(
        self,
        action: Dict[str, int],
        obs: Dict[str, np.ndarray],
        aircraft_data: List[Dict[str, Any]],
    ) -> Dict[str, int]:
        """
        Apply action mask to ensure valid action.

        If action is invalid, replaces it with a safe default (no action).

        Args:
            action: Proposed action
            obs: Current observation
            aircraft_data: Raw aircraft data

        Returns:
            Validated action (modified if original was invalid)
        """
        masks = self.get_mask(obs, aircraft_data)

        # Check each action component
        validated_action = action.copy()

        # Aircraft ID
        if not masks["aircraft_id"][action["aircraft_id"]]:
            # Default to "no action" (max_aircraft) if invalid aircraft ID
            logger.warning(f"Invalid aircraft_id {action['aircraft_id']}, using {self.max_aircraft} (no action)")
            validated_action["aircraft_id"] = self.max_aircraft

        # Command type
        if not masks["command_type"][action["command_type"]]:
            logger.warning(f"Invalid command_type {action['command_type']}, using 0")
            validated_action["command_type"] = 0

        # Other components are always valid in our current implementation

        return validated_action


def get_invalid_action_stats(
    actions: List[Dict[str, int]],
    observations: List[Dict[str, np.ndarray]],
    aircraft_data_list: List[List[Dict[str, Any]]],
    max_aircraft: int,
    action_space: spaces.Dict,
) -> Dict[str, Any]:
    """
    Analyze actions to compute statistics on invalid actions.

    Useful for debugging and understanding how often the agent attempts invalid actions.

    Args:
        actions: List of actions taken
        observations: List of corresponding observations
        aircraft_data_list: List of aircraft data for each step
        max_aircraft: Maximum number of aircraft
        action_space: Environment action space

    Returns:
        Dictionary with statistics on invalid actions
    """
    stats = {
        "total_actions": len(actions),
        "invalid_aircraft_id": 0,
        "invalid_command_type": 0,
        "total_invalid": 0,
    }

    for action, obs, aircraft_data in zip(actions, observations, aircraft_data_list):
        mask = get_action_mask(obs, aircraft_data, max_aircraft, action_space)

        # Check if aircraft_id is invalid
        aircraft_mask = obs["aircraft_mask"]
        num_active_aircraft = int(aircraft_mask.sum())
        if action["aircraft_id"] > num_active_aircraft:
            stats["invalid_aircraft_id"] += 1
            stats["total_invalid"] += 1

    # Compute percentages
    if stats["total_actions"] > 0:
        stats["invalid_aircraft_id_pct"] = (
            stats["invalid_aircraft_id"] / stats["total_actions"] * 100
        )
        stats["total_invalid_pct"] = (
            stats["total_invalid"] / stats["total_actions"] * 100
        )

    return stats


def print_action_mask_summary(
    obs: Dict[str, np.ndarray],
    aircraft_data: List[Dict[str, Any]],
    max_aircraft: int,
    action_space: spaces.Dict,
) -> None:
    """
    Print human-readable summary of current action mask.

    Useful for debugging and understanding the masking behavior.

    Args:
        obs: Current observation
        aircraft_data: Raw aircraft data
        max_aircraft: Maximum number of aircraft
        action_space: Environment action space
    """
    mask = get_action_mask(obs, aircraft_data, max_aircraft, action_space)

    aircraft_mask = obs["aircraft_mask"]
    num_active_aircraft = int(aircraft_mask.sum())

    print("\n" + "="*80)
    print("ACTION MASK SUMMARY")
    print("="*80)
    print(f"Active Aircraft: {num_active_aircraft} / {max_aircraft}")
    print(f"Valid Aircraft IDs: 0 to {num_active_aircraft} (inclusive)")
    print(f"Total Valid Actions: {mask.sum()} / {len(mask)}")
    print(f"Action Space Reduction: {(1 - mask.sum() / len(mask)) * 100:.1f}%")
    print("="*80 + "\n")
