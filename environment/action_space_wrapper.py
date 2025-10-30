"""
Action space wrapper for converting Dict to MultiDiscrete for Stable-Baselines3.

This wrapper converts the Dict action space used by the environment to a 
MultiDiscrete action space compatible with PPO and other SB3 algorithms.
"""

import numpy as np
from typing import Dict, Any, Tuple
from gymnasium import spaces
from gymnasium.core import Wrapper, ObsType, ActType

from .spaces import create_multi_discrete_action_space
from .constants import (
    COMMAND_TYPE_COUNT,
    ALTITUDE_LEVELS,
    HEADING_CHANGES_COUNT,
    SPEED_LEVELS,
)


class DictToMultiDiscreteWrapper(Wrapper):
    """
    Wrapper that converts Dict action space to MultiDiscrete for SB3 compatibility.
    
    The environment uses a Dict action space:
        {
            'aircraft_id': Discrete(max_aircraft + 1),
            'command_type': Discrete(5),
            'altitude': Discrete(18),
            'heading': Discrete(13),
            'speed': Discrete(8)
        }
    
    This wrapper converts it to MultiDiscrete([max_aircraft+1, 5, 18, 13, 8])
    for compatibility with PPO and other SB3 algorithms.
    
    Example:
        >>> env = PlaywrightEnv(...)
        >>> env = DictToMultiDiscreteWrapper(env)
        >>> model = PPO("MultiInputPolicy", env)  # Now works with PPO
    """
    
    def __init__(self, env):
        """
        Initialize wrapper.
        
        Args:
            env: Environment with Dict action space
        """
        super().__init__(env)
        
        # Store the original Dict action space
        if not isinstance(env.action_space, spaces.Dict):
            raise ValueError(
                f"Expected Dict action space, got {type(env.action_space)}"
            )
        
        self._original_action_space = env.action_space
        self.max_aircraft = env.config.max_aircraft
        
        # Create MultiDiscrete action space
        self.action_space = create_multi_discrete_action_space(self.max_aircraft)
    
    def step(self, action: np.ndarray) -> Tuple[ObsType, float, bool, bool, Dict[str, Any]]:
        """
        Convert MultiDiscrete action to Dict and step environment.
        
        Args:
            action: MultiDiscrete action array [aircraft_id, command_type, altitude, heading, speed]
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Ensure action is a numpy array and has the correct shape
        if not isinstance(action, np.ndarray):
            action = np.array(action, dtype=np.int32)
        
        # Flatten if needed (in case action is multi-dimensional)
        action = action.flatten()
        
        if len(action) != 5:
            raise ValueError(
                f"Expected action with 5 components, got {len(action)}: {action}"
            )
        
        # Convert MultiDiscrete to Dict
        dict_action = {
            "aircraft_id": int(action[0]),
            "command_type": int(action[1]),
            "altitude": int(action[2]),
            "heading": int(action[3]),
            "speed": int(action[4]),
        }
        
        # Step underlying environment with Dict action
        return self.env.step(dict_action)
    
    def action(self, action: np.ndarray) -> Dict[str, int]:
        """
        Convert MultiDiscrete action to Dict format.
        
        This method is used internally by SB3 when checking action spaces.
        """
        return {
            "aircraft_id": int(action[0]),
            "command_type": int(action[1]),
            "altitude": int(action[2]),
            "heading": int(action[3]),
            "speed": int(action[4]),
        }
