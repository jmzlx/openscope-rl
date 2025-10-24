"""
Observation and action space definitions for OpenScope RL environment.

This module defines the gymnasium spaces used for observations and actions,
providing a clean interface for space creation and validation.
"""

import numpy as np
from gymnasium import spaces
from typing import Dict, Any

from .constants import (
    AIRCRAFT_FEATURE_DIM,
    GLOBAL_STATE_DIM,
    MAX_AIRCRAFT_DEFAULT,
    COMMAND_TYPE_COUNT,
    ALTITUDE_LEVELS,
    HEADING_CHANGES_COUNT,
    SPEED_LEVELS,
)


def create_observation_space(max_aircraft: int = MAX_AIRCRAFT_DEFAULT) -> spaces.Dict:
    """
    Create observation space for the environment.
    
    Args:
        max_aircraft: Maximum number of aircraft to track
        
    Returns:
        Dict space containing aircraft observations, mask, global state, and conflict matrix
    """
    return spaces.Dict({
        "aircraft": spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(max_aircraft, AIRCRAFT_FEATURE_DIM),
            dtype=np.float32
        ),
        "aircraft_mask": spaces.Box(
            low=0,
            high=1,
            shape=(max_aircraft,),
            dtype=np.bool_
        ),
        "global_state": spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(GLOBAL_STATE_DIM,),
            dtype=np.float32
        ),
        "conflict_matrix": spaces.Box(
            low=0,
            high=1,
            shape=(max_aircraft, max_aircraft),
            dtype=np.float32
        ),
    })


def create_action_space(max_aircraft: int = MAX_AIRCRAFT_DEFAULT) -> spaces.Dict:
    """
    Create action space for the environment.
    
    Args:
        max_aircraft: Maximum number of aircraft to control
        
    Returns:
        Dict space containing aircraft selection and command parameters
    """
    return spaces.Dict({
        "aircraft_id": spaces.Discrete(max_aircraft + 1),  # +1 for "no action"
        "command_type": spaces.Discrete(COMMAND_TYPE_COUNT),
        "altitude": spaces.Discrete(ALTITUDE_LEVELS),
        "heading": spaces.Discrete(HEADING_CHANGES_COUNT),
        "speed": spaces.Discrete(SPEED_LEVELS),
    })


def create_multi_discrete_action_space(max_aircraft: int = MAX_AIRCRAFT_DEFAULT) -> spaces.MultiDiscrete:
    """
    Create MultiDiscrete action space for Stable-Baselines3 compatibility.
    
    Args:
        max_aircraft: Maximum number of aircraft to control
        
    Returns:
        MultiDiscrete space with action components
    """
    return spaces.MultiDiscrete([
        max_aircraft + 1,  # aircraft_id
        COMMAND_TYPE_COUNT,  # command_type
        ALTITUDE_LEVELS,  # altitude
        HEADING_CHANGES_COUNT,  # heading
        SPEED_LEVELS,  # speed
    ])


def validate_observation(obs: Dict[str, np.ndarray], max_aircraft: int) -> bool:
    """
    Validate observation structure and shapes.
    
    Args:
        obs: Observation dictionary
        max_aircraft: Maximum number of aircraft
        
    Returns:
        True if observation is valid
        
    Raises:
        ValueError: If observation is invalid
    """
    required_keys = ["aircraft", "aircraft_mask", "global_state", "conflict_matrix"]
    
    for key in required_keys:
        if key not in obs:
            raise ValueError(f"Missing observation key: {key}")
    
    # Validate shapes
    expected_shapes = {
        "aircraft": (max_aircraft, AIRCRAFT_FEATURE_DIM),
        "aircraft_mask": (max_aircraft,),
        "global_state": (GLOBAL_STATE_DIM,),
        "conflict_matrix": (max_aircraft, max_aircraft),
    }
    
    for key, expected_shape in expected_shapes.items():
        if obs[key].shape != expected_shape:
            raise ValueError(f"Invalid shape for {key}: expected {expected_shape}, "
                           f"got {obs[key].shape}")
    
    # Validate data types
    if obs["aircraft"].dtype != np.float32:
        raise ValueError(f"aircraft must be float32, got {obs['aircraft'].dtype}")
    
    if obs["aircraft_mask"].dtype != np.bool_:
        raise ValueError(f"aircraft_mask must be bool, got {obs['aircraft_mask'].dtype}")
    
    if obs["global_state"].dtype != np.float32:
        raise ValueError(f"global_state must be float32, got {obs['global_state'].dtype}")
    
    if obs["conflict_matrix"].dtype != np.float32:
        raise ValueError(f"conflict_matrix must be float32, got {obs['conflict_matrix'].dtype}")
    
    return True


def validate_action(action: Dict[str, int], max_aircraft: int) -> bool:
    """
    Validate action structure and values.
    
    Args:
        action: Action dictionary
        max_aircraft: Maximum number of aircraft
        
    Returns:
        True if action is valid
        
    Raises:
        ValueError: If action is invalid
    """
    required_keys = ["aircraft_id", "command_type", "altitude", "heading", "speed"]
    
    for key in required_keys:
        if key not in action:
            raise ValueError(f"Missing action key: {key}")
    
    # Validate aircraft_id
    if not (0 <= action["aircraft_id"] <= max_aircraft):
        raise ValueError(f"aircraft_id must be in [0, {max_aircraft}], "
                        f"got {action['aircraft_id']}")
    
    # Validate command_type
    if not (0 <= action["command_type"] < COMMAND_TYPE_COUNT):
        raise ValueError(f"command_type must be in [0, {COMMAND_TYPE_COUNT}), "
                        f"got {action['command_type']}")
    
    # Validate altitude
    if not (0 <= action["altitude"] < ALTITUDE_LEVELS):
        raise ValueError(f"altitude must be in [0, {ALTITUDE_LEVELS}), "
                        f"got {action['altitude']}")
    
    # Validate heading
    if not (0 <= action["heading"] < HEADING_CHANGES_COUNT):
        raise ValueError(f"heading must be in [0, {HEADING_CHANGES_COUNT}), "
                        f"got {action['heading']}")
    
    # Validate speed
    if not (0 <= action["speed"] < SPEED_LEVELS):
        raise ValueError(f"speed must be in [0, {SPEED_LEVELS}), "
                        f"got {action['speed']}")
    
    return True


def get_action_space_info(max_aircraft: int = MAX_AIRCRAFT_DEFAULT) -> Dict[str, Any]:
    """
    Get information about the action space.
    
    Args:
        max_aircraft: Maximum number of aircraft
        
    Returns:
        Dictionary with action space information
    """
    return {
        "aircraft_id": {
            "min": 0,
            "max": max_aircraft,
            "description": "Aircraft to control (0 = no action)"
        },
        "command_type": {
            "min": 0,
            "max": COMMAND_TYPE_COUNT - 1,
            "description": "Type of command to issue"
        },
        "altitude": {
            "min": 0,
            "max": ALTITUDE_LEVELS - 1,
            "description": "Altitude level index"
        },
        "heading": {
            "min": 0,
            "max": HEADING_CHANGES_COUNT - 1,
            "description": "Heading change index"
        },
        "speed": {
            "min": 0,
            "max": SPEED_LEVELS - 1,
            "description": "Speed level index"
        }
    }
