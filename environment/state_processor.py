"""
State processing module for OpenScope RL environment.

This module provides the StateProcessor class for converting raw game state
into normalized observations suitable for reinforcement learning.
"""

import logging
from typing import Any, Dict, List, Tuple
import numpy as np

from .config import OpenScopeConfig
from .exceptions import StateProcessingError
from .constants import (
    POSITION_SCALE_FACTOR,
    MAX_ANGLE,
    MAX_SPEED,
    MAX_ALTITUDE,
    MAX_GROUND_SPEED,
)


logger = logging.getLogger(__name__)


class StateProcessor:
    """
    Processes raw game state into normalized observations.
    
    This class handles the conversion of OpenScope game state into
    structured observations suitable for reinforcement learning algorithms.
    """
    
    def __init__(self, config: OpenScopeConfig):
        """
        Initialize state processor.
        
        Args:
            config: OpenScope configuration
        """
        self.config = config
        self.max_aircraft = config.max_aircraft
        self.aircraft_feature_dim = config.aircraft_feature_dim
        self.global_state_dim = config.global_state_dim
    
    def process_state(self, state: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Process raw game state into observation.
        
        Args:
            state: Raw game state from OpenScope
            
        Returns:
            Dictionary containing processed observations
            
        Raises:
            StateProcessingError: If state processing fails
        """
        try:
            aircraft_data = state.get("aircraft", [])
            conflicts = state.get("conflicts", [])
            
            # Process aircraft observations
            aircraft_obs, aircraft_mask = self._process_aircraft(aircraft_data)
            
            # Process global state
            global_state = self._process_global_state(state, aircraft_data, conflicts)
            
            # Process conflict matrix
            conflict_matrix = self._process_conflict_matrix(aircraft_data, conflicts)
            
            observation = {
                "aircraft": aircraft_obs,
                "aircraft_mask": aircraft_mask,
                "global_state": global_state,
                "conflict_matrix": conflict_matrix,
            }
            
            logger.debug(f"Processed state: {len(aircraft_data)} aircraft, "
                        f"{len(conflicts)} conflicts")
            
            return observation
            
        except Exception as e:
            logger.error(f"Failed to process state: {e}")
            raise StateProcessingError(f"State processing failed: {e}") from e
    
    def _process_aircraft(self, aircraft_data: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process aircraft data into observation format.
        
        Args:
            aircraft_data: List of aircraft dictionaries
            
        Returns:
            Tuple of (aircraft_obs, aircraft_mask)
        """
        aircraft_obs = np.zeros((self.max_aircraft, self.aircraft_feature_dim), dtype=np.float32)
        aircraft_mask = np.zeros(self.max_aircraft, dtype=bool)
        
        for i, ac in enumerate(aircraft_data[:self.max_aircraft]):
            aircraft_mask[i] = True
            
            # Extract and normalize position
            pos = ac.get("position", [0, 0])
            x_norm, y_norm = self._normalize_position(pos)
            
            # Extract and normalize other features
            altitude_norm = self._normalize_altitude(ac.get("altitude", 0))
            heading_norm = self._normalize_angle(ac.get("heading", 0))
            speed_norm = self._normalize_speed(ac.get("speed", 0))
            ground_speed_norm = self._normalize_speed(ac.get("groundSpeed", 0), MAX_GROUND_SPEED)
            
            assigned_altitude_norm = self._normalize_altitude(ac.get("assignedAltitude", 0))
            assigned_heading_norm = self._normalize_angle(ac.get("assignedHeading", 0))
            assigned_speed_norm = self._normalize_speed(ac.get("assignedSpeed", 0))
            
            # Boolean features
            is_on_ground = 1.0 if ac.get("isOnGround", False) else 0.0
            is_taxiing = 1.0 if ac.get("isTaxiing", False) else 0.0
            is_established = 1.0 if ac.get("isEstablished", False) else 0.0
            
            # Category features
            is_arrival = 1.0 if ac.get("category") == "arrival" else 0.0
            is_departure = 1.0 if ac.get("category") == "departure" else 0.0
            
            aircraft_obs[i, :] = [
                x_norm,                          # 0: x position
                y_norm,                          # 1: y position
                altitude_norm,                   # 2: altitude
                heading_norm,                    # 3: heading
                speed_norm,                      # 4: speed
                ground_speed_norm,               # 5: ground speed
                assigned_altitude_norm,          # 6: assigned altitude
                assigned_heading_norm,            # 7: assigned heading
                assigned_speed_norm,              # 8: assigned speed
                is_on_ground,                    # 9: is on ground
                is_taxiing,                      # 10: is taxiing
                is_established,                  # 11: is established
                is_arrival,                      # 12: is arrival
                is_departure,                    # 13: is departure
            ]
        
        return aircraft_obs, aircraft_mask
    
    def _process_global_state(self, state: Dict[str, Any], aircraft_data: List[Dict[str, Any]], 
                            conflicts: List[Dict[str, Any]]) -> np.ndarray:
        """
        Process global state information.
        
        Args:
            state: Raw game state
            aircraft_data: Aircraft data
            conflicts: Conflict data
            
        Returns:
            Global state array
        """
        global_state = np.array([
            state.get("time", 0) / 3600.0,  # Time in hours
            len(aircraft_data) / self.max_aircraft,  # Aircraft density
            len(conflicts) / 10.0,  # Conflict density (normalized)
            state.get("score", 0) / 1000.0,  # Score (normalized)
        ], dtype=np.float32)
        
        return global_state
    
    def _process_conflict_matrix(self, aircraft_data: List[Dict[str, Any]], 
                               conflicts: List[Dict[str, Any]]) -> np.ndarray:
        """
        Process conflict information into matrix format.
        
        Args:
            aircraft_data: Aircraft data
            conflicts: Conflict data
            
        Returns:
            Conflict matrix
        """
        conflict_matrix = np.zeros((self.max_aircraft, self.max_aircraft), dtype=np.float32)
        
        # Create callsign to index mapping
        callsign_to_idx = {
            ac.get("callsign"): i
            for i, ac in enumerate(aircraft_data[:self.max_aircraft])
        }
        
        # Fill conflict matrix
        for conflict in conflicts:
            cs1 = conflict.get("aircraft1")
            cs2 = conflict.get("aircraft2")
            
            if cs1 in callsign_to_idx and cs2 in callsign_to_idx:
                i, j = callsign_to_idx[cs1], callsign_to_idx[cs2]
                
                # Set conflict severity
                if conflict.get("hasViolation"):
                    severity = 1.0
                elif conflict.get("hasConflict"):
                    severity = 0.5
                else:
                    severity = 0.0
                
                conflict_matrix[i, j] = severity
                conflict_matrix[j, i] = severity  # Symmetric matrix
        
        return conflict_matrix
    
    def _normalize_position(self, position: List[float]) -> Tuple[float, float]:
        """Normalize position coordinates."""
        if not position or len(position) < 2:
            return 0.0, 0.0
        
        x_norm = position[0] / POSITION_SCALE_FACTOR
        y_norm = position[1] / POSITION_SCALE_FACTOR
        
        return x_norm, y_norm
    
    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [0, 1] range."""
        return (angle % MAX_ANGLE) / MAX_ANGLE
    
    def _normalize_speed(self, speed: float, max_speed: float = MAX_SPEED) -> float:
        """Normalize speed to [0, 1] range."""
        return min(speed / max_speed, 1.0)
    
    def _normalize_altitude(self, altitude: float) -> float:
        """Normalize altitude to [0, 1] range."""
        return min(altitude / MAX_ALTITUDE, 1.0)
    
    def get_observation_info(self) -> Dict[str, Any]:
        """
        Get information about observation structure.
        
        Returns:
            Dictionary with observation information
        """
        return {
            "aircraft_shape": (self.max_aircraft, self.aircraft_feature_dim),
            "aircraft_mask_shape": (self.max_aircraft,),
            "global_state_shape": (self.global_state_dim,),
            "conflict_matrix_shape": (self.max_aircraft, self.max_aircraft),
            "aircraft_features": [
                "x_position", "y_position", "altitude", "heading", "speed",
                "ground_speed", "assigned_altitude", "assigned_heading", 
                "assigned_speed", "is_on_ground", "is_taxiing", "is_established",
                "is_arrival", "is_departure"
            ],
            "global_features": [
                "time", "aircraft_density", "conflict_density", "score"
            ]
        }
    
    def validate_observation(self, obs: Dict[str, np.ndarray]) -> bool:
        """
        Validate observation structure and values.
        
        Args:
            obs: Observation dictionary
            
        Returns:
            True if observation is valid
            
        Raises:
            StateProcessingError: If observation is invalid
        """
        required_keys = ["aircraft", "aircraft_mask", "global_state", "conflict_matrix"]
        
        for key in required_keys:
            if key not in obs:
                raise StateProcessingError(f"Missing observation key: {key}")
        
        # Validate shapes
        expected_shapes = {
            "aircraft": (self.max_aircraft, self.aircraft_feature_dim),
            "aircraft_mask": (self.max_aircraft,),
            "global_state": (self.global_state_dim,),
            "conflict_matrix": (self.max_aircraft, self.max_aircraft),
        }
        
        for key, expected_shape in expected_shapes.items():
            if obs[key].shape != expected_shape:
                raise StateProcessingError(f"Invalid shape for {key}: expected {expected_shape}, "
                                         f"got {obs[key].shape}")
        
        # Validate data types
        if obs["aircraft"].dtype != np.float32:
            raise StateProcessingError(f"aircraft must be float32, got {obs['aircraft'].dtype}")
        
        if obs["aircraft_mask"].dtype != np.bool_:
            raise StateProcessingError(f"aircraft_mask must be bool, got {obs['aircraft_mask'].dtype}")
        
        if obs["global_state"].dtype != np.float32:
            raise StateProcessingError(f"global_state must be float32, got {obs['global_state'].dtype}")
        
        if obs["conflict_matrix"].dtype != np.float32:
            raise StateProcessingError(f"conflict_matrix must be float32, got {obs['conflict_matrix'].dtype}")
        
        # Validate value ranges
        if not np.all(np.isfinite(obs["aircraft"])):
            raise StateProcessingError("aircraft observations contain non-finite values")
        
        if not np.all(np.isfinite(obs["global_state"])):
            raise StateProcessingError("global_state observations contain non-finite values")
        
        if not np.all(np.isfinite(obs["conflict_matrix"])):
            raise StateProcessingError("conflict_matrix observations contain non-finite values")
        
        return True
