"""
ATC Episode Recording System

This module provides recording capabilities for ATC simulation episodes,
allowing users to capture complete scenarios with all aircraft movements over time.
"""

import pickle
import os
import time
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass

from .constants import AIRCRAFT_SPEED


logger = logging.getLogger(__name__)


@dataclass
class AircraftState:
    """Single aircraft state at a point in time."""
    callsign: str
    x: float
    y: float
    altitude: Optional[float] = None
    heading: float = 0.0
    speed: float = 0.0
    is_landing: bool = False
    runway_assigned: Optional[int] = None


class ATCRecorder:
    """Records complete episode state history for visualization."""
    
    def __init__(self, max_aircraft: int = 10, max_timesteps: int = 200):
        """
        Initialize ATC recorder.
        
        Args:
            max_aircraft: Maximum number of aircraft to track
            max_timesteps: Maximum number of timesteps to record
        """
        self.max_aircraft = max_aircraft
        self.max_timesteps = max_timesteps
        
        # Episode data storage
        self.timesteps = []
        self.aircraft_states = []
        self.metadata = {}
        self.current_step = 0
        
        # Pre-allocate arrays for efficiency
        self._reset_arrays()
    
    def _reset_arrays(self) -> None:
        """Reset arrays for new episode."""
        self.timesteps = []
        self.aircraft_states = []
        self.current_step = 0
        self.metadata = {}
    
    def start_episode(self, env_info: Dict[str, Any]) -> None:
        """
        Start recording a new episode.
        
        Args:
            env_info: Environment information dictionary
        """
        self._reset_arrays()
        self.metadata.update({
            'env_type': env_info.get('env_type', 'unknown'),
            'airspace_size': env_info.get('airspace_size', 20.0),
            'runways': env_info.get('runways', []),
            'max_aircraft': env_info.get('max_aircraft', self.max_aircraft),
            'episode_length': env_info.get('episode_length', 150),
            'start_time': time.time(),
        })
        
        logger.debug("Started recording new episode")
    
    def record_step(self, aircraft_list: List[Any], env_info: Dict[str, Any]) -> None:
        """
        Record aircraft states for current timestep.
        
        Args:
            aircraft_list: List of aircraft objects
            env_info: Environment information dictionary
        """
        if self.current_step >= self.max_timesteps:
            logger.warning(f"Maximum timesteps ({self.max_timesteps}) reached, skipping recording")
            return
        
        # Extract aircraft states
        states = []
        for aircraft in aircraft_list:
            state = AircraftState(
                callsign=aircraft.callsign,
                x=aircraft.x,
                y=aircraft.y,
                altitude=getattr(aircraft, 'altitude', None),
                heading=aircraft.heading,
                speed=getattr(aircraft, 'speed', AIRCRAFT_SPEED),
                is_landing=getattr(aircraft, 'is_landing', False),
                runway_assigned=getattr(aircraft, 'runway_assigned', None),
            )
            states.append(state)
        
        self.aircraft_states.append(states)
        self.timesteps.append(self.current_step)
        self.current_step += 1
        
        # Update metadata with final episode info
        self.metadata.update({
            'final_violations': env_info.get('violations', 0),
            'final_conflicts': env_info.get('conflicts', 0),
            'successful_exits': env_info.get('successful_exits', 0),
            'successful_landings': env_info.get('successful_landings', 0),
            'total_score': env_info.get('score', 0),
            'total_timesteps': self.current_step,
        })
    
    def get_episode_data(self) -> Dict[str, Any]:
        """
        Get complete episode data for visualization.
        
        Returns:
            Dictionary containing episode data
        """
        return {
            'timesteps': self.timesteps,
            'aircraft_states': self.aircraft_states,
            'metadata': self.metadata,
        }
    
    def save(self, filepath: str) -> None:
        """
        Save episode data to file.
        
        Args:
            filepath: Path to save the episode data
        """
        episode_data = self.get_episode_data()
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(episode_data, f)
        
        logger.info(f"Episode saved to {filepath}")
    
    @staticmethod
    def load(filepath: str) -> Dict[str, Any]:
        """
        Load episode data from file.
        
        Args:
            filepath: Path to the episode data file
            
        Returns:
            Dictionary containing episode data
        """
        with open(filepath, 'rb') as f:
            episode_data = pickle.load(f)
        
        logger.info(f"Episode loaded from {filepath}")
        return episode_data
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get episode summary statistics.
        
        Returns:
            Dictionary with summary statistics
        """
        if not self.aircraft_states:
            return {}
        
        # Calculate statistics
        total_aircraft = sum(len(states) for states in self.aircraft_states)
        max_aircraft_at_once = max(len(states) for states in self.aircraft_states)
        avg_aircraft = total_aircraft / len(self.aircraft_states) if self.aircraft_states else 0
        
        return {
            'total_timesteps': len(self.timesteps),
            'total_aircraft_instances': total_aircraft,
            'max_aircraft_at_once': max_aircraft_at_once,
            'avg_aircraft_per_timestep': avg_aircraft,
            'episode_duration': self.metadata.get('total_timesteps', 0),
            'final_score': self.metadata.get('total_score', 0),
            'violations': self.metadata.get('final_violations', 0),
            'conflicts': self.metadata.get('final_conflicts', 0),
            'successful_exits': self.metadata.get('successful_exits', 0),
            'successful_landings': self.metadata.get('successful_landings', 0),
        }


def create_recorder_for_env(env) -> ATCRecorder:
    """
    Create a recorder configured for a specific environment.
    
    Args:
        env: Environment instance
        
    Returns:
        Configured ATCRecorder instance
    """
    max_aircraft = getattr(env, 'max_aircraft', 10)
    episode_length = getattr(env, 'episode_length', 150)
    
    recorder = ATCRecorder(max_aircraft=max_aircraft, max_timesteps=episode_length)
    
    # Determine environment type and extract metadata
    env_type = '3d' if hasattr(env, 'runways') else '2d'
    env_info = {
        'env_type': env_type,
        'airspace_size': getattr(env, 'airspace_size', 20.0),
        'max_aircraft': max_aircraft,
        'episode_length': episode_length,
    }
    
    if env_type == '3d':
        env_info['runways'] = getattr(env, 'runways', [])
    
    recorder.start_episode(env_info)
    return recorder
