"""
Abstract interface for ATC environment interaction.

This module defines the abstract interface that separates OpenScope-specific
implementation from the models. This allows:
1. Easy testing with mock implementations
2. Swapping OpenScope with other simulators
3. Clear contract between environment and models
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import numpy as np


class ATCEnvironmentInterface(ABC):
    """
    Abstract interface for ATC environment interaction.
    
    This interface defines the contract that all ATC environments must implement,
    allowing models to work with any implementation (OpenScope, Cosmos, mock, etc.)
    """
    
    @abstractmethod
    def get_observation_space(self):
        """Get the observation space definition."""
        pass
    
    @abstractmethod
    def get_action_space(self):
        """Get the action space definition."""
        pass
    
    @abstractmethod
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Reset environment to initial state.
        
        Returns:
            Tuple of (observation, info)
        """
        pass
    
    @abstractmethod
    def step(self, action: Dict[str, int]) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Execute action and return next state.
        
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Clean up resources."""
        pass


class ATCStateExtractor(ABC):
    """
    Abstract interface for extracting state from environment.
    
    This separates state extraction logic, making it easier to test and swap implementations.
    """
    
    @abstractmethod
    def extract_state(self) -> Dict[str, Any]:
        """
        Extract raw game state from environment.
        
        Returns:
            Dictionary containing raw state data (aircraft, conflicts, score, etc.)
        """
        pass
    
    @abstractmethod
    def extract_observation(self, state: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Process raw state into normalized observation.
        
        Args:
            state: Raw game state
            
        Returns:
            Normalized observation dictionary
        """
        pass


class ATCCommandExecutor(ABC):
    """
    Abstract interface for executing commands in environment.
    
    This separates command execution logic from the environment implementation.
    """
    
    @abstractmethod
    def execute_command(self, command: str) -> None:
        """
        Execute a command in the environment.
        
        Args:
            command: Command string to execute
        """
        pass
    
    @abstractmethod
    def execute_action(self, action: Dict[str, int], aircraft_data: list) -> Optional[str]:
        """
        Execute an action in the environment.
        
        Args:
            action: Action dictionary
            aircraft_data: Current aircraft data
            
        Returns:
            Command string that was executed, or None if no action
        """
        pass


class OpenScopeAdapter(ATCEnvironmentInterface):
    """
    Adapter that wraps PlaywrightEnv to implement ATCEnvironmentInterface.
    
    This provides a clean interface that models can depend on, while the actual
    OpenScope implementation details are encapsulated.
    """
    
    def __init__(self, playwright_env):
        """
        Initialize adapter.
        
        Args:
            playwright_env: PlaywrightEnv instance
        """
        self._env = playwright_env
    
    def get_observation_space(self):
        """Get observation space from wrapped environment."""
        return self._env.observation_space
    
    def get_action_space(self):
        """Get action space from wrapped environment."""
        return self._env.action_space
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset wrapped environment."""
        return self._env.reset(seed=seed, options=options)
    
    def step(self, action: Dict[str, int]) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Step wrapped environment."""
        return self._env.step(action)
    
    def close(self) -> None:
        """Close wrapped environment."""
        self._env.close()
    
    @property
    def wrapped_env(self):
        """Access to wrapped environment for advanced usage."""
        return self._env


class MockATCEnvironment(ATCEnvironmentInterface):
    """
    Mock implementation of ATCEnvironmentInterface for testing.
    
    This allows testing models without requiring OpenScope to be running.
    """
    
    def __init__(self, max_aircraft: int = 20, aircraft_feature_dim: int = 14):
        """
        Initialize mock environment.
        
        Args:
            max_aircraft: Maximum number of aircraft
            aircraft_feature_dim: Dimension of aircraft features
        """
        from environment.spaces import create_observation_space, create_action_space
        
        self.max_aircraft = max_aircraft
        self.aircraft_feature_dim = aircraft_feature_dim
        self.observation_space = create_observation_space(max_aircraft)
        self.action_space = create_action_space(max_aircraft)
        self.current_step = 0
        self.episode_length = 1000
    
    def get_observation_space(self):
        """Get mock observation space."""
        return self.observation_space
    
    def get_action_space(self):
        """Get mock action space."""
        return self.action_space
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset mock environment."""
        self.current_step = 0
        
        # Generate mock observation
        obs = {
            "aircraft": np.random.rand(self.max_aircraft, self.aircraft_feature_dim).astype(np.float32),
            "aircraft_mask": np.array([True] * 2 + [False] * (self.max_aircraft - 2), dtype=bool),
            "global_state": np.array([0.0, 0.1, 0.0, 1.0], dtype=np.float32),
            "conflict_matrix": np.zeros((self.max_aircraft, self.max_aircraft), dtype=np.float32),
        }
        
        info = {
            "raw_state": {
                "aircraft": [],
                "conflicts": [],
                "score": 1000,
                "time": 0.0,
            }
        }
        
        return obs, info
    
    def step(self, action: Dict[str, int]) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Step mock environment."""
        self.current_step += 1
        
        # Generate next mock observation
        obs = {
            "aircraft": np.random.rand(self.max_aircraft, self.aircraft_feature_dim).astype(np.float32),
            "aircraft_mask": np.array([True] * 2 + [False] * (self.max_aircraft - 2), dtype=bool),
            "global_state": np.array([self.current_step / 3600.0, 0.1, 0.0, 1.0], dtype=np.float32),
            "conflict_matrix": np.zeros((self.max_aircraft, self.max_aircraft), dtype=np.float32),
        }
        
        # Simple mock reward
        reward = -0.01  # Small negative reward per step
        
        # Terminate after episode_length steps
        terminated = False
        truncated = self.current_step >= self.episode_length
        
        info = {
            "raw_state": {
                "aircraft": [],
                "conflicts": [],
                "score": 1000 - self.current_step,
                "time": self.current_step * 5.0,
            }
        }
        
        return obs, reward, terminated, truncated, info
    
    def close(self) -> None:
        """Close mock environment."""
        pass

