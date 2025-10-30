"""
Refactored OpenScope Environment using Playwright.

This module provides a clean, modular implementation of the OpenScope environment
using separated concerns for browser management, game interface, state processing,
and reward calculation.
"""

import logging
import time
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np

from .config import OpenScopeConfig, create_default_config, validate_config
from .utils import BrowserManager
from .game_interface import OpenScopeInterface
from .state_processor import StateProcessor
from .reward_calculator import RewardCalculator, create_reward_calculator
from .spaces import create_observation_space, create_action_space, validate_observation, validate_action
from .metrics import EpisodeMetrics
from .exceptions import OpenScopeError, BrowserError, GameInterfaceError, StateProcessingError


logger = logging.getLogger(__name__)


class PlaywrightEnv(gym.Env):
    """
    OpenScope environment using Playwright for browser automation.
    
    This environment provides a clean interface to the OpenScope air traffic
    control simulator using separated concerns for better maintainability.
    
    Example:
        >>> env = PlaywrightEnv(
        ...     airport="KJFK",
        ...     max_aircraft=10,
        ...     headless=False
        ... )
        >>> obs, info = env.reset()
        >>> action = env.action_space.sample()
        >>> obs, reward, terminated, truncated, info = env.step(action)
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 1}
    
    def __init__(
        self,
        game_url: str = "http://localhost:3003",
        airport: str = "KLAS",
        timewarp: int = 5,
        max_aircraft: int = 20,
        episode_length: int = 3600,
        action_interval: float = 5.0,
        headless: bool = True,
        config: Optional[Dict[str, Any]] = None,
        reward_strategy: str = "default",
    ):
        """
        Initialize OpenScope environment.
        
        Args:
            game_url: URL of the OpenScope game
            airport: ICAO airport code
            timewarp: Time acceleration factor
            max_aircraft: Maximum number of aircraft to track
            episode_length: Maximum episode length in seconds
            action_interval: Time between actions in seconds
            headless: Whether to run browser in headless mode
            config: Additional configuration dictionary
            reward_strategy: Reward calculation strategy ("default", "safety", "efficiency")
            
        Raises:
            OpenScopeError: If environment initialization fails
        """
        super().__init__()
        
        try:
            # Create configuration
            self.config = create_default_config(
                game_url=game_url,
                airport=airport,
                timewarp=timewarp,
                max_aircraft=max_aircraft,
                episode_length=episode_length,
                action_interval=action_interval,
                headless=headless,
                **config or {}
            )
            
            # Validate configuration
            validate_config(self.config)
            
            # Initialize components
            self._init_components(reward_strategy)
            
            # Initialize episode state
            self._init_episode_state()
            
            logger.info(f"PlaywrightEnv initialized: {airport}, {max_aircraft} aircraft, "
                       f"{reward_strategy} reward strategy")
            
        except Exception as e:
            logger.error(f"Failed to initialize PlaywrightEnv: {e}")
            raise OpenScopeError(f"Environment initialization failed: {e}") from e
    
    def _init_components(self, reward_strategy: str) -> None:
        """Initialize environment components."""
        # Browser management
        self.browser_manager = BrowserManager(self.config.browser_config)
        
        # Game interface
        self.game_interface = OpenScopeInterface(self.config)
        
        # State processing
        self.state_processor = StateProcessor(self.config)
        
        # Reward calculation
        self.reward_calculator = create_reward_calculator(
            self.config.reward_config, reward_strategy
        )
        
        # Episode metrics
        self.episode_metrics = EpisodeMetrics()
        
        # Define spaces
        self.observation_space = create_observation_space(self.config.max_aircraft)
        self.action_space = create_action_space(self.config.max_aircraft)
    
    def _init_episode_state(self) -> None:
        """Initialize episode state variables."""
        self.current_step = 0
        self.simulated_time = 0.0
        self.prev_state: Dict[str, Any] = {}
        self.step_start_time: Optional[float] = None
        self._is_initialized = False
    
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Reset environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional reset options
            
        Returns:
            Tuple of (observation, info)
            
        Raises:
            OpenScopeError: If reset fails
        """
        super().reset(seed=seed)
        
        try:
            logger.info("Resetting environment")
            
            # Initialize browser if needed
            if not self._is_initialized:
                self._initialize_browser()
            
            # Reset game
            self.game_interface.reset_game()
            
            # Reset episode state
            self._init_episode_state()
            self.episode_metrics.start_episode()
            self.reward_calculator.reset_episode()
            
            # Get initial state
            self.prev_state = self.game_interface.get_game_state()
            observation = self.state_processor.process_state(self.prev_state)
            
            # Validate observation
            validate_observation(observation, self.config.max_aircraft)
            
            info = {
                "raw_state": self.prev_state,
                "episode_metrics": self.episode_metrics.to_dict(),
                "config": self.config.__dict__,
            }
            
            logger.info("Environment reset completed")
            return observation, info
            
        except Exception as e:
            logger.error(f"Failed to reset environment: {e}")
            raise OpenScopeError(f"Environment reset failed: {e}") from e
    
    def step(
        self, 
        action: Dict[str, int]
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Execute action and return next state.
        
        Args:
            action: Action dictionary with aircraft_id, command_type, etc.
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
            
        Raises:
            OpenScopeError: If step execution fails
        """
        try:
            # Start timing this step
            self.step_start_time = time.time()
            
            # Validate action
            validate_action(action, self.config.max_aircraft)
            
            # Execute action
            command = self.game_interface.execute_action(action, self.prev_state.get("aircraft", []))
            
            # Record command metrics
            if command:
                command_type = self.config.action_config.command_types[action["command_type"]]
                self.episode_metrics.record_command(command_type.value)
            
            # Wait for game update
            self.game_interface.wait_for_update(1)
            self.simulated_time += self.config.action_interval
            
            # Get new state
            state = self.game_interface.get_game_state()
            observation = self.state_processor.process_state(state)
            
            # Calculate reward
            reward = self.reward_calculator.calculate_reward(state, self.prev_state, action)
            
            # Update episode metrics
            self._update_episode_metrics(state, reward)
            
            # Check termination conditions
            terminated, truncated = self._check_termination(state)
            
            # Add episode termination penalty
            if terminated:
                reward += self.config.reward_config.episode_termination_penalty
            
            # Calculate step time
            step_time = time.time() - self.step_start_time if self.step_start_time else 0
            
            # Prepare info
            info = self._prepare_info(state, command, action, step_time)
            
            # Update previous state
            self.prev_state = state
            
            logger.debug(f"Step {self.current_step}: reward={reward:.3f}, "
                        f"aircraft={len(state.get('aircraft', []))}, "
                        f"conflicts={len(state.get('conflicts', []))}")
            
            return observation, reward, terminated, truncated, info
            
        except Exception as e:
            logger.error(f"Failed to execute step: {e}")
            raise OpenScopeError(f"Step execution failed: {e}") from e
    
    def _initialize_browser(self) -> None:
        """Initialize browser and game interface."""
        try:
            logger.info("Initializing browser")
            
            # Initialize browser manager
            self.browser_manager.initialize()
            
            # Inject time tracking script
            self.browser_manager.inject_time_tracking_script()
            # Inject scoring event hook script
            self.browser_manager.inject_event_hook_script()
            
            # Navigate to game
            self.browser_manager.navigate_to_game(self.config.game_url)
            
            # Initialize game interface
            self.game_interface.initialize(self.browser_manager.page)
            
            # Wait for game to be ready
            max_retries = 60  # Increased to 60 (2 minutes total)
            retry_delay = 1.0  # Check every second
            for i in range(max_retries):
                if self.game_interface.is_game_ready():
                    logger.info(f"Game ready after {i+1} attempts")
                    break

                # Log progress every 5 attempts
                if (i + 1) % 5 == 0:
                    game_info = self.game_interface.get_game_info()
                    logger.info(f"Waiting for game ({i+1}/{max_retries}): {game_info}")

                time.sleep(retry_delay)
            else:
                # Try to get more information about why the game isn't ready
                game_info = self.game_interface.get_game_info()
                logger.error(f"Game readiness check failed. Game info: {game_info}")
                raise GameInterfaceError("Game failed to become ready after extended wait")
            
            self._is_initialized = True
            logger.info("Browser initialization completed")
            
        except Exception as e:
            logger.error(f"Failed to initialize browser: {e}")
            self.cleanup()
            raise BrowserError(f"Browser initialization failed: {e}") from e
    
    def _update_episode_metrics(self, state: Dict[str, Any], reward: float) -> None:
        """Update episode metrics with current state."""
        # Update basic metrics
        self.episode_metrics.add_reward(reward)
        self.episode_metrics.increment_step()
        
        # Update aircraft count
        aircraft_count = len(state.get("aircraft", []))
        self.episode_metrics.update_aircraft_count(aircraft_count)
        
        # Update conflicts
        conflicts = state.get("conflicts", [])
        for conflict in conflicts:
            self.episode_metrics.record_conflict(conflict.get("hasViolation", False))
        
        # Update step counter
        self.current_step += 1
    
    def _check_termination(self, state: Dict[str, Any]) -> Tuple[bool, bool]:
        """
        Check if episode should terminate.
        
        Args:
            state: Current game state
            
        Returns:
            Tuple of (terminated, truncated)
        """
        # Check for termination (low score)
        score = state.get("score", 0)
        terminated = score < self.config.custom_config.get("min_score_threshold", -2000)
        
        # Check for truncation (time limit)
        truncated = (
            state.get("time", 0) >= self.config.episode_length or 
            self.current_step >= self.config.custom_config.get("max_steps", 1000)
        )
        
        return terminated, truncated
    
    def _prepare_info(self, state: Dict[str, Any], command: Optional[str], 
                     action: Dict[str, int], step_time: float) -> Dict[str, Any]:
        """Prepare info dictionary for step return."""
        return {
            "raw_state": state,
            "score": state.get("score", 0),
            "aircraft_count": len(state.get("aircraft", [])),
            "conflict_count": len(state.get("conflicts", [])),
            "episode_metrics": self.episode_metrics.to_dict(),
            "command_issued": command,
            "action": action,
            "step_time": step_time,
            "simulated_time": self.simulated_time,
            "current_step": self.current_step,
        }
    
    def render(self) -> None:
        """
        Render the environment.
        
        Note: The game renders itself in the browser window.
        This method is provided for gymnasium compatibility.
        """
        pass
    
    def close(self) -> None:
        """Clean up environment resources."""
        try:
            logger.info("Closing environment")
            
            if hasattr(self, 'browser_manager'):
                self.browser_manager.cleanup()
            
            self._is_initialized = False
            logger.info("Environment closed")
            
        except Exception as e:
            logger.error(f"Error during environment cleanup: {e}")
    
    def cleanup(self) -> None:
        """Alias for close() for compatibility."""
        self.close()
    
    def get_episode_metrics(self) -> Dict[str, Any]:
        """Get current episode metrics."""
        return self.episode_metrics.to_dict()
    
    def get_reward_info(self) -> Dict[str, Any]:
        """Get reward calculation information."""
        return self.reward_calculator.get_reward_info()
    
    def set_reward_strategy(self, strategy_name: str) -> None:
        """
        Change reward calculation strategy.
        
        Args:
            strategy_name: Name of strategy ("default", "safety", "efficiency")
        """
        self.reward_calculator = create_reward_calculator(
            self.config.reward_config, strategy_name
        )
        logger.info(f"Reward strategy changed to: {strategy_name}")
    
    def get_config(self) -> OpenScopeConfig:
        """Get environment configuration."""
        return self.config
    
    @property
    def is_initialized(self) -> bool:
        """Check if environment is initialized."""
        return self._is_initialized and self.browser_manager.is_initialized
