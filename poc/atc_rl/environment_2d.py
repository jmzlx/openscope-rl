"""
Simple 2D ATC environment with vectorized conflict detection.

This module provides a clean, efficient 2D air traffic control environment
for rapid prototyping and testing of RL algorithms.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, FancyArrow

import gymnasium as gym
from gymnasium import spaces

from .constants import (
    AIRSPACE_SIZE, SEPARATION_MIN, CALLSIGNS, DEFAULT_MAX_AIRCRAFT_2D,
    DEFAULT_MAX_STEPS_2D, REWARD_SUCCESSFUL_EXIT, REWARD_FAILED_EXIT,
    REWARD_COLLISION, REWARD_ACTION, REWARD_TIMESTEP_PENALTY,
    BONUS_HIGH_SUCCESS_RATE, BONUS_MEDIUM_SUCCESS_RATE, BONUS_LOW_SUCCESS_RATE,
    HIGH_SUCCESS_THRESHOLD, MEDIUM_SUCCESS_THRESHOLD, LOW_SUCCESS_THRESHOLD
)
from .physics import (
    Aircraft2D, update_aircraft_2d, check_conflicts_vectorized,
    has_exited_2d, is_near_exit_2d
)


logger = logging.getLogger(__name__)


class Simple2DATCEnv(gym.Env):
    """
    Simple 2D ATC environment with vectorized conflict detection.
    
    This environment provides a clean interface for 2D air traffic control
    with efficient vectorized operations for conflict detection.
    
    Example:
        >>> env = Simple2DATCEnv(max_aircraft=5, render_mode="human")
        >>> obs, info = env.reset()
        >>> action = env.action_space.sample()
        >>> obs, reward, terminated, truncated, info = env.step(action)
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 2}
    
    def __init__(
        self,
        max_aircraft: int = DEFAULT_MAX_AIRCRAFT_2D,
        max_steps: int = DEFAULT_MAX_STEPS_2D,
        render_mode: Optional[str] = None,
        recorder: Optional[Any] = None,
        progressive_difficulty: bool = True,
        initial_aircraft: int = 2,
    ):
        """
        Initialize 2D ATC environment.
        
        Args:
            max_aircraft: Maximum number of aircraft to track
            max_steps: Maximum steps per episode
            render_mode: Rendering mode ("human" or None)
            recorder: Optional recorder for episode tracking
            progressive_difficulty: Whether to use progressive difficulty
            initial_aircraft: Number of aircraft to start with
        """
        super().__init__()
        
        self.max_aircraft = max_aircraft
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.recorder = recorder
        self.progressive_difficulty = progressive_difficulty
        self.initial_aircraft = initial_aircraft
        
        # Environment state
        self.aircraft: List[Aircraft2D] = []
        self.current_step = 0
        self.total_reward = 0.0
        self.separations_lost = 0
        self.successful_exits = 0
        self.total_aircraft_spawned = 0
        self.collided_aircraft = 0
        self.failed_exits = 0
        self.all_aircraft_resolved = False
        
        # Rendering
        self.fig: Optional[plt.Figure] = None
        self.ax: Optional[plt.Axes] = None
        
        # Define spaces
        self.observation_space = spaces.Dict({
            "aircraft": spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(max_aircraft, 6),
                dtype=np.float32,
            ),
            "mask": spaces.Box(
                low=0,
                high=1,
                shape=(max_aircraft,),
                dtype=np.uint8,
            ),
        })
        
        self.action_space = spaces.MultiDiscrete([max_aircraft + 1, 25])
        
        logger.info(f"Simple2DATCEnv initialized: {max_aircraft} aircraft, "
                   f"{max_steps} max steps")
    
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Reset environment to initial state.
        
        Args:
            seed: Random seed
            options: Additional reset options
            
        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)
        
        # Reset state
        self.aircraft.clear()
        self.current_step = 0
        self.total_reward = 0.0
        self.separations_lost = 0
        self.successful_exits = 0
        self.total_aircraft_spawned = 0
        self.collided_aircraft = 0
        self.failed_exits = 0
        self.all_aircraft_resolved = False
        
        # Spawn initial aircraft
        if self.progressive_difficulty:
            num_initial = self.initial_aircraft
        else:
            num_initial = int(self.np_random.integers(2, min(4, self.max_aircraft) + 1))
        
        for _ in range(num_initial):
            self._spawn_aircraft()
        
        observation = self._get_observation()
        info = self._get_info()
        
        logger.debug(f"Environment reset: {len(self.aircraft)} aircraft")
        return observation, info
    
    def step(self, action: Tuple[int, int]) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Execute action and return next state.
        
        Args:
            action: Tuple of (aircraft_id, heading_change_idx)
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        aircraft_id, heading_change_idx = action
        
        # Base reward
        reward = REWARD_TIMESTEP_PENALTY
        
        # Apply action if valid
        if aircraft_id < len(self.aircraft):
            heading_change = (heading_change_idx - 12) * 15
            aircraft = self.aircraft[aircraft_id]
            aircraft.target_heading = (aircraft.heading + heading_change) % 360
            reward += REWARD_ACTION
        
        # Update all aircraft
        for ac in self.aircraft:
            update_aircraft_2d(ac)
        
        # Check for conflicts
        violations, conflicts = check_conflicts_vectorized(self.aircraft)
        if violations:
            self.separations_lost += violations
            self.collided_aircraft += violations
            reward += REWARD_COLLISION * violations
        
        # Process aircraft exits
        remaining_aircraft: List[Aircraft2D] = []
        for ac in self.aircraft:
            if has_exited_2d(ac):
                if is_near_exit_2d(ac):
                    reward += REWARD_SUCCESSFUL_EXIT
                    self.successful_exits += 1
                else:
                    reward += REWARD_FAILED_EXIT
                    self.failed_exits += 1
            else:
                remaining_aircraft.append(ac)
        self.aircraft = remaining_aircraft
        
        # Spawn new aircraft
        self._spawn_new_aircraft()
        
        self.current_step += 1
        self.total_reward += reward
        
        # Record step if recorder is provided
        if self.recorder is not None:
            env_info = self._get_info()
            env_info['airspace_size'] = AIRSPACE_SIZE
            self.recorder.record_step(self.aircraft, env_info)
        
        # Check termination
        terminated, truncated = self._check_termination()
        
        # Add episode completion bonus
        if terminated:
            reward += self._calculate_episode_bonus()
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _spawn_aircraft(self) -> None:
        """Spawn a new aircraft at the edge of the airspace."""
        if len(self.aircraft) >= self.max_aircraft:
            return
        
        entry_edge = int(self.np_random.integers(0, 4))
        exit_edge = (entry_edge + 2) % 4
        
        half = AIRSPACE_SIZE / 2
        
        if entry_edge == 0:  # North
            x = float(self.np_random.uniform(-half + 2, half - 2))
            y = half
            heading = 180
        elif entry_edge == 1:  # East
            x = half
            y = float(self.np_random.uniform(-half + 2, half - 2))
            heading = 270
        elif entry_edge == 2:  # South
            x = float(self.np_random.uniform(-half + 2, half - 2))
            y = -half
            heading = 0
        else:  # West
            x = -half
            y = float(self.np_random.uniform(-half + 2, half - 2))
            heading = 90
        
        callsign = CALLSIGNS[self.total_aircraft_spawned % len(CALLSIGNS)]
        aircraft = Aircraft2D(
            callsign=callsign,
            x=x,
            y=y,
            heading=float(heading),
            target_heading=float(heading),
            exit_edge=exit_edge,
        )
        self.aircraft.append(aircraft)
        self.total_aircraft_spawned += 1
    
    def _spawn_new_aircraft(self) -> None:
        """Spawn new aircraft based on difficulty settings."""
        if self.progressive_difficulty:
            # Gradually increase spawn rate and max aircraft
            spawn_probability = min(0.9, 0.3 + (self.current_step * 0.001))
            max_current_aircraft = min(self.max_aircraft, self.initial_aircraft + (self.current_step // 50))
            
            if (self.current_step % 10 == 0 and 
                len(self.aircraft) < max_current_aircraft and 
                self.np_random.random() < spawn_probability):
                self._spawn_aircraft()
        else:
            # Original spawning logic
            if (self.current_step % 5 == 0 and 
                len(self.aircraft) < self.max_aircraft and 
                self.np_random.random() < 0.8):
                self._spawn_aircraft()
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get current observation."""
        aircraft_obs = np.zeros((self.max_aircraft, 6), dtype=np.float32)
        mask = np.zeros(self.max_aircraft, dtype=np.uint8)
        
        half = AIRSPACE_SIZE / 2
        
        for i, ac in enumerate(self.aircraft[:self.max_aircraft]):
            if ac.exit_edge == 0:  # North
                exit_x, exit_y = ac.x, half
            elif ac.exit_edge == 1:  # East
                exit_x, exit_y = half, ac.y
            elif ac.exit_edge == 2:  # South
                exit_x, exit_y = ac.x, -half
            else:  # West
                exit_x, exit_y = -half, ac.y
            
            dx_to_exit = exit_x - ac.x
            dy_to_exit = exit_y - ac.y
            
            aircraft_obs[i] = [
                ac.x / half,
                ac.y / half,
                np.sin(np.radians(ac.heading)),
                np.cos(np.radians(ac.heading)),
                dx_to_exit / AIRSPACE_SIZE,
                dy_to_exit / AIRSPACE_SIZE,
            ]
            mask[i] = 1
        
        return {"aircraft": aircraft_obs, "mask": mask}
    
    def _get_info(self) -> Dict[str, Any]:
        """Get environment information."""
        return {
            "step": self.current_step,
            "num_aircraft": len(self.aircraft),
            "total_reward": self.total_reward,
            "separations_lost": self.separations_lost,
            "successful_exits": self.successful_exits,
            "total_aircraft_spawned": self.total_aircraft_spawned,
            "collided_aircraft": self.collided_aircraft,
            "failed_exits": self.failed_exits,
            "all_aircraft_resolved": self.all_aircraft_resolved,
            "success_rate": self.successful_exits / max(self.total_aircraft_spawned, 1),
        }
    
    def _check_termination(self) -> Tuple[bool, bool]:
        """Check if episode should terminate."""
        # Check if all aircraft have been resolved
        total_resolved = self.successful_exits + self.failed_exits + self.collided_aircraft
        max_spawn_steps = 200
        
        if (len(self.aircraft) == 0 and 
            (self.current_step > max_spawn_steps or total_resolved >= self.total_aircraft_spawned)):
            self.all_aircraft_resolved = True
            return True, False
        
        # Fallback termination
        terminated = self.current_step >= self.max_steps
        return terminated, False
    
    def _calculate_episode_bonus(self) -> float:
        """Calculate episode completion bonus."""
        success_rate = self.successful_exits / max(self.total_aircraft_spawned, 1)
        
        if success_rate >= HIGH_SUCCESS_THRESHOLD:
            return BONUS_HIGH_SUCCESS_RATE
        elif success_rate >= MEDIUM_SUCCESS_THRESHOLD:
            return BONUS_MEDIUM_SUCCESS_RATE
        elif success_rate >= LOW_SUCCESS_THRESHOLD:
            return BONUS_LOW_SUCCESS_RATE
        else:
            return 0.0
    
    def render(self) -> None:
        """Render the environment."""
        if self.render_mode != "human":
            return
        
        if self.fig is None:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(8, 8))
        
        assert self.ax is not None
        self.ax.clear()
        
        half = AIRSPACE_SIZE / 2
        self.ax.add_patch(
            mpatches.Rectangle(
                (-half, -half),
                AIRSPACE_SIZE,
                AIRSPACE_SIZE,
                fill=False,
                edgecolor="gray",
                linewidth=2,
            )
        )
        
        for ac in self.aircraft:
            circle = Circle(
                (ac.x, ac.y),
                SEPARATION_MIN,
                fill=False,
                edgecolor="orange",
                linestyle="--",
                alpha=0.3,
            )
            self.ax.add_patch(circle)
            
            heading_rad = np.radians(ac.heading)
            dx = 1.5 * np.sin(heading_rad)
            dy = 1.5 * np.cos(heading_rad)
            arrow = FancyArrow(
                ac.x,
                ac.y,
                dx,
                dy,
                width=0.5,
                head_width=1.0,
                head_length=0.8,
                color="blue",
                alpha=0.7,
            )
            self.ax.add_patch(arrow)
            
            self.ax.text(
                ac.x,
                ac.y + 1.2,
                ac.callsign,
                ha="center",
                va="bottom",
                fontsize=8,
                color="blue",
            )
        
        info_text = (
            f"Step: {self.current_step}/{self.max_steps}\n"
            f"Aircraft: {len(self.aircraft)}\n"
            f"Exits: {self.successful_exits}\n"
            f"Violations: {self.separations_lost}\n"
            f"Reward: {self.total_reward:.1f}"
        )
        self.ax.text(
            -half + 1,
            half - 1,
            info_text,
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )
        
        self.ax.set_xlim(-half - 1, half + 1)
        self.ax.set_ylim(-half - 1, half + 1)
        self.ax.set_aspect("equal")
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel("X (nautical miles)")
        self.ax.set_ylabel("Y (nautical miles)")
        self.ax.set_title("Simple 2D ATC Environment")
        
        plt.pause(0.01)
    
    def close(self) -> None:
        """Clean up resources."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
