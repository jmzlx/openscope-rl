"""
Realistic 3D ATC environment with full physics.

This module provides a comprehensive 3D air traffic control environment
with realistic physics, altitude management, and runway operations.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

import gymnasium as gym
from gymnasium import spaces

from .constants import (
    AIRSPACE_SIZE, SEPARATION_LATERAL_NM, SEPARATION_VERTICAL_FT,
    CONFLICT_WARNING_BUFFER_NM, CALLSIGNS, RUNWAYS,
    DEFAULT_MAX_AIRCRAFT_3D, DEFAULT_EPISODE_LENGTH_3D, DEFAULT_SPAWN_INTERVAL_3D
)
from .physics import (
    Aircraft3D, update_aircraft_3d, check_conflicts_vectorized_3d,
    has_exited_3d, is_ready_for_landing, calculate_landing_distance
)


logger = logging.getLogger(__name__)


class Realistic3DATCEnv(gym.Env):
    """
    Realistic 3D ATC environment with full physics.
    
    This environment provides comprehensive 3D air traffic control with
    realistic physics, altitude management, and runway operations.
    
    Example:
        >>> env = Realistic3DATCEnv(max_aircraft=8, render_mode="human")
        >>> obs, info = env.reset()
        >>> action = env.action_space.sample()
        >>> obs, reward, terminated, truncated, info = env.step(action)
    """
    
    metadata = {'render_modes': ['human'], 'render_fps': 2}
    
    def __init__(
        self,
        max_aircraft: int = DEFAULT_MAX_AIRCRAFT_3D,
        episode_length: int = DEFAULT_EPISODE_LENGTH_3D,
        spawn_interval: float = DEFAULT_SPAWN_INTERVAL_3D,
        render_mode: Optional[str] = None,
        recorder: Optional[Any] = None,
    ):
        """
        Initialize 3D ATC environment.
        
        Args:
            max_aircraft: Maximum number of aircraft to track
            episode_length: Episode length in seconds
            spawn_interval: Aircraft spawn interval in seconds
            render_mode: Rendering mode ("human" or None)
            recorder: Optional recorder for episode tracking
        """
        super().__init__()
        
        self.max_aircraft = max_aircraft
        self.episode_length = episode_length
        self.spawn_interval = spawn_interval
        self.render_mode = render_mode
        self.recorder = recorder
        
        # Runway configuration
        self.runways = RUNWAYS
        
        # Environment state
        self.aircraft: List[Aircraft3D] = []
        self.time_elapsed = 0.0
        self.last_spawn_time = 0.0
        self.score = 0
        self.violations = 0
        self.conflicts = 0
        self.successful_landings = 0
        
        # Rendering
        self.fig = None
        self.ax = None
        
        # Define spaces
        self.observation_space = spaces.Dict({
            'aircraft': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(max_aircraft, 14),
                dtype=np.float32
            ),
            'aircraft_mask': spaces.Box(
                low=0, high=1,
                shape=(max_aircraft,),
                dtype=np.uint8
            ),
            'conflict_matrix': spaces.Box(
                low=0.0, high=1.0,
                shape=(max_aircraft, max_aircraft),
                dtype=np.float32
            ),
            'global_state': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(4,),
                dtype=np.float32
            )
        })
        
        # Action space (MultiDiscrete for easier SB3 compatibility)
        self.action_space = spaces.MultiDiscrete([
            max_aircraft + 1,  # aircraft_id
            5,  # command_type
            18,  # altitude (0-17k ft in 1k increments)
            12,  # heading (0-330 in 30° increments)
            8,   # speed (150-360 knots in 30kt increments)
        ])
        
        logger.info(f"Realistic3DATCEnv initialized: {max_aircraft} aircraft, "
                   f"{episode_length}s episodes")
    
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
        self.aircraft = []
        self.time_elapsed = 0.0
        self.last_spawn_time = 0.0
        self.score = 0
        self.violations = 0
        self.conflicts = 0
        self.successful_landings = 0
        
        # Spawn initial aircraft
        for _ in range(min(3, self.max_aircraft)):
            self._spawn_aircraft()
        
        obs = self._get_observation()
        info = self._get_info()
        
        logger.debug(f"Environment reset: {len(self.aircraft)} aircraft")
        return obs, info
    
    def step(self, action: Tuple[int, int, int, int, int]) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Execute action and return next state.
        
        Args:
            action: Tuple of (aircraft_id, command_type, altitude_idx, heading_idx, speed_idx)
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        dt = 1.0  # 1 second per step
        
        # Unpack action
        aircraft_id, command_type, altitude_idx, heading_idx, speed_idx = action
        
        # Execute action
        reward = self._execute_action(aircraft_id, command_type, 
                                    altitude_idx, heading_idx, speed_idx)
        
        # Update all aircraft
        for aircraft in self.aircraft:
            update_aircraft_3d(aircraft, dt)
        
        # Check for conflicts/violations
        conflict_penalty, violation_penalty = self._check_conflicts()
        reward += conflict_penalty + violation_penalty
        
        # Remove aircraft that left the area
        self.aircraft = [
            ac for ac in self.aircraft
            if not has_exited_3d(ac)
        ]
        
        # Check for successful landings
        landing_reward = self._check_landings()
        reward += landing_reward
        
        # Spawn new aircraft periodically
        self.time_elapsed += dt
        if self.time_elapsed - self.last_spawn_time >= self.spawn_interval:
            self._spawn_aircraft()
            self.last_spawn_time = self.time_elapsed
        
        # Update score
        self.score += reward
        
        # Record step if recorder is provided
        if self.recorder is not None:
            env_info = self._get_info()
            env_info['airspace_size'] = AIRSPACE_SIZE
            env_info['runways'] = self.runways
            self.recorder.record_step(self.aircraft, env_info)
        
        # Check termination
        terminated = self.time_elapsed >= self.episode_length
        truncated = False
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, reward, terminated, truncated, info
    
    def _spawn_aircraft(self) -> None:
        """Spawn a new aircraft at the edge of the simulation area."""
        if len(self.aircraft) >= self.max_aircraft:
            return
        
        # Random spawn position at edge
        edge = self.np_random.integers(0, 4)  # 0=North, 1=East, 2=South, 3=West
        
        if edge == 0:  # North
            x = self.np_random.uniform(-AIRSPACE_SIZE/2, AIRSPACE_SIZE/2)
            y = AIRSPACE_SIZE / 2
            heading = 180  # South
        elif edge == 1:  # East
            x = AIRSPACE_SIZE / 2
            y = self.np_random.uniform(-AIRSPACE_SIZE/2, AIRSPACE_SIZE/2)
            heading = 270  # West
        elif edge == 2:  # South
            x = self.np_random.uniform(-AIRSPACE_SIZE/2, AIRSPACE_SIZE/2)
            y = -AIRSPACE_SIZE / 2
            heading = 0  # North
        else:  # West
            x = -AIRSPACE_SIZE / 2
            y = self.np_random.uniform(-AIRSPACE_SIZE/2, AIRSPACE_SIZE/2)
            heading = 90  # East
        
        altitude = self.np_random.uniform(3000, 10000)
        speed = self.np_random.uniform(200, 280)
        
        callsign = CALLSIGNS[len(self.aircraft) % len(CALLSIGNS)]
        
        aircraft = Aircraft3D(
            callsign=callsign,
            x=x, y=y,
            altitude=altitude,
            heading=heading,
            speed=speed,
        )
        
        self.aircraft.append(aircraft)
    
    def _execute_action(self, aircraft_id: int, command_type: int, 
                        altitude_idx: int, heading_idx: int, speed_idx: int) -> float:
        """Execute the commanded action and return immediate reward."""
        # No-op
        if aircraft_id >= len(self.aircraft) or command_type == 4:
            return 0.0
        
        aircraft = self.aircraft[aircraft_id]
        reward = 0.0
        
        if command_type == 0:  # Altitude
            altitude_ft = altitude_idx * 1000
            aircraft.target_altitude = altitude_ft
            reward = 0.1
        
        elif command_type == 1:  # Heading
            heading_deg = heading_idx * 30
            aircraft.target_heading = heading_deg
            reward = 0.1
        
        elif command_type == 2:  # Speed
            speed_kts = 150 + speed_idx * 30
            aircraft.target_speed = speed_kts
            reward = 0.1
        
        elif command_type == 3:  # Land
            # Assign to nearest runway
            best_runway = None
            best_dist = float('inf')
            for i, runway in enumerate(self.runways):
                dist = np.sqrt((aircraft.x - runway['x2'])**2 + (aircraft.y - runway['y2'])**2)
                if dist < best_dist:
                    best_dist = dist
                    best_runway = i
            
            aircraft.is_landing = True
            aircraft.runway_assigned = best_runway
            aircraft.target_altitude = 0
            aircraft.target_speed = 150
            reward = 0.5
        
        return reward
    
    def _check_conflicts(self) -> Tuple[float, float]:
        """Check for conflicts and violations between aircraft."""
        conflict_penalty = 0.0
        violation_penalty = 0.0
        
        violations, conflicts = check_conflicts_vectorized_3d(self.aircraft)
        
        if violations:
            violation_penalty -= 10.0 * violations
            self.violations += violations
        
        if conflicts:
            conflict_penalty -= 1.0 * conflicts
            self.conflicts += conflicts
        
        return conflict_penalty, violation_penalty
    
    def _check_landings(self) -> float:
        """Check for successful landings and remove landed aircraft."""
        reward = 0.0
        aircraft_to_remove = []
        
        for ac in self.aircraft:
            if is_ready_for_landing(ac):
                # Check if close to runway
                if ac.runway_assigned is not None:
                    runway = self.runways[ac.runway_assigned]
                    dist_to_runway = calculate_landing_distance(ac, runway)
                    
                    if dist_to_runway < 1.0:  # Within 1nm of runway end
                        reward += 20.0
                        self.successful_landings += 1
                        aircraft_to_remove.append(ac)
        
        for ac in aircraft_to_remove:
            self.aircraft.remove(ac)
        
        return reward
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get current observation."""
        aircraft_features = np.zeros((self.max_aircraft, 14), dtype=np.float32)
        aircraft_mask = np.zeros(self.max_aircraft, dtype=np.uint8)
        conflict_matrix = np.zeros((self.max_aircraft, self.max_aircraft), dtype=np.float32)
        
        for i, ac in enumerate(self.aircraft):
            if i >= self.max_aircraft:
                break
            
            # Calculate distance and bearing to airport
            distance = np.sqrt(ac.x**2 + ac.y**2)
            bearing = np.degrees(np.arctan2(ac.x, ac.y)) % 360
            
            # Find nearest runway
            nearest_runway_dx = 0
            nearest_runway_dy = 0
            if ac.runway_assigned is not None:
                runway = self.runways[ac.runway_assigned]
                nearest_runway_dx = runway['x2'] - ac.x
                nearest_runway_dy = runway['y2'] - ac.y
            
            aircraft_features[i] = [
                ac.x / AIRSPACE_SIZE,
                ac.y / AIRSPACE_SIZE,
                ac.altitude / 10000.0,
                ac.heading / 360.0,
                ac.speed / 300.0,
                ac.target_altitude / 10000.0 if ac.target_altitude is not None else 0.0,
                ac.target_heading / 360.0 if ac.target_heading is not None else 0.0,
                ac.target_speed / 300.0 if ac.target_speed is not None else 0.0,
                nearest_runway_dx / AIRSPACE_SIZE,
                nearest_runway_dy / AIRSPACE_SIZE,
                float(ac.runway_assigned if ac.runway_assigned is not None else -1) / 4.0,
                float(ac.is_landing),
                distance / AIRSPACE_SIZE,
                bearing / 360.0,
            ]
            aircraft_mask[i] = 1
        
        # Compute conflict matrix
        for i, ac1 in enumerate(self.aircraft):
            for j, ac2 in enumerate(self.aircraft):
                if i != j and i < self.max_aircraft and j < self.max_aircraft:
                    lateral_dist = np.sqrt((ac1.x - ac2.x)**2 + (ac1.y - ac2.y)**2)
                    vertical_sep = abs(ac1.altitude - ac2.altitude)
                    
                    if vertical_sep < SEPARATION_VERTICAL_FT:
                        if lateral_dist < SEPARATION_LATERAL_NM:
                            conflict_matrix[i, j] = 1.0
                        elif lateral_dist < SEPARATION_LATERAL_NM + CONFLICT_WARNING_BUFFER_NM:
                            conflict_matrix[i, j] = 0.5
        
        global_state = np.array([
            len(self.aircraft) / self.max_aircraft,
            self.time_elapsed / self.episode_length,
            self.score / 100.0,
            self.violations / 10.0,
        ], dtype=np.float32)
        
        return {
            'aircraft': aircraft_features,
            'aircraft_mask': aircraft_mask,
            'conflict_matrix': conflict_matrix,
            'global_state': global_state,
        }
    
    def _get_info(self) -> Dict[str, Any]:
        """Get environment information."""
        return {
            'time_elapsed': self.time_elapsed,
            'num_aircraft': len(self.aircraft),
            'score': self.score,
            'violations': self.violations,
            'conflicts': self.conflicts,
            'successful_landings': self.successful_landings,
        }
    
    def render(self) -> None:
        """Render the environment."""
        if self.render_mode != 'human':
            return
        
        if self.fig is None:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(10, 10))
        
        self.ax.clear()
        self.ax.set_xlim(-AIRSPACE_SIZE/2, AIRSPACE_SIZE/2)
        self.ax.set_ylim(-AIRSPACE_SIZE/2, AIRSPACE_SIZE/2)
        self.ax.set_aspect('equal')
        self.ax.set_facecolor('black')
        self.fig.patch.set_facecolor('black')
        
        # Draw runways (white lines)
        for runway in self.runways:
            self.ax.plot(
                [runway['x1'], runway['x2']],
                [runway['y1'], runway['y2']],
                color='white',
                linewidth=3,
                label=runway['name']
            )
        
        # Draw aircraft (yellow triangles)
        for ac in self.aircraft:
            heading_rad = np.radians(ac.heading)
            
            # Triangle vertices (pointing up by default)
            size = 0.5
            vertices = np.array([
                [0, size],
                [-size/2, -size/2],
                [size/2, -size/2],
            ])
            
            # Rotate by heading
            cos_h = np.cos(heading_rad)
            sin_h = np.sin(heading_rad)
            rotation_matrix = np.array([
                [sin_h, cos_h],
                [cos_h, -sin_h]
            ])
            rotated = vertices @ rotation_matrix.T
            
            # Translate to aircraft position
            rotated[:, 0] += ac.x
            rotated[:, 1] += ac.y
            
            triangle = Polygon(
                rotated,
                closed=True,
                facecolor='yellow',
                edgecolor='orange',
                linewidth=1
            )
            self.ax.add_patch(triangle)
            
            # Label with callsign and altitude
            self.ax.text(
                ac.x, ac.y + 0.8,
                f"{ac.callsign}\n{int(ac.altitude)}ft",
                color='white',
                fontsize=8,
                ha='center',
                va='bottom'
            )
        
        # Add info text
        info_text = (
            f"Time: {self.time_elapsed:.1f}s\n"
            f"Aircraft: {len(self.aircraft)}\n"
            f"Score: {self.score:.1f}\n"
            f"Violations: {self.violations}\n"
            f"Landings: {self.successful_landings}"
        )
        self.ax.text(
            -AIRSPACE_SIZE/2 + 1, AIRSPACE_SIZE/2 - 1,
            info_text,
            color='white',
            fontsize=10,
            va='top',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7)
        )
        
        self.ax.grid(True, color='gray', alpha=0.3)
        self.ax.set_xlabel('X (nautical miles)', color='white')
        self.ax.set_ylabel('Y (nautical miles)', color='white')
        self.ax.tick_params(colors='white')
        
        plt.tight_layout()
        plt.pause(0.01)
    
    def close(self) -> None:
        """Clean up resources."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
