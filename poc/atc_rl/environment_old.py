"""
Optimized Simple 2D ATC environment with vectorized conflict detection.
This module mirrors the original notebook environment but packages it so the
optimized notebook can import and reuse the logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, FancyArrow

import gymnasium as gym
from gymnasium import spaces

# Constants shared with the notebooks
AIRSPACE_SIZE = 20.0  # 20nm x 20nm
SEPARATION_MIN = 3.0  # Minimum separation (nm)
CONFLICT_BUFFER = 1.0  # Warning buffer
AIRCRAFT_SPEED = 4.0  # 4 nm / step
TURN_RATE = 15.0  # 15 degrees / step

CALLSIGNS = [
    "AAL123",
    "UAL456",
    "DAL789",
    "SWA101",
    "JBU202",
    "FFT303",
    "SKW404",
    "ASA505",
    "NKS606",
    "FFT707",
]


@dataclass
class Aircraft:
    """Simple 2D aircraft with position, heading, and target."""

    callsign: str
    x: float
    y: float
    heading: float
    target_heading: float
    exit_edge: int

    def update(self, dt: float = 1.0) -> None:
        """Update heading and position."""
        heading_diff = (self.target_heading - self.heading + 180) % 360 - 180
        turn_amount = np.clip(heading_diff, -TURN_RATE, TURN_RATE)
        self.heading = (self.heading + turn_amount) % 360

        heading_rad = np.radians(self.heading)
        self.x += AIRCRAFT_SPEED * np.sin(heading_rad) * dt
        self.y += AIRCRAFT_SPEED * np.cos(heading_rad) * dt

    def distance_to(self, other: "Aircraft") -> float:
        """Euclidean distance to another aircraft."""
        return float(np.hypot(self.x - other.x, self.y - other.y))

    def has_exited(self) -> bool:
        """Return True if the aircraft left the airspace."""
        half = AIRSPACE_SIZE / 2
        return abs(self.x) > half or abs(self.y) > half

    def is_near_exit(self) -> bool:
        """Return True if the aircraft is exiting via the correct edge."""
        half = AIRSPACE_SIZE / 2
        heading = self.heading % 360

        if self.exit_edge == 0:  # North
            return self.y > half - 2 and (heading <= 45 or heading >= 315)
        if self.exit_edge == 1:  # East
            return self.x > half - 2 and 45 <= heading <= 135
        if self.exit_edge == 2:  # South
            return self.y < -half + 2 and 135 <= heading <= 225
        # West
        return self.x < -half + 2 and 225 <= heading <= 315


def _pairwise_distances(positions: np.ndarray) -> np.ndarray:
    """Compute pairwise distances with broadcasting."""
    deltas = positions[:, None, :] - positions[None, :, :]
    return np.sqrt(np.sum(deltas**2, axis=-1))


def check_conflicts_vectorized(aircraft_list: List[Aircraft]) -> Tuple[int, int]:
    """
    Efficiently compute conflict and separation violation counts.

    Returns
    -------
    violations : int
        Number of pairs violating minimum separation.
    conflicts : int
        Number of pairs inside the conflict buffer but outside violation range.
    """
    n = len(aircraft_list)
    if n < 2:
        return 0, 0

    positions = np.array([[ac.x, ac.y] for ac in aircraft_list], dtype=float)
    distances = _pairwise_distances(positions)

    # Only consider upper triangle to avoid double counting
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    separation_mask = distances < SEPARATION_MIN
    buffer_mask = distances < (SEPARATION_MIN + CONFLICT_BUFFER)

    violation_pairs = separation_mask & mask
    conflict_pairs = buffer_mask & mask & ~separation_mask

    violation_count = int(np.count_nonzero(violation_pairs))
    conflict_count = int(np.count_nonzero(conflict_pairs))
    return violation_count, conflict_count


class Simple2DATCEnv(gym.Env):
    """
    Simple 2D ATC environment with vectorized conflict detection.

    The observation, action, reward, and rendering logic follow the original
    notebook implementation so behaviour remains familiar, but conflict checks
    use numpy broadcasting for a sizeable speedup.
    """

    metadata = {"render_modes": ["human"], "render_fps": 2}

    def __init__(
        self,
        max_aircraft: int = 5,
        max_steps: int = 1000,  # Increased as fallback only
        render_mode: Optional[str] = None,
        recorder: Optional[Any] = None,
        progressive_difficulty: bool = True,
        initial_aircraft: int = 2,  # Start with fewer aircraft
    ):
        super().__init__()

        self.max_aircraft = max_aircraft
        self.max_steps = max_steps  # Fallback only - episodes should end on win/lose
        self.render_mode = render_mode
        self.recorder = recorder
        self.progressive_difficulty = progressive_difficulty
        self.initial_aircraft = initial_aircraft

        self.aircraft: List[Aircraft] = []
        self.current_step = 0
        self.total_reward = 0.0
        self.separations_lost = 0
        self.successful_exits = 0
        self.total_aircraft_spawned = 0
        self.collided_aircraft = 0
        self.failed_exits = 0
        self.all_aircraft_resolved = False

        self.fig: Optional[plt.Figure] = None
        self.ax: Optional[plt.Axes] = None

        self.observation_space = spaces.Dict(
            {
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
            }
        )

        self.action_space = spaces.MultiDiscrete([max_aircraft + 1, 25])

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ):
        super().reset(seed=seed)
        self.aircraft.clear()
        self.current_step = 0
        self.total_reward = 0.0
        self.separations_lost = 0
        self.successful_exits = 0
        self.total_aircraft_spawned = 0
        self.collided_aircraft = 0
        self.failed_exits = 0
        self.all_aircraft_resolved = False

        # Progressive difficulty: Start with fewer aircraft
        if self.progressive_difficulty:
            num_initial = self.initial_aircraft
        else:
            num_initial = int(self.np_random.integers(2, min(4, self.max_aircraft) + 1))
        
        for _ in range(num_initial):
            self._spawn_aircraft()

        return self._get_observation(), self._get_info()

    def step(self, action):
        aircraft_id, heading_change_idx = action

        # Base reward - minimal time penalty since episodes run until resolution
        reward = -0.001

        # Apply action if valid
        if aircraft_id < len(self.aircraft):
            heading_change = (heading_change_idx - 12) * 15
            aircraft = self.aircraft[aircraft_id]
            aircraft.target_heading = (aircraft.heading + heading_change) % 360
            
            # Small positive reward for taking action
            reward += 0.05

        # Update all aircraft
        for ac in self.aircraft:
            ac.update()

        # Check for conflicts - track collisions but don't end episode immediately
        violations, conflicts = check_conflicts_vectorized(self.aircraft)
        if violations:
            self.separations_lost += violations
            self.collided_aircraft += violations
            reward -= 50.0 * violations  # Penalty for collisions but continue episode

        # Process aircraft exits and track final states
        remaining_aircraft: List[Aircraft] = []
        for ac in self.aircraft:
            if ac.has_exited():
                if ac.is_near_exit():
                    reward += 100.0  # Large positive reward for successful exit
                    self.successful_exits += 1
                else:
                    reward -= 20.0  # Penalty for wrong exit
                    self.failed_exits += 1
            else:
                remaining_aircraft.append(ac)
        self.aircraft = remaining_aircraft

        # Progressive difficulty: Spawn more aircraft over time
        if self.progressive_difficulty:
            # Gradually increase spawn rate and max aircraft
            spawn_probability = min(0.9, 0.3 + (self.current_step * 0.001))
            max_current_aircraft = min(self.max_aircraft, self.initial_aircraft + (self.current_step // 50))
            
            if (
                self.current_step % 10 == 0  # Every 10 steps
                and len(self.aircraft) < max_current_aircraft
                and self.np_random.random() < spawn_probability
            ):
                self._spawn_aircraft()
        else:
            # Original spawning logic
            if (
                self.current_step % 5 == 0
                and len(self.aircraft) < self.max_aircraft
                and self.np_random.random() < 0.8
            ):
                self._spawn_aircraft()

        self.current_step += 1
        self.total_reward += reward

        # Record step if recorder is provided
        if self.recorder is not None:
            env_info = self._get_info()
            env_info['airspace_size'] = AIRSPACE_SIZE
            self.recorder.record_step(self.aircraft, env_info)

        # NEW TERMINATION CONDITION: Episode ends when ALL aircraft have reached final states
        # This means: no active aircraft AND no more aircraft will be spawned
        total_resolved = self.successful_exits + self.failed_exits + self.collided_aircraft
        
        # Stop spawning new aircraft after a certain point to allow resolution
        max_spawn_steps = 200  # Stop spawning after 200 steps
        
        if (len(self.aircraft) == 0 and 
            (self.current_step > max_spawn_steps or total_resolved >= self.total_aircraft_spawned)):
            # All aircraft have been resolved
            self.all_aircraft_resolved = True
            terminated = True
            truncated = False
            
            # Final episode reward based on overall performance
            success_rate = self.successful_exits / max(self.total_aircraft_spawned, 1)
            if success_rate >= 0.8:  # 80% success rate
                reward += 200.0  # Bonus for good performance
            elif success_rate >= 0.6:  # 60% success rate
                reward += 100.0  # Moderate bonus
            elif success_rate >= 0.4:  # 40% success rate
                reward += 50.0   # Small bonus
            
            # Penalty for collisions
            if self.collided_aircraft > 0:
                reward -= 100.0 * self.collided_aircraft
            
            return (
                self._get_observation(),
                reward,
                terminated,
                truncated,
                self._get_info(),
            )

        # Fallback termination (should rarely be reached)
        terminated = self.current_step >= self.max_steps
        truncated = False

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info(),
        )

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #

    def _spawn_aircraft(self) -> None:
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
        aircraft = Aircraft(
            callsign=callsign,
            x=x,
            y=y,
            heading=float(heading),
            target_heading=float(heading),
            exit_edge=exit_edge,
        )
        self.aircraft.append(aircraft)
        self.total_aircraft_spawned += 1

    def _get_observation(self) -> Dict[str, np.ndarray]:
        aircraft_obs = np.zeros((self.max_aircraft, 6), dtype=np.float32)
        mask = np.zeros(self.max_aircraft, dtype=np.uint8)

        half = AIRSPACE_SIZE / 2

        for i, ac in enumerate(self.aircraft[: self.max_aircraft]):
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

    # --------------------------------------------------------------------- #
    # Rendering helpers
    # --------------------------------------------------------------------- #

    def render(self):
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

    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None


# Additional constants for 3D environment
SEPARATION_LATERAL_NM = 3.0  # 3 nautical miles
SEPARATION_VERTICAL_FT = 1000.0  # 1000 feet
CONFLICT_WARNING_BUFFER_NM = 1.0  # Additional warning buffer
TURN_RATE_DEG_PER_SEC = 3.0  # 3 degrees per second
FT_PER_SEC_CLIMB = 2000 / 60  # ~2000 fpm = 33.3 ft/s
FT_PER_SEC_DESCENT = 2000 / 60


@dataclass
class Aircraft3D:
    """Aircraft with realistic 3D dynamics."""
    callsign: str
    x: float  # Position in nm (relative to airport)
    y: float  # Position in nm (relative to airport)
    altitude: float  # Altitude in feet
    heading: float  # Heading in degrees (0-360, 0=North)
    speed: float  # Speed in knots
    
    # Commanded values
    target_altitude: float = None
    target_heading: float = None
    target_speed: float = None
    
    # State
    is_landing: bool = False
    runway_assigned: int = None  # 0, 1, 2, 3 for four runways
    
    def __post_init__(self):
        if self.target_altitude is None:
            self.target_altitude = self.altitude
        if self.target_heading is None:
            self.target_heading = self.heading
        if self.target_speed is None:
            self.target_speed = self.speed
    
    def update(self, dt: float):
        """Update aircraft state based on physics (dt in seconds)."""
        # Update heading (turn at TURN_RATE_DEG_PER_SEC)
        heading_diff = self.target_heading - self.heading
        # Normalize to [-180, 180]
        heading_diff = (heading_diff + 180) % 360 - 180
        
        max_turn = TURN_RATE_DEG_PER_SEC * dt
        if abs(heading_diff) <= max_turn:
            self.heading = self.target_heading
        else:
            self.heading += max_turn * np.sign(heading_diff)
        self.heading = self.heading % 360
        
        # Update altitude
        alt_diff = self.target_altitude - self.altitude
        if abs(alt_diff) > 0:
            climb_rate = FT_PER_SEC_CLIMB if alt_diff > 0 else -FT_PER_SEC_DESCENT
            max_alt_change = abs(climb_rate * dt)
            if abs(alt_diff) <= max_alt_change:
                self.altitude = self.target_altitude
            else:
                self.altitude += max_alt_change * np.sign(alt_diff)
        
        # Update speed (instant for simplicity)
        self.speed = self.target_speed
        
        # Update position based on heading and speed
        heading_rad = np.radians(self.heading)
        speed_nm_per_sec = self.speed / 3600.0
        
        # Heading 0 = North = +y, heading 90 = East = +x
        self.x += speed_nm_per_sec * dt * np.sin(heading_rad)
        self.y += speed_nm_per_sec * dt * np.cos(heading_rad)
    
    def distance_to(self, other: 'Aircraft3D') -> float:
        """Calculate lateral distance in nautical miles."""
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def vertical_separation(self, other: 'Aircraft3D') -> float:
        """Calculate vertical separation in feet."""
        return abs(self.altitude - other.altitude)
    
    def check_conflict(self, other: 'Aircraft3D') -> Tuple[bool, bool]:
        """Check for conflicts and violations.
        
        Returns:
            (is_violation, is_conflict)
        """
        lateral_dist = self.distance_to(other)
        vertical_sep = self.vertical_separation(other)
        
        # If vertical separation is sufficient, no conflict
        if vertical_sep >= SEPARATION_VERTICAL_FT:
            return False, False
        
        # Check lateral separation
        is_violation = lateral_dist < SEPARATION_LATERAL_NM
        is_conflict = lateral_dist < (SEPARATION_LATERAL_NM + CONFLICT_WARNING_BUFFER_NM)
        
        return is_violation, is_conflict


def _pairwise_distances_3d(positions: np.ndarray) -> np.ndarray:
    """Compute lateral distances between all aircraft pairs using numpy broadcasting."""
    deltas = positions[:, None, :2] - positions[None, :, :2]  # xy only
    return np.sqrt(np.sum(deltas**2, axis=-1))


def check_conflicts_vectorized_3d(aircraft_list: List[Aircraft3D]) -> Tuple[int, int]:
    """
    Efficiently compute conflict and separation violation counts for 3D aircraft.
    
    Returns:
        violations: Number of pairs violating minimum separation
        conflicts: Number of pairs in conflict warning zone
    """
    if len(aircraft_list) < 2:
        return 0, 0
    
    # Extract positions [x, y, altitude]
    positions = np.array([[ac.x, ac.y, ac.altitude] for ac in aircraft_list])
    
    # Compute lateral distances using broadcasting
    lateral_distances = _pairwise_distances_3d(positions)
    
    # Compute vertical separations using broadcasting
    altitudes = positions[:, 2]  # altitude column
    vertical_separations = np.abs(altitudes[:, None] - altitudes[None, :])
    
    # Mask for sufficient vertical separation (no conflict if >= 1000ft)
    sufficient_vertical = vertical_separations >= SEPARATION_VERTICAL_FT
    
    # Check violations (insufficient lateral separation AND insufficient vertical)
    violations = np.sum(
        (lateral_distances < SEPARATION_LATERAL_NM) & 
        (~sufficient_vertical) &
        (lateral_distances > 0)  # exclude self-pairs
    ) // 2  # divide by 2 since we count each pair twice
    
    # Check conflicts (warning zone)
    conflicts = np.sum(
        (lateral_distances < SEPARATION_LATERAL_NM + CONFLICT_WARNING_BUFFER_NM) &
        (lateral_distances >= SEPARATION_LATERAL_NM) &
        (~sufficient_vertical) &
        (lateral_distances > 0)
    ) // 2
    
    return violations, conflicts


class Realistic3DATCEnv(gym.Env):
    """Realistic 3D ATC environment with full physics."""
    
    metadata = {'render_modes': ['human'], 'render_fps': 2}
    
    def __init__(
        self,
        max_aircraft: int = 8,
        episode_length: int = 180,  # seconds
        spawn_interval: float = 25.0,  # spawn every 25 seconds
        render_mode: Optional[str] = None,
        recorder: Optional[Any] = None,
    ):
        super().__init__()
        
        self.max_aircraft = max_aircraft
        self.episode_length = episode_length
        self.spawn_interval = spawn_interval
        self.render_mode = render_mode
        self.recorder = recorder
        
        # Runway configuration (two intersecting runways)
        self.runways = [
            {'name': '09', 'heading': 90, 'x1': -2, 'y1': 0, 'x2': 2, 'y2': 0},   # East
            {'name': '27', 'heading': 270, 'x1': 2, 'y1': 0, 'x2': -2, 'y2': 0},  # West
            {'name': '04', 'heading': 45, 'x1': -1.4, 'y1': -1.4, 'x2': 1.4, 'y2': 1.4},  # NE
            {'name': '22', 'heading': 225, 'x1': 1.4, 'y1': 1.4, 'x2': -1.4, 'y2': -1.4}, # SW
        ]
        
        # State
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
        
        # Observation space
        # Per-aircraft: x, y, alt, hdg, spd, tgt_alt, tgt_hdg, tgt_spd,
        #               dx_to_runway, dy_to_runway, runway_id, is_landing, dist, bearing
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
        # [aircraft_id (0 to max_aircraft), command_type (0-4), 
        #  altitude_idx (0-17), heading_idx (0-11), speed_idx (0-7)]
        # command_type: 0=altitude, 1=heading, 2=speed, 3=land, 4=no-op
        self.action_space = spaces.MultiDiscrete([
            max_aircraft + 1,  # aircraft_id
            5,  # command_type
            18,  # altitude (0-17k ft in 1k increments)
            12,  # heading (0-330 in 30° increments)
            8,   # speed (150-360 knots in 30kt increments)
        ])
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
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
        
        return obs, info
    
    def _spawn_aircraft(self):
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
    
    def step(self, action):
        dt = 1.0  # 1 second per step
        
        # Unpack action
        aircraft_id, command_type, altitude_idx, heading_idx, speed_idx = action
        
        # Execute action
        reward = self._execute_action(aircraft_id, command_type, 
                                       altitude_idx, heading_idx, speed_idx)
        
        # Update all aircraft
        for aircraft in self.aircraft:
            aircraft.update(dt)
        
        # Check for conflicts/violations
        conflict_penalty, violation_penalty = self._check_conflicts()
        reward += conflict_penalty + violation_penalty
        
        # Remove aircraft that left the area
        self.aircraft = [
            ac for ac in self.aircraft
            if abs(ac.x) <= AIRSPACE_SIZE/2 and abs(ac.y) <= AIRSPACE_SIZE/2
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
    
    def _execute_action(self, aircraft_id, command_type, altitude_idx, heading_idx, speed_idx) -> float:
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
        
        for i, ac1 in enumerate(self.aircraft):
            for ac2 in self.aircraft[i+1:]:
                is_violation, is_conflict = ac1.check_conflict(ac2)
                
                if is_violation:
                    violation_penalty -= 10.0
                    self.violations += 1
                elif is_conflict:
                    conflict_penalty -= 1.0
                    self.conflicts += 1
        
        return conflict_penalty, violation_penalty
    
    def _check_landings(self) -> float:
        """Check for successful landings and remove landed aircraft."""
        reward = 0.0
        aircraft_to_remove = []
        
        for ac in self.aircraft:
            if ac.is_landing and ac.altitude < 100:
                # Check if close to runway
                if ac.runway_assigned is not None:
                    runway = self.runways[ac.runway_assigned]
                    dist_to_runway = np.sqrt((ac.x - runway['x2'])**2 + (ac.y - runway['y2'])**2)
                    
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
                ac.target_altitude / 10000.0,
                ac.target_heading / 360.0,
                ac.target_speed / 300.0,
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
                    is_violation, is_conflict = ac1.check_conflict(ac2)
                    if is_violation:
                        conflict_matrix[i, j] = 1.0
                    elif is_conflict:
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
        return {
            'time_elapsed': self.time_elapsed,
            'num_aircraft': len(self.aircraft),
            'score': self.score,
            'violations': self.violations,
            'conflicts': self.conflicts,
            'successful_landings': self.successful_landings,
        }
    
    def render(self):
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
    
    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None


__all__ = ["Simple2DATCEnv", "Realistic3DATCEnv", "check_conflicts_vectorized", "check_conflicts_vectorized_3d", "Aircraft", "Aircraft3D"]

