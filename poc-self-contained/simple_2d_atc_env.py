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
        max_steps: int = 100,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.max_aircraft = max_aircraft
        self.max_steps = max_steps
        self.render_mode = render_mode

        self.aircraft: List[Aircraft] = []
        self.current_step = 0
        self.total_reward = 0.0
        self.separations_lost = 0
        self.successful_exits = 0

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

        num_initial = int(self.np_random.integers(2, min(4, self.max_aircraft) + 1))
        for _ in range(num_initial):
            self._spawn_aircraft()

        return self._get_observation(), self._get_info()

    def step(self, action):
        aircraft_id, heading_change_idx = action

        reward = -0.1  # timestep cost

        if aircraft_id < len(self.aircraft):
            heading_change = (heading_change_idx - 12) * 15
            aircraft = self.aircraft[aircraft_id]
            aircraft.target_heading = (aircraft.heading + heading_change) % 360

        for ac in self.aircraft:
            ac.update()

        violations, conflicts = check_conflicts_vectorized(self.aircraft)
        if violations:
            reward -= 100.0 * violations
            self.separations_lost += violations
        if conflicts:
            reward -= 5.0 * conflicts

        remaining_aircraft: List[Aircraft] = []
        for ac in self.aircraft:
            if ac.has_exited():
                if ac.is_near_exit():
                    reward += 20.0
                    self.successful_exits += 1
                else:
                    reward -= 10.0
            else:
                remaining_aircraft.append(ac)
        self.aircraft = remaining_aircraft

        if (
            self.current_step % 10 == 0
            and len(self.aircraft) < self.max_aircraft
            and self.np_random.random() < 0.5
        ):
            self._spawn_aircraft()

        self.current_step += 1
        self.total_reward += reward

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

        callsign = CALLSIGNS[len(self.aircraft) % len(CALLSIGNS)]
        aircraft = Aircraft(
            callsign=callsign,
            x=x,
            y=y,
            heading=float(heading),
            target_heading=float(heading),
            exit_edge=exit_edge,
        )
        self.aircraft.append(aircraft)

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


__all__ = ["Simple2DATCEnv", "check_conflicts_vectorized", "Aircraft"]

