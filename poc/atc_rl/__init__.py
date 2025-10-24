"""
POC ATC RL Module - Self-contained 2D/3D Air Traffic Control environments

This module provides simple, fast environments for rapid prototyping and testing
of air traffic control reinforcement learning algorithms.

Key Components:
- Simple2DATCEnv: 2D airspace with vectorized conflict detection
- Realistic3DATCEnv: 3D environment with altitude, runways, realistic physics
- Physics utilities: Efficient conflict detection and aircraft updates
- Constants: All configuration constants in one place

Usage:
    from poc.atc_rl.environment_2d import Simple2DATCEnv
    from poc.atc_rl.environment_3d import Realistic3DATCEnv
    from poc.atc_rl.physics import check_conflicts_vectorized
"""

from .environment_2d import Simple2DATCEnv
from .environment_3d import Realistic3DATCEnv
from .physics import (
    Aircraft2D,
    Aircraft3D,
    check_conflicts_vectorized,
    check_conflicts_vectorized_3d,
    update_aircraft_2d,
    update_aircraft_3d,
    calculate_distance_2d,
    calculate_distance_3d,
    check_conflict_3d,
    has_exited_2d,
    is_near_exit_2d,
    has_exited_3d,
    is_ready_for_landing,
    calculate_landing_distance,
)
from .recorder import ATCRecorder, AircraftState, create_recorder_for_env
from .player import ATCPlayer, visualize_episode
from .rendering import (
    render_2d_airspace,
    render_3d_runways,
    render_aircraft_2d,
    render_aircraft_3d,
    render_aircraft_trail,
    render_conflict_warning,
    render_info_panel,
    setup_2d_plot,
    setup_3d_plot,
    create_aircraft_colors,
    render_episode_summary,
)
from .constants import (
    AIRSPACE_SIZE,
    SEPARATION_MIN,
    CONFLICT_BUFFER,
    AIRCRAFT_SPEED,
    TURN_RATE,
    SEPARATION_LATERAL_NM,
    SEPARATION_VERTICAL_FT,
    CONFLICT_WARNING_BUFFER_NM,
    TURN_RATE_DEG_PER_SEC,
    FT_PER_SEC_CLIMB,
    FT_PER_SEC_DESCENT,
    CALLSIGNS,
    EXTENDED_CALLSIGNS,
    RUNWAYS,
    DEFAULT_MAX_AIRCRAFT_2D,
    DEFAULT_MAX_AIRCRAFT_3D,
    DEFAULT_MAX_STEPS_2D,
    DEFAULT_EPISODE_LENGTH_3D,
    DEFAULT_SPAWN_INTERVAL_3D,
)

# Public API
__all__ = [
    # Environments
    "Simple2DATCEnv",
    "Realistic3DATCEnv",
    
    # Physics
    "Aircraft2D",
    "Aircraft3D",
    "check_conflicts_vectorized",
    "check_conflicts_vectorized_3d",
    "update_aircraft_2d",
    "update_aircraft_3d",
    "calculate_distance_2d",
    "calculate_distance_3d",
    "check_conflict_3d",
    "has_exited_2d",
    "is_near_exit_2d",
    "has_exited_3d",
    "is_ready_for_landing",
    "calculate_landing_distance",
    
    # Recording and Visualization
    "ATCRecorder",
    "AircraftState",
    "create_recorder_for_env",
    "ATCPlayer",
    "visualize_episode",
    "render_2d_airspace",
    "render_3d_runways",
    "render_aircraft_2d",
    "render_aircraft_3d",
    "render_aircraft_trail",
    "render_conflict_warning",
    "render_info_panel",
    "setup_2d_plot",
    "setup_3d_plot",
    "create_aircraft_colors",
    "render_episode_summary",
    
    # Constants
    "AIRSPACE_SIZE",
    "SEPARATION_MIN",
    "CONFLICT_BUFFER",
    "AIRCRAFT_SPEED",
    "TURN_RATE",
    "SEPARATION_LATERAL_NM",
    "SEPARATION_VERTICAL_FT",
    "CONFLICT_WARNING_BUFFER_NM",
    "TURN_RATE_DEG_PER_SEC",
    "FT_PER_SEC_CLIMB",
    "FT_PER_SEC_DESCENT",
    "CALLSIGNS",
    "EXTENDED_CALLSIGNS",
    "RUNWAYS",
    "DEFAULT_MAX_AIRCRAFT_2D",
    "DEFAULT_MAX_AIRCRAFT_3D",
    "DEFAULT_MAX_STEPS_2D",
    "DEFAULT_EPISODE_LENGTH_3D",
    "DEFAULT_SPAWN_INTERVAL_3D",
]

# Version info
__version__ = "0.1.0"