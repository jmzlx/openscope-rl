"""
Constants for POC ATC environments.

This module defines all constants used in the POC environments,
ensuring consistency and easy configuration.
"""

from typing import List

# 2D Environment Constants
AIRSPACE_SIZE = 20.0  # 20nm x 20nm
SEPARATION_MIN = 3.0  # Minimum separation (nm)
CONFLICT_BUFFER = 1.0  # Warning buffer
AIRCRAFT_SPEED = 4.0  # 4 nm / step
TURN_RATE = 15.0  # 15 degrees / step

# 3D Environment Constants
SEPARATION_LATERAL_NM = 3.0  # 3 nautical miles
SEPARATION_VERTICAL_FT = 1000.0  # 1000 feet
CONFLICT_WARNING_BUFFER_NM = 1.0  # Additional warning buffer
TURN_RATE_DEG_PER_SEC = 3.0  # 3 degrees per second
FT_PER_SEC_CLIMB = 2000 / 60  # ~2000 fpm = 33.3 ft/s
FT_PER_SEC_DESCENT = 2000 / 60

# Aircraft Callsigns
CALLSIGNS: List[str] = [
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

# Extended callsigns for larger environments
EXTENDED_CALLSIGNS: List[str] = [
    "AAL123", "UAL456", "DAL789", "SWA101", "FDX202",
    "JBU303", "ASA404", "SKW505", "NKS606", "FFT707",
    "AAL808", "UAL909", "DAL111", "SWA222", "FDX333",
    "JBU444", "ASA555", "SKW666", "NKS777", "FFT888"
]

# Runway Configuration for 3D Environment
RUNWAYS = [
    {'name': '09', 'heading': 90, 'x1': -2, 'y1': 0, 'x2': 2, 'y2': 0},   # East
    {'name': '27', 'heading': 270, 'x1': 2, 'y1': 0, 'x2': -2, 'y2': 0},  # West
    {'name': '04', 'heading': 45, 'x1': -1.4, 'y1': -1.4, 'x2': 1.4, 'y2': 1.4},  # NE
    {'name': '22', 'heading': 225, 'x1': 1.4, 'y1': 1.4, 'x2': -1.4, 'y2': -1.4}, # SW
]

# Environment Defaults
DEFAULT_MAX_AIRCRAFT_2D = 5
DEFAULT_MAX_AIRCRAFT_3D = 8
DEFAULT_MAX_STEPS_2D = 1000
DEFAULT_EPISODE_LENGTH_3D = 180  # seconds
DEFAULT_SPAWN_INTERVAL_3D = 25.0  # seconds

# Reward Constants
REWARD_SUCCESSFUL_EXIT = 100.0
REWARD_FAILED_EXIT = -20.0
REWARD_COLLISION = -50.0
REWARD_ACTION = 0.05
REWARD_TIMESTEP_PENALTY = -0.001

# Episode Completion Bonuses
BONUS_HIGH_SUCCESS_RATE = 200.0
BONUS_MEDIUM_SUCCESS_RATE = 100.0
BONUS_LOW_SUCCESS_RATE = 50.0

# Success Rate Thresholds
HIGH_SUCCESS_THRESHOLD = 0.8
MEDIUM_SUCCESS_THRESHOLD = 0.6
LOW_SUCCESS_THRESHOLD = 0.4

# Physics Constants
GRAVITY_FT_PER_SEC2 = 32.174  # Standard gravity
AIR_DENSITY_SLUG_PER_FT3 = 0.002377  # Sea level air density
SOUND_SPEED_FT_PER_SEC = 1116.5  # Speed of sound at sea level

# Conversion Factors
NM_TO_KM = 1.852  # Nautical miles to kilometers
KM_TO_NM = 1.0 / NM_TO_KM
FT_TO_M = 0.3048  # Feet to meters
M_TO_FT = 1.0 / FT_TO_M
KT_TO_FT_PER_SEC = 1.68781  # Knots to feet per second
FT_PER_SEC_TO_KT = 1.0 / KT_TO_FT_PER_SEC
