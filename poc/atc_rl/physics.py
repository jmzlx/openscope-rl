"""
Physics calculations and conflict detection for POC ATC environments.

This module provides efficient vectorized functions for physics calculations
and conflict detection used in both 2D and 3D environments.
"""

import numpy as np
from typing import List, Tuple, Any
from dataclasses import dataclass

from .constants import (
    SEPARATION_MIN, CONFLICT_BUFFER, SEPARATION_LATERAL_NM, 
    SEPARATION_VERTICAL_FT, CONFLICT_WARNING_BUFFER_NM,
    TURN_RATE, AIRCRAFT_SPEED, AIRSPACE_SIZE
)


@dataclass
class Aircraft2D:
    """Simple 2D aircraft representation for physics calculations."""
    callsign: str
    x: float
    y: float
    heading: float
    target_heading: float
    exit_edge: int


@dataclass
class Aircraft3D:
    """3D aircraft representation for physics calculations."""
    callsign: str
    x: float  # Position in nm
    y: float  # Position in nm
    altitude: float  # Altitude in feet
    heading: float  # Heading in degrees
    speed: float  # Speed in knots
    target_altitude: float = None
    target_heading: float = None
    target_speed: float = None
    is_landing: bool = False
    runway_assigned: int = None


def _pairwise_distances(positions: np.ndarray) -> np.ndarray:
    """
    Compute pairwise distances with broadcasting.
    
    Args:
        positions: Array of shape (n_aircraft, 2) with x,y coordinates
        
    Returns:
        Distance matrix of shape (n_aircraft, n_aircraft)
    """
    deltas = positions[:, None, :] - positions[None, :, :]
    return np.sqrt(np.sum(deltas**2, axis=-1))


def check_conflicts_vectorized(aircraft_list: List[Aircraft2D]) -> Tuple[int, int]:
    """
    Efficiently compute conflict and separation violation counts for 2D aircraft.
    
    Args:
        aircraft_list: List of Aircraft2D objects
        
    Returns:
        Tuple of (violations, conflicts)
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


def _pairwise_distances_3d(positions: np.ndarray) -> np.ndarray:
    """
    Compute lateral distances between all aircraft pairs using numpy broadcasting.
    
    Args:
        positions: Array of shape (n_aircraft, 3) with x,y,altitude coordinates
        
    Returns:
        Lateral distance matrix of shape (n_aircraft, n_aircraft)
    """
    deltas = positions[:, None, :2] - positions[None, :, :2]  # xy only
    return np.sqrt(np.sum(deltas**2, axis=-1))


def check_conflicts_vectorized_3d(aircraft_list: List[Aircraft3D]) -> Tuple[int, int]:
    """
    Efficiently compute conflict and separation violation counts for 3D aircraft.
    
    Args:
        aircraft_list: List of Aircraft3D objects
        
    Returns:
        Tuple of (violations, conflicts)
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


def update_aircraft_2d(aircraft: Aircraft2D, dt: float = 1.0) -> None:
    """
    Update 2D aircraft state based on physics.
    
    Args:
        aircraft: Aircraft2D object to update
        dt: Time step
    """
    # Update heading
    heading_diff = (aircraft.target_heading - aircraft.heading + 180) % 360 - 180
    turn_amount = np.clip(heading_diff, -TURN_RATE, TURN_RATE)
    aircraft.heading = (aircraft.heading + turn_amount) % 360

    # Update position
    heading_rad = np.radians(aircraft.heading)
    aircraft.x += AIRCRAFT_SPEED * np.sin(heading_rad) * dt
    aircraft.y += AIRCRAFT_SPEED * np.cos(heading_rad) * dt


def update_aircraft_3d(aircraft: Aircraft3D, dt: float) -> None:
    """
    Update 3D aircraft state based on realistic physics.
    
    Args:
        aircraft: Aircraft3D object to update
        dt: Time step in seconds
    """
    from .constants import TURN_RATE_DEG_PER_SEC, FT_PER_SEC_CLIMB, FT_PER_SEC_DESCENT
    
    # Update heading (turn at TURN_RATE_DEG_PER_SEC)
    if aircraft.target_heading is not None:
        heading_diff = aircraft.target_heading - aircraft.heading
        # Normalize to [-180, 180]
        heading_diff = (heading_diff + 180) % 360 - 180
        
        max_turn = TURN_RATE_DEG_PER_SEC * dt
        if abs(heading_diff) <= max_turn:
            aircraft.heading = aircraft.target_heading
        else:
            aircraft.heading += max_turn * np.sign(heading_diff)
        aircraft.heading = aircraft.heading % 360
    
    # Update altitude
    if aircraft.target_altitude is not None:
        alt_diff = aircraft.target_altitude - aircraft.altitude
        if abs(alt_diff) > 0:
            climb_rate = FT_PER_SEC_CLIMB if alt_diff > 0 else -FT_PER_SEC_DESCENT
            max_alt_change = abs(climb_rate * dt)
            if abs(alt_diff) <= max_alt_change:
                aircraft.altitude = aircraft.target_altitude
            else:
                aircraft.altitude += max_alt_change * np.sign(alt_diff)
    
    # Update speed (instant for simplicity)
    if aircraft.target_speed is not None:
        aircraft.speed = aircraft.target_speed
    
    # Update position based on heading and speed
    heading_rad = np.radians(aircraft.heading)
    speed_nm_per_sec = aircraft.speed / 3600.0
    
    # Heading 0 = North = +y, heading 90 = East = +x
    aircraft.x += speed_nm_per_sec * dt * np.sin(heading_rad)
    aircraft.y += speed_nm_per_sec * dt * np.cos(heading_rad)


def calculate_distance_2d(ac1: Aircraft2D, ac2: Aircraft2D) -> float:
    """
    Calculate Euclidean distance between two 2D aircraft.
    
    Args:
        ac1: First aircraft
        ac2: Second aircraft
        
    Returns:
        Distance in nautical miles
    """
    return float(np.hypot(ac1.x - ac2.x, ac1.y - ac2.y))


def calculate_distance_3d(ac1: Aircraft3D, ac2: Aircraft3D) -> Tuple[float, float]:
    """
    Calculate lateral and vertical distances between two 3D aircraft.
    
    Args:
        ac1: First aircraft
        ac2: Second aircraft
        
    Returns:
        Tuple of (lateral_distance_nm, vertical_distance_ft)
    """
    lateral_dist = np.sqrt((ac1.x - ac2.x)**2 + (ac1.y - ac2.y)**2)
    vertical_dist = abs(ac1.altitude - ac2.altitude)
    return lateral_dist, vertical_dist


def check_conflict_3d(ac1: Aircraft3D, ac2: Aircraft3D) -> Tuple[bool, bool]:
    """
    Check for conflicts and violations between two 3D aircraft.
    
    Args:
        ac1: First aircraft
        ac2: Second aircraft
        
    Returns:
        Tuple of (is_violation, is_conflict)
    """
    lateral_dist, vertical_dist = calculate_distance_3d(ac1, ac2)
    
    # If vertical separation is sufficient, no conflict
    if vertical_dist >= SEPARATION_VERTICAL_FT:
        return False, False
    
    # Check lateral separation
    is_violation = lateral_dist < SEPARATION_LATERAL_NM
    is_conflict = lateral_dist < (SEPARATION_LATERAL_NM + CONFLICT_WARNING_BUFFER_NM)
    
    return is_violation, is_conflict


def has_exited_2d(aircraft: Aircraft2D) -> bool:
    """
    Check if 2D aircraft has exited the airspace.
    
    Args:
        aircraft: Aircraft2D object
        
    Returns:
        True if aircraft has exited
    """
    from .constants import AIRSPACE_SIZE
    half = AIRSPACE_SIZE / 2
    return abs(aircraft.x) > half or abs(aircraft.y) > half


def is_near_exit_2d(aircraft: Aircraft2D) -> bool:
    """
    Check if 2D aircraft is exiting via the correct edge.
    
    Args:
        aircraft: Aircraft2D object
        
    Returns:
        True if aircraft is near correct exit
    """
    from .constants import AIRSPACE_SIZE
    half = AIRSPACE_SIZE / 2
    heading = aircraft.heading % 360

    if aircraft.exit_edge == 0:  # North
        return aircraft.y > half - 2 and (heading <= 45 or heading >= 315)
    if aircraft.exit_edge == 1:  # East
        return aircraft.x > half - 2 and 45 <= heading <= 135
    if aircraft.exit_edge == 2:  # South
        return aircraft.y < -half + 2 and 135 <= heading <= 225
    # West
    return aircraft.x < -half + 2 and 225 <= heading <= 315


def has_exited_3d(aircraft: Aircraft3D) -> bool:
    """
    Check if 3D aircraft has exited the airspace.
    
    Args:
        aircraft: Aircraft3D object
        
    Returns:
        True if aircraft has exited
    """
    from .constants import AIRSPACE_SIZE
    return abs(aircraft.x) > AIRSPACE_SIZE/2 or abs(aircraft.y) > AIRSPACE_SIZE/2


def is_ready_for_landing(aircraft: Aircraft3D) -> bool:
    """
    Check if 3D aircraft is ready for landing.
    
    Args:
        aircraft: Aircraft3D object
        
    Returns:
        True if aircraft is ready for landing
    """
    return (aircraft.is_landing and 
            aircraft.altitude < 100 and 
            aircraft.runway_assigned is not None)


def calculate_landing_distance(aircraft: Aircraft3D, runway: dict) -> float:
    """
    Calculate distance from aircraft to runway end.
    
    Args:
        aircraft: Aircraft3D object
        runway: Runway dictionary with x2, y2 coordinates
        
    Returns:
        Distance in nautical miles
    """
    return np.sqrt((aircraft.x - runway['x2'])**2 + (aircraft.y - runway['y2'])**2)
