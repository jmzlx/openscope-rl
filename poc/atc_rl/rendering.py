"""
Rendering utilities for ATC environments.

This module provides rendering utilities and helper functions for
visualizing ATC environments and episodes.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, FancyArrow, Polygon

from .constants import AIRSPACE_SIZE, SEPARATION_MIN, SEPARATION_LATERAL_NM, SEPARATION_VERTICAL_FT


logger = logging.getLogger(__name__)


def render_2d_airspace(ax: plt.Axes, airspace_size: float = AIRSPACE_SIZE) -> None:
    """
    Render 2D airspace boundaries and exit zones.
    
    Args:
        ax: Matplotlib axes to render on
        airspace_size: Size of the airspace
    """
    half = airspace_size / 2
    
    # Draw airspace boundary
    boundary = mpatches.Rectangle(
        (-half, -half),
        airspace_size,
        airspace_size,
        fill=False,
        edgecolor="gray",
        linewidth=2,
    )
    ax.add_patch(boundary)
    
    # Draw exit zones
    exit_size = 2.0
    exits = [
        {'x': 0, 'y': half, 'width': airspace_size, 'height': exit_size},
        {'x': 0, 'y': -half, 'width': airspace_size, 'height': exit_size},
        {'x': half, 'y': 0, 'width': exit_size, 'height': airspace_size},
        {'x': -half, 'y': 0, 'width': exit_size, 'height': airspace_size},
    ]
    
    for exit_zone in exits:
        rect = mpatches.Rectangle(
            (exit_zone['x'] - exit_zone['width']/2, 
             exit_zone['y'] - exit_zone['height']/2),
            exit_zone['width'], exit_zone['height'],
            facecolor='green', alpha=0.3, edgecolor='green', linewidth=2
        )
        ax.add_patch(rect)


def render_3d_runways(ax: plt.Axes, runways: List[Dict[str, Any]]) -> None:
    """
    Render 3D runways on the airspace.
    
    Args:
        ax: Matplotlib axes to render on
        runways: List of runway dictionaries
    """
    for runway in runways:
        ax.plot(
            [runway['x1'], runway['x2']],
            [runway['y1'], runway['y2']],
            color='white', linewidth=3, label=runway['name']
        )


def render_aircraft_2d(ax: plt.Axes, aircraft: Any, color: str = "blue", 
                      show_separation: bool = True) -> None:
    """
    Render a 2D aircraft with heading arrow and separation circle.
    
    Args:
        ax: Matplotlib axes to render on
        aircraft: Aircraft object with x, y, heading, callsign attributes
        color: Color for the aircraft
        show_separation: Whether to show separation circle
    """
    # Draw separation circle
    if show_separation:
        circle = Circle(
            (aircraft.x, aircraft.y),
            SEPARATION_MIN,
            fill=False,
            edgecolor="orange",
            linestyle="--",
            alpha=0.3,
        )
        ax.add_patch(circle)
    
    # Draw heading arrow
    heading_rad = np.radians(aircraft.heading)
    dx = 1.5 * np.sin(heading_rad)
    dy = 1.5 * np.cos(heading_rad)
    arrow = FancyArrow(
        aircraft.x,
        aircraft.y,
        dx,
        dy,
        width=0.5,
        head_width=1.0,
        head_length=0.8,
        color=color,
        alpha=0.7,
    )
    ax.add_patch(arrow)
    
    # Draw callsign label
    ax.text(
        aircraft.x,
        aircraft.y + 1.2,
        aircraft.callsign,
        ha="center",
        va="bottom",
        fontsize=8,
        color=color,
    )


def render_aircraft_3d(ax: plt.Axes, aircraft: Any, color: str = "yellow", 
                      size: float = 0.5) -> None:
    """
    Render a 3D aircraft as a triangle pointing in heading direction.
    
    Args:
        ax: Matplotlib axes to render on
        aircraft: Aircraft object with x, y, heading, callsign, altitude attributes
        color: Color for the aircraft
        size: Size of the aircraft triangle
    """
    heading_rad = np.radians(aircraft.heading)
    
    # Triangle vertices (pointing up by default)
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
    rotated[:, 0] += aircraft.x
    rotated[:, 1] += aircraft.y
    
    triangle = Polygon(
        rotated,
        closed=True,
        facecolor=color,
        edgecolor='orange',
        linewidth=1
    )
    ax.add_patch(triangle)
    
    # Label with callsign and altitude
    ax.text(
        aircraft.x, aircraft.y + 0.8,
        f"{aircraft.callsign}\n{int(aircraft.altitude)}ft",
        color='white',
        fontsize=8,
        ha='center',
        va='bottom'
    )


def render_aircraft_trail(ax: plt.Axes, trail_points: List[Tuple[float, float]], 
                         color: str = "blue", alpha: float = 0.6) -> None:
    """
    Render aircraft trail as a line.
    
    Args:
        ax: Matplotlib axes to render on
        trail_points: List of (x, y) points
        color: Color for the trail
        alpha: Transparency of the trail
    """
    if len(trail_points) < 2:
        return
    
    x_coords = [p[0] for p in trail_points]
    y_coords = [p[1] for p in trail_points]
    
    ax.plot(x_coords, y_coords, color=color, alpha=alpha, linewidth=1)


def render_conflict_warning(ax: plt.Axes, aircraft1: Any, aircraft2: Any, 
                           is_violation: bool = False) -> None:
    """
    Render conflict warning between two aircraft.
    
    Args:
        ax: Matplotlib axes to render on
        aircraft1: First aircraft
        aircraft2: Second aircraft
        is_violation: Whether this is a violation (red) or warning (yellow)
    """
    color = "red" if is_violation else "yellow"
    alpha = 0.8 if is_violation else 0.5
    
    # Draw line between aircraft
    ax.plot([aircraft1.x, aircraft2.x], [aircraft1.y, aircraft2.y], 
            color=color, linewidth=2, alpha=alpha)
    
    # Add conflict label
    mid_x = (aircraft1.x + aircraft2.x) / 2
    mid_y = (aircraft1.y + aircraft2.y) / 2
    
    label = "VIOLATION" if is_violation else "CONFLICT"
    ax.text(mid_x, mid_y, label, ha="center", va="center", 
            color=color, fontsize=8, fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))


def render_info_panel(ax: plt.Axes, info: Dict[str, Any], 
                     position: Tuple[float, float] = None) -> None:
    """
    Render information panel with episode statistics.
    
    Args:
        ax: Matplotlib axes to render on
        info: Dictionary with information to display
        position: Position for the panel (x, y)
    """
    if position is None:
        airspace_size = info.get('airspace_size', AIRSPACE_SIZE)
        position = (-airspace_size/2 + 1, airspace_size/2 - 1)
    
    info_text = (
        f"Step: {info.get('step', 0)}\n"
        f"Aircraft: {info.get('num_aircraft', 0)}\n"
        f"Score: {info.get('score', 0):.1f}\n"
        f"Violations: {info.get('violations', 0)}\n"
        f"Conflicts: {info.get('conflicts', 0)}\n"
        f"Exits: {info.get('successful_exits', 0)}\n"
        f"Landings: {info.get('successful_landings', 0)}"
    )
    
    ax.text(
        position[0], position[1],
        info_text,
        va="top",
        fontsize=9,
        color="white",
        bbox=dict(boxstyle="round", facecolor="black", alpha=0.8)
    )


def setup_2d_plot(airspace_size: float = AIRSPACE_SIZE, 
                  title: str = "ATC Simulation") -> Tuple[plt.Figure, plt.Axes]:
    """
    Setup a 2D plot for ATC visualization.
    
    Args:
        airspace_size: Size of the airspace
        title: Title for the plot
        
    Returns:
        Tuple of (figure, axes)
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    half = airspace_size / 2
    ax.set_xlim(-half - 1, half + 1)
    ax.set_ylim(-half - 1, half + 1)
    ax.set_aspect("equal")
    ax.set_facecolor("black")
    fig.patch.set_facecolor("black")
    
    ax.grid(True, alpha=0.3, color="gray")
    ax.set_xlabel("X (nautical miles)", color="white")
    ax.set_ylabel("Y (nautical miles)", color="white")
    ax.tick_params(colors="white")
    ax.set_title(title, color="white", fontsize=14)
    
    return fig, ax


def setup_3d_plot(airspace_size: float = AIRSPACE_SIZE, 
                  title: str = "ATC Simulation") -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    """
    Setup a 3D plot with top-down and altitude views.
    
    Args:
        airspace_size: Size of the airspace
        title: Title for the plot
        
    Returns:
        Tuple of (figure, (top_axes, altitude_axes))
    """
    fig, (ax_top, ax_alt) = plt.subplots(1, 2, figsize=(16, 8))
    
    half = airspace_size / 2
    
    # Top-down view
    ax_top.set_xlim(-half, half)
    ax_top.set_ylim(-half, half)
    ax_top.set_aspect("equal")
    ax_top.set_facecolor("black")
    ax_top.set_title(f"{title} - Top View", color="white")
    ax_top.grid(True, color="gray", alpha=0.3)
    ax_top.set_xlabel("X (nautical miles)", color="white")
    ax_top.set_ylabel("Y (nautical miles)", color="white")
    ax_top.tick_params(colors="white")
    
    # Altitude profile view
    ax_alt.set_xlim(-half, half)
    ax_alt.set_ylim(0, 12000)  # 0-12000 feet
    ax_alt.set_facecolor("black")
    ax_alt.set_title("Altitude Profile", color="white")
    ax_alt.set_xlabel("X (nautical miles)", color="white")
    ax_alt.set_ylabel("Altitude (feet)", color="white")
    ax_alt.tick_params(colors="white")
    ax_alt.grid(True, color="gray", alpha=0.3)
    
    fig.patch.set_facecolor("black")
    
    return fig, (ax_top, ax_alt)


def create_aircraft_colors(num_aircraft: int) -> List[str]:
    """
    Create a list of distinct colors for aircraft.
    
    Args:
        num_aircraft: Number of aircraft
        
    Returns:
        List of color strings
    """
    colors = plt.cm.tab10(np.linspace(0, 1, min(num_aircraft, 10)))
    if num_aircraft > 10:
        colors = np.tile(colors, (num_aircraft // 10 + 1, 1))[:num_aircraft]
    
    return [plt.colors.to_hex(color) for color in colors]


def render_episode_summary(episode_data: Dict[str, Any]) -> None:
    """
    Render a summary visualization of an episode.
    
    Args:
        episode_data: Episode data from ATCRecorder
    """
    metadata = episode_data['metadata']
    aircraft_states = episode_data['aircraft_states']
    
    env_type = metadata.get('env_type', '2d')
    airspace_size = metadata.get('airspace_size', AIRSPACE_SIZE)
    
    if env_type == '3d':
        fig, (ax_top, ax_alt) = setup_3d_plot(airspace_size, "Episode Summary")
        
        # Draw runways
        runways = metadata.get('runways', [])
        render_3d_runways(ax_top, runways)
        
        # Show all aircraft trails
        colors = create_aircraft_colors(len(aircraft_states[0]) if aircraft_states else 0)
        
        for i, states in enumerate(aircraft_states):
            for j, state in enumerate(states):
                if j < len(colors):
                    # Top-down trail
                    trail_points = [(s.x, s.y) for s in states if s.callsign == state.callsign]
                    render_aircraft_trail(ax_top, trail_points, colors[j])
                    
                    # Altitude trail
                    alt_trail_points = [(s.x, s.altitude) for s in states if s.callsign == state.callsign and s.altitude is not None]
                    render_aircraft_trail(ax_alt, alt_trail_points, colors[j])
        
        # Render info panel
        render_info_panel(ax_top, metadata)
        
    else:
        fig, ax = setup_2d_plot(airspace_size, "Episode Summary")
        
        # Draw airspace
        render_2d_airspace(ax, airspace_size)
        
        # Show all aircraft trails
        colors = create_aircraft_colors(len(aircraft_states[0]) if aircraft_states else 0)
        
        for i, states in enumerate(aircraft_states):
            for j, state in enumerate(states):
                if j < len(colors):
                    trail_points = [(s.x, s.y) for s in states if s.callsign == state.callsign]
                    render_aircraft_trail(ax, trail_points, colors[j])
        
        # Render info panel
        render_info_panel(ax, metadata)
    
    plt.tight_layout()
    plt.show()
