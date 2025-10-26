"""
Visualization utilities for OpenScope RL experiments.

Provides plotting and visualization functions for comparing approaches
and analyzing agent behavior.
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, FancyArrow


def plot_learning_curves(metrics_dict: Dict[str, List[float]],
                         title: str = "Learning Curves",
                         xlabel: str = "Episode",
                         ylabel: str = "Metric Value",
                         figsize: Tuple[int, int] = (12, 6),
                         save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot learning curves for multiple metrics.

    Args:
        metrics_dict: Dictionary of {metric_name: values}
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, len(metrics_dict), figsize=figsize)

    if len(metrics_dict) == 1:
        axes = [axes]

    for ax, (metric_name, values) in zip(axes, metrics_dict.items()):
        episodes = np.arange(len(values))

        # Plot raw values with transparency
        ax.plot(episodes, values, alpha=0.3, label=f"Raw {metric_name}")

        # Plot moving average
        if len(values) >= 10:
            window = min(10, len(values))
            moving_avg = np.convolve(values, np.ones(window)/window, mode='valid')
            ax.plot(episodes[window-1:], moving_avg,
                   label=f"{window}-Episode MA", linewidth=2)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(metric_name)
        ax.set_title(metric_name.replace("_", " ").title())
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_comparison(results: Dict[str, Dict[str, float]],
                   metrics: List[str],
                   title: str = "Approach Comparison",
                   figsize: Tuple[int, int] = (14, 6),
                   save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot comparison of multiple approaches across metrics.

    Args:
        results: Dictionary of {approach_name: {metric_name: value}}
        metrics: List of metrics to plot
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    approach_names = list(results.keys())
    num_metrics = len(metrics)

    fig, axes = plt.subplots(1, num_metrics, figsize=figsize)

    if num_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        values = [results[name].get(metric, 0) for name in approach_names]

        # Create bar chart
        x = np.arange(len(approach_names))
        bars = ax.bar(x, values, alpha=0.7)

        # Color bars by performance (higher is better for most metrics)
        colors = plt.cm.viridis(np.array(values) / max(values))
        for bar, color in zip(bars, colors):
            bar.set_color(color)

        ax.set_xticks(x)
        ax.set_xticklabels(approach_names, rotation=45, ha='right')
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(metric.replace("_", " ").title())
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.2f}',
                   ha='center', va='bottom', fontsize=9)

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def visualize_episode(aircraft_states: List[Dict[str, Any]],
                     airspace_size: float = 20.0,
                     save_path: Optional[str] = None) -> plt.Figure:
    """
    Visualize aircraft trajectories from an episode.

    Args:
        aircraft_states: List of aircraft state dictionaries at each timestep
        airspace_size: Size of airspace in nm
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # Setup airspace view
    ax.set_xlim(-airspace_size/2, airspace_size/2)
    ax.set_ylim(-airspace_size/2, airspace_size/2)
    ax.set_aspect('equal')
    ax.set_facecolor('#1a1a2e')
    ax.grid(True, alpha=0.2, color='white')
    ax.set_xlabel('X Position (nm)', color='white', fontsize=12)
    ax.set_ylabel('Y Position (nm)', color='white', fontsize=12)
    ax.tick_params(colors='white')

    # Draw exit zones
    exit_zone_size = 2.0
    exits = [
        mpatches.Rectangle((-airspace_size/2, -airspace_size/2), exit_zone_size, airspace_size,
                          facecolor='green', alpha=0.2, edgecolor='green', linewidth=2),
        mpatches.Rectangle((airspace_size/2-exit_zone_size, -airspace_size/2), exit_zone_size, airspace_size,
                          facecolor='green', alpha=0.2, edgecolor='green', linewidth=2),
        mpatches.Rectangle((-airspace_size/2, -airspace_size/2), airspace_size, exit_zone_size,
                          facecolor='green', alpha=0.2, edgecolor='green', linewidth=2),
        mpatches.Rectangle((-airspace_size/2, airspace_size/2-exit_zone_size), airspace_size, exit_zone_size,
                          facecolor='green', alpha=0.2, edgecolor='green', linewidth=2)
    ]
    for exit_rect in exits:
        ax.add_patch(exit_rect)

    # Collect all aircraft callsigns
    all_callsigns = set()
    for timestep in aircraft_states:
        for aircraft in timestep:
            all_callsigns.add(aircraft['callsign'])

    # Assign colors to aircraft
    colors = plt.cm.tab20(np.linspace(0, 1, len(all_callsigns)))
    color_map = {callsign: color for callsign, color in zip(all_callsigns, colors)}

    # Draw trajectories
    for callsign in all_callsigns:
        positions = []
        for timestep in aircraft_states:
            for aircraft in timestep:
                if aircraft['callsign'] == callsign:
                    positions.append((aircraft['x'], aircraft['y']))
                    break

        if len(positions) > 1:
            positions = np.array(positions)
            ax.plot(positions[:, 0], positions[:, 1],
                   color=color_map[callsign], alpha=0.6, linewidth=2,
                   label=callsign)

            # Draw start and end points
            ax.plot(positions[0, 0], positions[0, 1], 'o',
                   color=color_map[callsign], markersize=8)
            ax.plot(positions[-1, 0], positions[-1, 1], 's',
                   color=color_map[callsign], markersize=8)

    ax.set_title('Aircraft Trajectories', color='white', fontsize=14, pad=20)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#1a1a2e')

    return fig


def plot_sample_efficiency(results: Dict[str, Tuple[List[int], List[float]]],
                          title: str = "Sample Efficiency Comparison",
                          xlabel: str = "Training Steps",
                          ylabel: str = "Success Rate",
                          figsize: Tuple[int, int] = (10, 6),
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot sample efficiency curves for different approaches.

    Args:
        results: Dictionary of {approach_name: (steps, success_rates)}
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    for approach_name, (steps, values) in results.items():
        ax.plot(steps, values, label=approach_name, linewidth=2, marker='o', markersize=4)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_attention_weights(attention_weights: np.ndarray,
                          aircraft_ids: List[str],
                          title: str = "Aircraft Selection Attention",
                          figsize: Tuple[int, int] = (10, 6),
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Visualize attention weights over aircraft (for hierarchical/attention models).

    Args:
        attention_weights: Array of attention weights (timesteps, num_aircraft)
        aircraft_ids: List of aircraft callsigns
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    im = ax.imshow(attention_weights.T, aspect='auto', cmap='hot', interpolation='nearest')

    # Set labels
    ax.set_xlabel('Timestep', fontsize=12)
    ax.set_ylabel('Aircraft', fontsize=12)
    ax.set_title(title, fontsize=14)

    # Set y-ticks to aircraft IDs
    ax.set_yticks(np.arange(len(aircraft_ids)))
    ax.set_yticklabels(aircraft_ids)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Attention Weight', fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig
