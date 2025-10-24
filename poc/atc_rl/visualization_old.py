"""
ATC Episode Visualization System

This module provides recording and playback capabilities for ATC simulation episodes,
allowing users to visualize complete scenarios with all aircraft movements over time.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Polygon, Circle
from matplotlib.collections import LineCollection
from matplotlib.widgets import Button, Slider
import pickle
import os
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import time

# Import constants from environment module
try:
    from .environment import AIRCRAFT_SPEED
except ImportError:
    # Fallback if importing from same module
    AIRCRAFT_SPEED = 4.0  # Default speed for 2D aircraft


@dataclass
class AircraftState:
    """Single aircraft state at a point in time."""
    callsign: str
    x: float
    y: float
    altitude: Optional[float] = None
    heading: float = 0.0
    speed: float = 0.0
    is_landing: bool = False
    runway_assigned: Optional[int] = None


class ATCRecorder:
    """Records complete episode state history for visualization."""
    
    def __init__(self, max_aircraft: int = 10, max_timesteps: int = 200):
        self.max_aircraft = max_aircraft
        self.max_timesteps = max_timesteps
        
        # Episode data storage
        self.timesteps = []
        self.aircraft_states = []
        self.metadata = {}
        self.current_step = 0
        
        # Pre-allocate arrays for efficiency
        self._reset_arrays()
    
    def _reset_arrays(self):
        """Reset arrays for new episode."""
        self.timesteps = []
        self.aircraft_states = []
        self.current_step = 0
        self.metadata = {}
    
    def start_episode(self, env_info: Dict[str, Any]):
        """Start recording a new episode."""
        self._reset_arrays()
        self.metadata.update({
            'env_type': env_info.get('env_type', 'unknown'),
            'airspace_size': env_info.get('airspace_size', 20.0),
            'runways': env_info.get('runways', []),
            'max_aircraft': env_info.get('max_aircraft', self.max_aircraft),
            'episode_length': env_info.get('episode_length', 150),
            'start_time': time.time(),
        })
    
    def record_step(self, aircraft_list: List[Any], env_info: Dict[str, Any]):
        """Record aircraft states for current timestep."""
        if self.current_step >= self.max_timesteps:
            return
        
        # Extract aircraft states
        states = []
        for aircraft in aircraft_list:
            state = AircraftState(
                callsign=aircraft.callsign,
                x=aircraft.x,
                y=aircraft.y,
                altitude=getattr(aircraft, 'altitude', None),
                heading=aircraft.heading,
                speed=getattr(aircraft, 'speed', AIRCRAFT_SPEED),  # Use constant for 2D
                is_landing=getattr(aircraft, 'is_landing', False),
                runway_assigned=getattr(aircraft, 'runway_assigned', None),
            )
            states.append(state)
        
        self.aircraft_states.append(states)
        self.timesteps.append(self.current_step)
        self.current_step += 1
        
        # Update metadata with final episode info
        self.metadata.update({
            'final_violations': env_info.get('violations', 0),
            'final_conflicts': env_info.get('conflicts', 0),
            'successful_exits': env_info.get('successful_exits', 0),
            'successful_landings': env_info.get('successful_landings', 0),
            'total_score': env_info.get('score', 0),
            'total_timesteps': self.current_step,
        })
    
    def get_episode_data(self) -> Dict[str, Any]:
        """Get complete episode data for visualization."""
        return {
            'timesteps': self.timesteps,
            'aircraft_states': self.aircraft_states,
            'metadata': self.metadata,
        }
    
    def save(self, filepath: str):
        """Save episode data to file."""
        episode_data = self.get_episode_data()
        with open(filepath, 'wb') as f:
            pickle.dump(episode_data, f)
        print(f"Episode saved to {filepath}")
    
    @staticmethod
    def load(filepath: str) -> Dict[str, Any]:
        """Load episode data from file."""
        with open(filepath, 'rb') as f:
            episode_data = pickle.load(f)
        return episode_data


class ATCPlayer:
    """Visualizes recorded ATC episodes with interactive controls."""
    
    def __init__(self, episode_data: Dict[str, Any], trail_length: int = 30):
        self.episode_data = episode_data
        self.timesteps = episode_data['timesteps']
        self.aircraft_states = episode_data['aircraft_states']
        self.metadata = episode_data['metadata']
        self.trail_length = trail_length
        
        # Animation state
        self.current_frame = 0
        self.is_playing = False
        self.playback_speed = 1.0
        
        # Visual elements
        self.fig = None
        self.axes = None
        self.aircraft_artists = []
        self.trail_collections = []
        self.text_artists = []
        self.info_text = None
        
        # Colors for aircraft
        self.colors = plt.cm.tab10(np.linspace(0, 1, 10))
        
    def _create_aircraft_triangle(self, x: float, y: float, heading: float, 
                                color: str, size: float = 0.3) -> Polygon:
        """Create aircraft triangle pointing in heading direction."""
        heading_rad = np.radians(heading)
        
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
        rotated[:, 0] += x
        rotated[:, 1] += y
        
        return Polygon(rotated, closed=True, facecolor=color, 
                      edgecolor='black', linewidth=1, alpha=0.8)
    
    def _setup_2d_view(self):
        """Setup 2D top-down view."""
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        
        airspace_size = self.metadata.get('airspace_size', 20.0)
        self.ax.set_xlim(-airspace_size/2, airspace_size/2)
        self.ax.set_ylim(-airspace_size/2, airspace_size/2)
        self.ax.set_aspect('equal')
        self.ax.set_facecolor('black')
        self.fig.patch.set_facecolor('black')
        
        # Draw exit zones
        exit_size = 2.0
        exits = [
            {'x': 0, 'y': airspace_size/2, 'width': airspace_size, 'height': exit_size},
            {'x': 0, 'y': -airspace_size/2, 'width': airspace_size, 'height': exit_size},
            {'x': airspace_size/2, 'y': 0, 'width': exit_size, 'height': airspace_size},
            {'x': -airspace_size/2, 'y': 0, 'width': exit_size, 'height': airspace_size},
        ]
        
        for exit_zone in exits:
            rect = plt.Rectangle(
                (exit_zone['x'] - exit_zone['width']/2, 
                 exit_zone['y'] - exit_zone['height']/2),
                exit_zone['width'], exit_zone['height'],
                facecolor='green', alpha=0.3, edgecolor='green', linewidth=2
            )
            self.ax.add_patch(rect)
        
        self.ax.grid(True, color='gray', alpha=0.3)
        self.ax.set_xlabel('X (nautical miles)', color='white')
        self.ax.set_ylabel('Y (nautical miles)', color='white')
        self.ax.tick_params(colors='white')
        self.ax.set_title('ATC Simulation - 2D View', color='white', fontsize=14)
    
    def _setup_3d_view(self):
        """Setup 3D view with top-down and altitude profile."""
        self.fig, (self.ax_top, self.ax_alt) = plt.subplots(1, 2, figsize=(16, 8))
        
        airspace_size = self.metadata.get('airspace_size', 20.0)
        
        # Top-down view
        self.ax_top.set_xlim(-airspace_size/2, airspace_size/2)
        self.ax_top.set_ylim(-airspace_size/2, airspace_size/2)
        self.ax_top.set_aspect('equal')
        self.ax_top.set_facecolor('black')
        self.ax_top.set_title('ATC Simulation - Top View', color='white')
        
        # Draw runways
        runways = self.metadata.get('runways', [])
        for runway in runways:
            self.ax_top.plot(
                [runway['x1'], runway['x2']],
                [runway['y1'], runway['y2']],
                color='white', linewidth=3, label=runway['name']
            )
        
        # Altitude profile view
        self.ax_alt.set_xlim(-airspace_size/2, airspace_size/2)
        self.ax_alt.set_ylim(0, 12000)  # 0-12000 feet
        self.ax_alt.set_facecolor('black')
        self.ax_alt.set_title('Altitude Profile', color='white')
        self.ax_alt.set_xlabel('X (nautical miles)', color='white')
        self.ax_alt.set_ylabel('Altitude (feet)', color='white')
        self.ax_alt.tick_params(colors='white')
        
        self.fig.patch.set_facecolor('black')
        self.ax_top.grid(True, color='gray', alpha=0.3)
        self.ax_alt.grid(True, color='gray', alpha=0.3)
    
    def _update_frame(self, frame: int):
        """Update visualization for given frame."""
        if frame >= len(self.aircraft_states):
            return
        
        # Clear previous artists
        for artist in self.aircraft_artists:
            artist.remove()
        for collection in self.trail_collections:
            collection.remove()
        for text in self.text_artists:
            text.remove()
        
        self.aircraft_artists = []
        self.trail_collections = []
        self.text_artists = []
        
        # Get current aircraft states
        current_states = self.aircraft_states[frame]
        
        # Determine which axes to use
        axes = [self.ax] if hasattr(self, 'ax') else [self.ax_top, self.ax_alt]
        
        for i, state in enumerate(current_states):
            color = self.colors[i % len(self.colors)]
            
            # Draw aircraft
            for ax in axes:
                triangle = self._create_aircraft_triangle(
                    state.x, state.y, state.heading, color
                )
                ax.add_patch(triangle)
                self.aircraft_artists.append(triangle)
                
                # Add callsign label
                label_text = ax.text(
                    state.x, state.y + 0.8,
                    f"{state.callsign}\n{int(state.speed)}kt",
                    color='white', fontsize=8, ha='center', va='bottom'
                )
                self.text_artists.append(label_text)
                
                # Add altitude label for 3D view
                if state.altitude is not None and len(axes) > 1 and ax == axes[1]:
                    alt_text = ax.text(
                        state.x, state.altitude/1000,  # Scale altitude
                        f"{int(state.altitude)}ft",
                        color=color, fontsize=7, ha='center', va='bottom'
                    )
                    self.text_artists.append(alt_text)
            
            # Draw trails
            trail_start = max(0, frame - self.trail_length)
            if trail_start < frame:
                trail_x = []
                trail_y = []
                trail_alt = []
                
                for t in range(trail_start, frame + 1):
                    if t < len(self.aircraft_states):
                        states_at_t = self.aircraft_states[t]
                        for s in states_at_t:
                            if s.callsign == state.callsign:
                                trail_x.append(s.x)
                                trail_y.append(s.y)
                                if s.altitude is not None:
                                    trail_alt.append(s.altitude)
                                break
                
                if len(trail_x) > 1:
                    # Top-down trail
                    trail_points = np.array([trail_x, trail_y]).T.reshape(-1, 1, 2)
                    trail_segments = np.concatenate([trail_points[:-1], trail_points[1:]], axis=1)
                    trail_collection = LineCollection(
                        trail_segments, colors=color, alpha=0.6, linewidths=1
                    )
                    axes[0].add_collection(trail_collection)
                    self.trail_collections.append(trail_collection)
                    
                    # Altitude trail for 3D view
                    if len(axes) > 1 and len(trail_alt) > 1:
                        alt_trail_points = np.array([trail_x, trail_alt]).T.reshape(-1, 1, 2)
                        alt_trail_segments = np.concatenate([alt_trail_points[:-1], alt_trail_points[1:]], axis=1)
                        alt_trail_collection = LineCollection(
                            alt_trail_segments, colors=color, alpha=0.6, linewidths=1
                        )
                        axes[1].add_collection(alt_trail_collection)
                        self.trail_collections.append(alt_trail_collection)
        
        # Update info panel
        self._update_info_panel(frame)
    
    def _update_info_panel(self, frame: int):
        """Update information panel with current stats."""
        if frame >= len(self.aircraft_states):
            return
        
        current_states = self.aircraft_states[frame]
        
        info_text = (
            f"Time: {frame}s\n"
            f"Aircraft: {len(current_states)}\n"
            f"Final Score: {self.metadata.get('total_score', 0):.1f}\n"
            f"Violations: {self.metadata.get('final_violations', 0)}\n"
            f"Landings: {self.metadata.get('successful_landings', 0)}\n"
            f"Exits: {self.metadata.get('successful_exits', 0)}"
        )
        
        # Remove old info text
        if self.info_text is not None:
            self.info_text.remove()
        
        # Add new info text
        axes = [self.ax] if hasattr(self, 'ax') else [self.ax_top]
        airspace_size = self.metadata.get('airspace_size', 20.0)
        
        self.info_text = axes[0].text(
            -airspace_size/2 + 1, airspace_size/2 - 1,
            info_text,
            color='white', fontsize=10, va='top',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7)
        )
    
    def show(self, interactive: bool = True):
        """Display the visualization."""
        env_type = self.metadata.get('env_type', '2d')
        
        if env_type == '3d':
            self._setup_3d_view()
        else:
            self._setup_2d_view()
        
        # Initial frame
        self._update_frame(0)
        
        if interactive:
            # Add controls
            self._add_controls()
        
        plt.tight_layout()
        plt.show()
    
    def _add_controls(self):
        """Add interactive controls."""
        # Play/Pause button
        ax_play = plt.axes([0.1, 0.02, 0.1, 0.04])
        self.btn_play = Button(ax_play, 'Play')
        self.btn_play.on_clicked(self._toggle_play)
        
        # Speed slider
        ax_speed = plt.axes([0.25, 0.02, 0.2, 0.04])
        self.slider_speed = Slider(ax_speed, 'Speed', 0.1, 3.0, valinit=1.0)
        self.slider_speed.on_changed(self._update_speed)
        
        # Timeline slider
        ax_timeline = plt.axes([0.5, 0.02, 0.4, 0.04])
        self.slider_timeline = Slider(ax_timeline, 'Time', 0, len(self.timesteps)-1, 
                                     valinit=0, valfmt='%d')
        self.slider_timeline.on_changed(self._seek_to_frame)
        
        # Animation timer
        self.animation_timer = None
    
    def _toggle_play(self, event):
        """Toggle play/pause."""
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.btn_play.label.set_text('Pause')
            self._start_animation()
        else:
            self.btn_play.label.set_text('Play')
            self._stop_animation()
    
    def _start_animation(self):
        """Start the animation loop."""
        if self.animation_timer is not None:
            self.animation_timer.stop()
        
        # Calculate interval based on speed (milliseconds)
        interval = max(50, int(100 / self.playback_speed))
        self.animation_timer = self.fig.canvas.new_timer(interval)
        self.animation_timer.add_callback(self._animate_frame)
        self.animation_timer.start()
    
    def _stop_animation(self):
        """Stop the animation loop."""
        if self.animation_timer is not None:
            self.animation_timer.stop()
            self.animation_timer = None
    
    def _animate_frame(self):
        """Animation frame callback."""
        if not self.is_playing:
            return
        
        if self.current_frame < len(self.timesteps) - 1:
            self.current_frame += 1
            self._update_frame(self.current_frame)
            self.slider_timeline.set_val(self.current_frame)
            self.fig.canvas.draw()
        else:
            # Reached end, stop playing
            self.is_playing = False
            self.btn_play.label.set_text('Play')
            self._stop_animation()
    
    def _update_speed(self, val):
        """Update playback speed."""
        self.playback_speed = val
        # Restart animation with new speed if currently playing
        if self.is_playing:
            self._start_animation()
    
    def _seek_to_frame(self, val):
        """Seek to specific frame."""
        frame = int(val)
        self.current_frame = frame
        self._update_frame(frame)
        self.fig.canvas.draw()
    
    def save_video(self, filepath: str, fps: int = 10):
        """Save animation as video file."""
        env_type = self.metadata.get('env_type', '2d')
        
        if env_type == '3d':
            self._setup_3d_view()
        else:
            self._setup_2d_view()
        
        def animate(frame):
            self._update_frame(frame)
            return self.aircraft_artists + self.trail_collections + self.text_artists
        
        anim = animation.FuncAnimation(
            self.fig, animate, frames=len(self.timesteps),
            interval=1000//fps, blit=False, repeat=False
        )
        
        anim.save(filepath, writer='ffmpeg', fps=fps)
        print(f"Video saved to {filepath}")
    
    def close(self):
        """Clean up resources."""
        self._stop_animation()
        if self.fig is not None:
            plt.close(self.fig)


def create_recorder_for_env(env) -> ATCRecorder:
    max_aircraft = getattr(env, 'max_aircraft', 10)
    episode_length = getattr(env, 'episode_length', 150)
    
    recorder = ATCRecorder(max_aircraft=max_aircraft, max_timesteps=episode_length)
    
    # Determine environment type and extract metadata
    env_type = '3d' if hasattr(env, 'runways') else '2d'
    env_info = {
        'env_type': env_type,
        'airspace_size': getattr(env, 'airspace_size', 20.0),
        'max_aircraft': max_aircraft,
        'episode_length': episode_length,
    }
    
    if env_type == '3d':
        env_info['runways'] = getattr(env, 'runways', [])
    
    recorder.start_episode(env_info)
    return recorder


def visualize_episode(episode_data: Dict[str, Any], interactive: bool = True):
    """Convenience function to visualize episode data."""
    player = ATCPlayer(episode_data)
    player.show(interactive=interactive)
    return player
