"""
Episode metrics tracking for OpenScope RL environment.

This module provides classes for tracking episode metrics and performance statistics.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import time


@dataclass
class EpisodeMetrics:
    """
    Tracks metrics for a single episode.
    
    This class provides a clean interface for tracking various episode statistics
    including rewards, commands, conflicts, and performance metrics.
    """
    
    # Episode tracking
    episode_reward: float = 0.0
    episode_length: int = 0
    episode_start_time: Optional[float] = None
    
    # Command tracking
    commands_issued: int = 0
    commands_by_type: Dict[str, int] = field(default_factory=lambda: {
        'altitude': 0, 'heading': 0, 'speed': 0, 'ils': 0, 'direct': 0
    })
    
    # Conflict tracking
    conflicts_encountered: int = 0
    violations: int = 0
    
    # Aircraft tracking
    max_aircraft: int = 0
    total_aircraft_spawned: int = 0
    
    # Performance tracking
    successful_exits: int = 0
    failed_exits: int = 0
    successful_landings: int = 0
    
    def start_episode(self) -> None:
        """Start tracking a new episode."""
        self.episode_reward = 0.0
        self.episode_length = 0
        self.episode_start_time = time.time()
        self.commands_issued = 0
        self.commands_by_type = {
            'altitude': 0, 'heading': 0, 'speed': 0, 'ils': 0, 'direct': 0
        }
        self.conflicts_encountered = 0
        self.violations = 0
        self.max_aircraft = 0
        self.total_aircraft_spawned = 0
        self.successful_exits = 0
        self.failed_exits = 0
        self.successful_landings = 0
    
    def add_reward(self, reward: float) -> None:
        """Add reward to episode total."""
        self.episode_reward += reward
    
    def increment_step(self) -> None:
        """Increment episode step counter."""
        self.episode_length += 1
    
    def record_command(self, command_type: str) -> None:
        """Record a command being issued."""
        self.commands_issued += 1
        if command_type in self.commands_by_type:
            self.commands_by_type[command_type] += 1
    
    def record_conflict(self, is_violation: bool = False) -> None:
        """Record a conflict or violation."""
        self.conflicts_encountered += 1
        if is_violation:
            self.violations += 1
    
    def update_aircraft_count(self, count: int) -> None:
        """Update maximum aircraft count."""
        self.max_aircraft = max(self.max_aircraft, count)
    
    def record_exit(self, successful: bool) -> None:
        """Record an aircraft exit."""
        if successful:
            self.successful_exits += 1
        else:
            self.failed_exits += 1
    
    def record_landing(self) -> None:
        """Record a successful landing."""
        self.successful_landings += 1
    
    def get_episode_duration(self) -> float:
        """Get episode duration in seconds."""
        if self.episode_start_time is None:
            return 0.0
        return time.time() - self.episode_start_time
    
    def get_success_rate(self) -> float:
        """Get success rate for exits."""
        total_exits = self.successful_exits + self.failed_exits
        if total_exits == 0:
            return 0.0
        return self.successful_exits / total_exits
    
    def get_safety_score(self) -> float:
        """Get safety score based on violations."""
        return 1.0 / (1.0 + self.violations)
    
    def get_efficiency_score(self) -> float:
        """Get efficiency score based on successful exits per step."""
        if self.episode_length == 0:
            return 0.0
        return self.successful_exits / self.episode_length
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'episode_reward': self.episode_reward,
            'episode_length': self.episode_length,
            'episode_duration': self.get_episode_duration(),
            'commands_issued': self.commands_issued,
            'commands_by_type': self.commands_by_type.copy(),
            'conflicts_encountered': self.conflicts_encountered,
            'violations': self.violations,
            'max_aircraft': self.max_aircraft,
            'total_aircraft_spawned': self.total_aircraft_spawned,
            'successful_exits': self.successful_exits,
            'failed_exits': self.failed_exits,
            'successful_landings': self.successful_landings,
            'success_rate': self.get_success_rate(),
            'safety_score': self.get_safety_score(),
            'efficiency_score': self.get_efficiency_score(),
        }
    
    def copy(self) -> 'EpisodeMetrics':
        """Create a copy of the metrics."""
        return EpisodeMetrics(
            episode_reward=self.episode_reward,
            episode_length=self.episode_length,
            episode_start_time=self.episode_start_time,
            commands_issued=self.commands_issued,
            commands_by_type=self.commands_by_type.copy(),
            conflicts_encountered=self.conflicts_encountered,
            violations=self.violations,
            max_aircraft=self.max_aircraft,
            total_aircraft_spawned=self.total_aircraft_spawned,
            successful_exits=self.successful_exits,
            failed_exits=self.failed_exits,
            successful_landings=self.successful_landings,
        )


class MetricsTracker:
    """
    Tracks metrics across multiple episodes.
    
    This class provides functionality for tracking performance across multiple
    episodes and calculating rolling averages and statistics.
    """
    
    def __init__(self, window_size: int = 100):
        """
        Initialize metrics tracker.
        
        Args:
            window_size: Size of rolling window for averages
        """
        self.window_size = window_size
        self.episodes: list[EpisodeMetrics] = []
        self.current_episode: Optional[EpisodeMetrics] = None
    
    def start_episode(self) -> EpisodeMetrics:
        """Start tracking a new episode."""
        self.current_episode = EpisodeMetrics()
        self.current_episode.start_episode()
        return self.current_episode
    
    def end_episode(self) -> Optional[EpisodeMetrics]:
        """End current episode and add to history."""
        if self.current_episode is None:
            return None
        
        episode = self.current_episode
        self.episodes.append(episode)
        
        # Keep only recent episodes in memory
        if len(self.episodes) > self.window_size * 2:
            self.episodes = self.episodes[-self.window_size:]
        
        self.current_episode = None
        return episode
    
    def get_current_episode(self) -> Optional[EpisodeMetrics]:
        """Get current episode metrics."""
        return self.current_episode
    
    def get_recent_episodes(self, n: int = None) -> list[EpisodeMetrics]:
        """Get recent episodes."""
        if n is None:
            n = self.window_size
        return self.episodes[-n:] if self.episodes else []
    
    def get_rolling_average_reward(self, n: int = None) -> float:
        """Get rolling average reward."""
        if n is None:
            n = self.window_size
        
        recent = self.get_recent_episodes(n)
        if not recent:
            return 0.0
        
        return sum(ep.episode_reward for ep in recent) / len(recent)
    
    def get_rolling_average_length(self, n: int = None) -> float:
        """Get rolling average episode length."""
        if n is None:
            n = self.window_size
        
        recent = self.get_recent_episodes(n)
        if not recent:
            return 0.0
        
        return sum(ep.episode_length for ep in recent) / len(recent)
    
    def get_rolling_success_rate(self, n: int = None) -> float:
        """Get rolling average success rate."""
        if n is None:
            n = self.window_size
        
        recent = self.get_recent_episodes(n)
        if not recent:
            return 0.0
        
        return sum(ep.get_success_rate() for ep in recent) / len(recent)
    
    def get_rolling_safety_score(self, n: int = None) -> float:
        """Get rolling average safety score."""
        if n is None:
            n = self.window_size
        
        recent = self.get_recent_episodes(n)
        if not recent:
            return 0.0
        
        return sum(ep.get_safety_score() for ep in recent) / len(recent)
    
    def get_total_episodes(self) -> int:
        """Get total number of episodes tracked."""
        return len(self.episodes)
    
    def get_summary_stats(self, n: int = None) -> Dict[str, Any]:
        """Get summary statistics for recent episodes."""
        if n is None:
            n = self.window_size
        
        recent = self.get_recent_episodes(n)
        if not recent:
            return {
                'total_episodes': 0,
                'avg_reward': 0.0,
                'avg_length': 0.0,
                'avg_success_rate': 0.0,
                'avg_safety_score': 0.0,
                'avg_efficiency_score': 0.0,
            }
        
        return {
            'total_episodes': len(recent),
            'avg_reward': sum(ep.episode_reward for ep in recent) / len(recent),
            'avg_length': sum(ep.episode_length for ep in recent) / len(recent),
            'avg_success_rate': sum(ep.get_success_rate() for ep in recent) / len(recent),
            'avg_safety_score': sum(ep.get_safety_score() for ep in recent) / len(recent),
            'avg_efficiency_score': sum(ep.get_efficiency_score() for ep in recent) / len(recent),
            'total_violations': sum(ep.violations for ep in recent),
            'total_conflicts': sum(ep.conflicts_encountered for ep in recent),
        }
