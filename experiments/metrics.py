"""
Metrics tracking for OpenScope RL experiments.

This module provides utilities for tracking and analyzing metrics during
training and evaluation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import numpy as np


@dataclass
class EpisodeMetrics:
    """Metrics for a single episode."""

    episode_number: int
    total_reward: float
    episode_length: int
    success_rate: float = 0.0
    separation_violations: int = 0
    collisions: int = 0
    successful_exits: int = 0
    failed_exits: int = 0
    total_aircraft: int = 0
    commands_issued: int = 0

    # Additional metrics
    extra_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "episode_number": self.episode_number,
            "total_reward": self.total_reward,
            "episode_length": self.episode_length,
            "success_rate": self.success_rate,
            "separation_violations": self.separation_violations,
            "collisions": self.collisions,
            "successful_exits": self.successful_exits,
            "failed_exits": self.failed_exits,
            "total_aircraft": self.total_aircraft,
            "commands_issued": self.commands_issued,
            **self.extra_metrics,
        }


class MetricsTracker:
    """Track metrics across multiple episodes."""

    def __init__(self):
        """Initialize metrics tracker."""
        self.episode_metrics: List[EpisodeMetrics] = []
        self.current_episode_number = 0
        self.current_episode_reward = 0.0
        self.current_episode_length = 0
        self.episode_started = False

    def start_episode(self):
        """Start tracking a new episode."""
        self.current_episode_number += 1
        self.current_episode_reward = 0.0
        self.current_episode_length = 0
        self.episode_started = True

    def update(self, reward: float, info: Dict[str, Any]):
        """
        Update metrics with step information.

        Args:
            reward: Step reward
            info: Step info dictionary
        """
        if not self.episode_started:
            self.start_episode()

        self.current_episode_reward += reward
        self.current_episode_length += 1

    def end_episode(self, episode_info: Optional[Dict[str, Any]] = None) -> EpisodeMetrics:
        """
        End current episode and record metrics.

        Args:
            episode_info: Optional episode information dictionary

        Returns:
            EpisodeMetrics for the completed episode
        """
        if episode_info is None:
            episode_info = {}

        metrics = EpisodeMetrics(
            episode_number=self.current_episode_number,
            total_reward=self.current_episode_reward,
            episode_length=self.current_episode_length,
            success_rate=episode_info.get("success_rate", 0.0),
            separation_violations=episode_info.get("separation_violations", 0),
            collisions=episode_info.get("collisions", 0),
            successful_exits=episode_info.get("successful_exits", 0),
            failed_exits=episode_info.get("failed_exits", 0),
            total_aircraft=episode_info.get("total_aircraft", 0),
            commands_issued=episode_info.get("commands_issued", 0),
            extra_metrics=episode_info.get("extra_metrics", {}),
        )

        self.episode_metrics.append(metrics)
        self.episode_started = False

        return metrics

    def get_summary(self) -> Dict[str, float]:
        """
        Get summary statistics across all episodes.

        Returns:
            Dictionary of summary statistics
        """
        if not self.episode_metrics:
            return {}

        rewards = [m.total_reward for m in self.episode_metrics]
        lengths = [m.episode_length for m in self.episode_metrics]
        success_rates = [m.success_rate for m in self.episode_metrics]
        violations = [m.separation_violations for m in self.episode_metrics]
        collisions = [m.collisions for m in self.episode_metrics]

        return {
            "num_episodes": len(self.episode_metrics),
            "avg_total_reward": np.mean(rewards),
            "std_total_reward": np.std(rewards),
            "min_total_reward": np.min(rewards),
            "max_total_reward": np.max(rewards),
            "avg_episode_length": np.mean(lengths),
            "std_episode_length": np.std(lengths),
            "avg_success_rate": np.mean(success_rates),
            "std_success_rate": np.std(success_rates),
            "avg_separation_violations": np.mean(violations),
            "avg_collisions": np.mean(collisions),
        }

    def print_summary(self):
        """Print summary statistics."""
        summary = self.get_summary()

        if not summary:
            print("No episodes recorded yet.")
            return

        print("\n" + "="*80)
        print("EPISODE SUMMARY")
        print("="*80)
        print(f"Episodes: {summary['num_episodes']}")
        print(f"\nRewards:")
        print(f"  Average: {summary['avg_total_reward']:.2f} ± {summary['std_total_reward']:.2f}")
        print(f"  Range: [{summary['min_total_reward']:.2f}, {summary['max_total_reward']:.2f}]")
        print(f"\nEpisode Length:")
        print(f"  Average: {summary['avg_episode_length']:.1f} ± {summary['std_episode_length']:.1f}")
        print(f"\nPerformance:")
        print(f"  Success Rate: {summary['avg_success_rate']:.2%} ± {summary['std_success_rate']:.2%}")
        print(f"  Avg Violations: {summary['avg_separation_violations']:.2f}")
        print(f"  Avg Collisions: {summary['avg_collisions']:.2f}")
        print("="*80 + "\n")

    def reset(self):
        """Reset all tracked metrics."""
        self.episode_metrics.clear()
        self.current_episode_number = 0
        self.current_episode_reward = 0.0
        self.current_episode_length = 0
        self.episode_started = False
