"""
Metrics collection and tracking for OpenScope RL experiments.

Provides standardized metrics for fair comparison across approaches.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import numpy as np
from collections import defaultdict


@dataclass
class ATCMetrics:
    """
    Standard metrics for ATC performance evaluation.

    Attributes:
        success_rate: Percentage of aircraft that exited successfully
        separation_violations: Number of separation violations
        collisions: Number of collisions
        throughput: Aircraft per hour
        avg_exit_time: Average time for aircraft to exit
        command_efficiency: Commands per aircraft
        episode_length: Total episode length in steps
        total_reward: Total episode reward
    """
    success_rate: float = 0.0
    separation_violations: int = 0
    collisions: int = 0
    throughput: float = 0.0
    avg_exit_time: float = 0.0
    command_efficiency: float = 0.0
    episode_length: int = 0
    total_reward: float = 0.0

    # Additional detailed metrics
    successful_exits: int = 0
    failed_exits: int = 0
    total_aircraft: int = 0
    total_commands: int = 0

    def __str__(self) -> str:
        """Human-readable string representation."""
        return (
            f"ATCMetrics(\n"
            f"  Success Rate: {self.success_rate:.2%}\n"
            f"  Violations: {self.separation_violations}\n"
            f"  Collisions: {self.collisions}\n"
            f"  Throughput: {self.throughput:.2f} aircraft/hour\n"
            f"  Avg Exit Time: {self.avg_exit_time:.1f}s\n"
            f"  Command Efficiency: {self.command_efficiency:.2f} commands/aircraft\n"
            f"  Episode Length: {self.episode_length} steps\n"
            f"  Total Reward: {self.total_reward:.2f}\n"
            f")"
        )


class MetricsTracker:
    """
    Track and aggregate metrics across multiple episodes.

    Example:
        >>> tracker = MetricsTracker()
        >>> for episode in range(100):
        ...     obs, info = env.reset()
        ...     tracker.start_episode()
        ...     while not done:
        ...         action = agent.act(obs)
        ...         obs, reward, done, info = env.step(action)
        ...         tracker.update(reward, info)
        ...     tracker.end_episode(info)
        >>> summary = tracker.get_summary()
        >>> print(f"Avg Success Rate: {summary['avg_success_rate']:.2%}")
    """

    def __init__(self):
        self.episode_metrics: List[ATCMetrics] = []
        self.current_episode: Dict[str, Any] = {}
        self._reset_current()

    def _reset_current(self) -> None:
        """Reset current episode tracking."""
        self.current_episode = {
            "step": 0,
            "total_reward": 0.0,
            "commands_issued": 0,
            "violations": 0,
            "collisions": 0,
        }

    def start_episode(self) -> None:
        """Start tracking a new episode."""
        self._reset_current()

    def update(self, reward: float, info: Dict[str, Any]) -> None:
        """
        Update metrics for current step.

        Args:
            reward: Step reward
            info: Step info dictionary from environment
        """
        self.current_episode["step"] += 1
        self.current_episode["total_reward"] += reward

        # Track violations/collisions
        if "violations" in info:
            self.current_episode["violations"] = info["violations"]
        if "collisions" in info:
            self.current_episode["collisions"] = info["collisions"]

    def end_episode(self, final_info: Dict[str, Any]) -> ATCMetrics:
        """
        End current episode and compute metrics.

        Args:
            final_info: Final info dictionary from environment

        Returns:
            Computed metrics for the episode
        """
        # Extract metrics from final info
        successful_exits = final_info.get("successful_exits", 0)
        failed_exits = final_info.get("failed_exits", 0)
        total_aircraft = final_info.get("total_aircraft_spawned", 0)
        violations = final_info.get("separations_lost", 0)
        collisions = final_info.get("collided_aircraft", 0)
        episode_length = self.current_episode["step"]

        # Compute derived metrics
        success_rate = successful_exits / max(total_aircraft, 1)

        # Throughput: aircraft per hour (assuming 1 step = 1 second)
        episode_hours = episode_length / 3600.0
        throughput = successful_exits / max(episode_hours, 0.01)

        # Command efficiency (if available)
        command_efficiency = 0.0
        if "total_commands" in final_info and total_aircraft > 0:
            command_efficiency = final_info["total_commands"] / total_aircraft

        # Average exit time (if available)
        avg_exit_time = 0.0
        if "avg_exit_time" in final_info:
            avg_exit_time = final_info["avg_exit_time"]

        # Create metrics object
        metrics = ATCMetrics(
            success_rate=success_rate,
            separation_violations=violations,
            collisions=collisions,
            throughput=throughput,
            avg_exit_time=avg_exit_time,
            command_efficiency=command_efficiency,
            episode_length=episode_length,
            total_reward=self.current_episode["total_reward"],
            successful_exits=successful_exits,
            failed_exits=failed_exits,
            total_aircraft=total_aircraft,
        )

        self.episode_metrics.append(metrics)
        return metrics

    def get_summary(self) -> Dict[str, float]:
        """
        Get summary statistics across all tracked episodes.

        Returns:
            Dictionary of aggregated metrics
        """
        if not self.episode_metrics:
            return {}

        metrics_dict = defaultdict(list)

        for metrics in self.episode_metrics:
            metrics_dict["success_rate"].append(metrics.success_rate)
            metrics_dict["separation_violations"].append(metrics.separation_violations)
            metrics_dict["collisions"].append(metrics.collisions)
            metrics_dict["throughput"].append(metrics.throughput)
            metrics_dict["avg_exit_time"].append(metrics.avg_exit_time)
            metrics_dict["command_efficiency"].append(metrics.command_efficiency)
            metrics_dict["episode_length"].append(metrics.episode_length)
            metrics_dict["total_reward"].append(metrics.total_reward)

        # Compute mean and std for each metric
        summary = {}
        for key, values in metrics_dict.items():
            summary[f"avg_{key}"] = np.mean(values)
            summary[f"std_{key}"] = np.std(values)
            summary[f"min_{key}"] = np.min(values)
            summary[f"max_{key}"] = np.max(values)

        summary["num_episodes"] = len(self.episode_metrics)

        return summary

    def print_summary(self) -> None:
        """Print human-readable summary of metrics."""
        summary = self.get_summary()

        if not summary:
            print("No episodes tracked yet.")
            return

        print("\n" + "="*80)
        print(f"METRICS SUMMARY ({summary['num_episodes']} episodes)")
        print("="*80)

        print(f"\nSuccess Rate:        {summary['avg_success_rate']:.2%} ± {summary['std_success_rate']:.2%}")
        print(f"Violations:          {summary['avg_separation_violations']:.2f} ± {summary['std_separation_violations']:.2f}")
        print(f"Collisions:          {summary['avg_collisions']:.2f} ± {summary['std_collisions']:.2f}")
        print(f"Throughput:          {summary['avg_throughput']:.2f} ± {summary['std_throughput']:.2f} aircraft/hour")
        print(f"Avg Exit Time:       {summary['avg_avg_exit_time']:.1f} ± {summary['std_avg_exit_time']:.1f}s")
        print(f"Command Efficiency:  {summary['avg_command_efficiency']:.2f} ± {summary['std_command_efficiency']:.2f} commands/aircraft")
        print(f"Episode Length:      {summary['avg_episode_length']:.0f} ± {summary['std_episode_length']:.0f} steps")
        print(f"Total Reward:        {summary['avg_total_reward']:.2f} ± {summary['std_total_reward']:.2f}")

        print("\n" + "="*80)

    def get_episode_metrics(self, episode_idx: int) -> Optional[ATCMetrics]:
        """
        Get metrics for a specific episode.

        Args:
            episode_idx: Episode index

        Returns:
            Metrics for the episode or None if index out of range
        """
        if 0 <= episode_idx < len(self.episode_metrics):
            return self.episode_metrics[episode_idx]
        return None

    def reset(self) -> None:
        """Reset all tracked metrics."""
        self.episode_metrics.clear()
        self._reset_current()
