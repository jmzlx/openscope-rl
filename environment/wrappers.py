"""
Gymnasium wrappers for OpenScope environment
Provides normalization, monitoring, and other standard preprocessing
"""

from typing import Optional

import gymnasium as gym
from gymnasium.wrappers import (
    NormalizeObservation,
    NormalizeReward,
    RecordEpisodeStatistics,
)

from environment.openscope_env import OpenScopeEnv


def make_openscope_env(
    game_url: str = "http://localhost:3003",
    airport: str = "KLAS",
    timewarp: int = 5,
    max_aircraft: int = 20,
    episode_length: int = 3600,
    action_interval: float = 5.0,
    headless: bool = True,
    config: Optional[dict] = None,
    normalize_obs: bool = True,
    normalize_reward: bool = True,
    gamma: float = 0.99,
    record_stats: bool = True,
) -> gym.Env:
    """
    Create OpenScope environment with standard wrappers

    Args:
        game_url: URL of the openScope game server
        airport: Airport code (e.g., "KLAS")
        timewarp: Game speed multiplier (1-10)
        max_aircraft: Maximum number of aircraft
        episode_length: Episode duration in game seconds
        action_interval: Command interval in seconds
        headless: Run browser in headless mode
        config: Full configuration dict (optional)
        normalize_obs: Apply observation normalization (running mean/std)
        normalize_reward: Apply reward normalization (running mean/std)
        gamma: Discount factor for reward normalization
        record_stats: Record episode statistics (return, length)

    Returns:
        Wrapped environment
    """
    # Create base environment
    env = OpenScopeEnv(
        game_url=game_url,
        airport=airport,
        timewarp=timewarp,
        max_aircraft=max_aircraft,
        episode_length=episode_length,
        action_interval=action_interval,
        headless=headless,
        config=config,
    )

    # Add episode statistics tracking
    if record_stats:
        env = RecordEpisodeStatistics(env)

    # Add observation normalization (running mean/std)
    if normalize_obs:
        env = NormalizeObservation(env)

    # Add reward normalization
    if normalize_reward:
        env = NormalizeReward(env, gamma=gamma)

    return env
