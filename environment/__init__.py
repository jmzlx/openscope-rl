"""
Environment module for OpenScope RL.

This module provides the main environment class and supporting components
for reinforcement learning with the OpenScope air traffic control simulator.
"""

from .playwright_env import PlaywrightEnv
from .config import (
    OpenScopeConfig,
    RewardConfig,
    ActionConfig,
    BrowserConfig,
    create_default_config,
    validate_config,
)
from .exceptions import (
    OpenScopeError,
    BrowserError,
    GameInterfaceError,
    StateProcessingError,
    RewardCalculationError,
    ConfigurationError,
    ActionError,
    EpisodeError,
)

# Public API
__all__ = [
    # Main environment
    "PlaywrightEnv",
    
    # Configuration
    "OpenScopeConfig",
    "RewardConfig", 
    "ActionConfig",
    "BrowserConfig",
    "create_default_config",
    "validate_config",
    
    # Exceptions
    "OpenScopeError",
    "BrowserError",
    "GameInterfaceError", 
    "StateProcessingError",
    "RewardCalculationError",
    "ConfigurationError",
    "ActionError",
    "EpisodeError",
]

# Version info
__version__ = "0.1.0"