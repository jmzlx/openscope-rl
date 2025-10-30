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
from .action_masking import (
    get_action_mask,
    create_action_mask_fn,
    ActionMaskingWrapper,
    get_invalid_action_stats,
    print_action_mask_summary,
)
from .action_space_wrapper import DictToMultiDiscreteWrapper
from .utils import get_device
from .interface import (
    ATCEnvironmentInterface,
    ATCStateExtractor,
    ATCCommandExecutor,
    OpenScopeAdapter,
    MockATCEnvironment,
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
    
    # Action masking
    "get_action_mask",
    "create_action_mask_fn",
    "ActionMaskingWrapper",
    "get_invalid_action_stats",
    "print_action_mask_summary",
    
    # Action space conversion
    "DictToMultiDiscreteWrapper",
    
    # Utilities
    "get_device",
    
    # Interface abstraction
    "ATCEnvironmentInterface",
    "ATCStateExtractor",
    "ATCCommandExecutor",
    "OpenScopeAdapter",
    "MockATCEnvironment",
]

# Version info
__version__ = "0.1.0"