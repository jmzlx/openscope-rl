"""
Configuration module for OpenScope RL environment.

This module defines dataclasses for all configuration options,
providing type safety and validation for environment parameters.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from enum import Enum


class CommandType(Enum):
    """Available command types for aircraft."""
    ALTITUDE = "altitude"
    HEADING = "heading"
    SPEED = "speed"
    ILS = "ils"
    DIRECT = "direct"


class AircraftCategory(Enum):
    """Aircraft categories."""
    ARRIVAL = "arrival"
    DEPARTURE = "departure"


@dataclass
class RewardConfig:
    """Configuration for reward calculation."""
    
    # Base rewards (enhanced for denser feedback)
    timestep_penalty: float = -0.1  # Was -0.01, make more significant
    action_reward: float = 0.0
    
    # Safety rewards
    separation_loss: float = -200.0
    conflict_warning: float = -2.0
    safe_separation_bonus: float = 0.1  # Was 0.02, reward good behavior more
    
    # Performance rewards
    successful_exit_bonus: float = 50.0  # Was 100.0, balanced with penalties
    failed_exit_penalty: float = -20.0
    episode_termination_penalty: float = -100.0
    
    # Episode completion bonuses
    high_success_rate_bonus: float = 200.0
    medium_success_rate_bonus: float = 100.0
    low_success_rate_bonus: float = 50.0
    
    # Success rate thresholds
    high_success_threshold: float = 0.8
    medium_success_threshold: float = 0.6
    low_success_threshold: float = 0.4
    
    # Curriculum-specific (override per stage)
    min_score_threshold: float = -5000.0  # Termination threshold for curriculum stages


@dataclass
class ActionConfig:
    """Configuration for action space and mappings."""
    
    # Action space dimensions (reduced for efficiency)
    max_aircraft: int = 20
    command_type_count: int = 5
    altitude_levels: int = 9  # Was 18 (reduced 2x)
    heading_changes_count: int = 7  # Was 13 (reduced ~2x)
    speed_levels: int = 4  # Was 8 (reduced 2x)
    
    # Action mappings (coarser but still expressive)
    altitude_values: List[int] = field(default_factory=lambda: [
        0, 20, 40, 60, 80, 100, 120, 140, 160  # Every 20,000 ft (was every 10,000 ft)
    ])
    
    heading_changes: List[int] = field(default_factory=lambda: [
        -90, -45, -20, 0, 20, 45, 90  # Key heading changes (was 13 options)
    ])
    
    speed_values: List[int] = field(default_factory=lambda: [
        180, 220, 260, 300  # 40 knot increments (was 20 knot increments)
    ])
    
    command_types: List[CommandType] = field(default_factory=lambda: [
        CommandType.ALTITUDE, CommandType.HEADING, CommandType.SPEED, 
        CommandType.ILS, CommandType.DIRECT
    ])
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if len(self.altitude_values) != self.altitude_levels:
            raise ValueError(f"altitude_values length ({len(self.altitude_values)}) "
                           f"must match altitude_levels ({self.altitude_levels})")
        
        if len(self.heading_changes) != self.heading_changes_count:
            raise ValueError(f"heading_changes length ({len(self.heading_changes)}) "
                           f"must match heading_changes_count ({self.heading_changes_count})")
        
        if len(self.speed_values) != self.speed_levels:
            raise ValueError(f"speed_values length ({len(self.speed_values)}) "
                           f"must match speed_levels ({self.speed_levels})")
        
        if len(self.command_types) != self.command_type_count:
            raise ValueError(f"command_types length ({len(self.command_types)}) "
                           f"must match command_type_count ({self.command_type_count})")


@dataclass
class BrowserConfig:
    """Configuration for browser initialization."""
    
    headless: bool = True
    browser_args: List[str] = field(default_factory=lambda: [
        '--no-sandbox',
        '--disable-dev-shm-usage',
        '--disable-gpu',
        '--disable-web-security',
        '--disable-features=VizDisplayCompositor',
        '--memory-pressure-off',
        '--max_old_space_size=512'
    ])
    
    # Timing configuration
    page_load_timeout: float = 2.0
    airport_setup_delay: float = 2.0  # Reduced from 10.0 - game loads much faster
    timewarp_setup_delay: float = 0.3  # Reduced from 0.5
    game_update_delay: float = 0.02


@dataclass
class OpenScopeConfig:
    """Main configuration for OpenScope environment."""
    
    # Game settings
    game_url: str = "http://localhost:3003"
    airport: str = "KLAS"
    timewarp: int = 5
    
    # Episode settings
    episode_length: int = 3600
    action_interval: float = 5.0
    
    # Observation settings
    max_aircraft: int = 20
    aircraft_feature_dim: int = 14
    global_state_dim: int = 4
    
    # Component configurations
    reward_config: RewardConfig = field(default_factory=RewardConfig)
    action_config: ActionConfig = field(default_factory=ActionConfig)
    browser_config: BrowserConfig = field(default_factory=BrowserConfig)
    
    # Custom configuration
    custom_config: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate configuration consistency."""
        if self.max_aircraft != self.action_config.max_aircraft:
            self.action_config.max_aircraft = self.max_aircraft
        
        if self.episode_length <= 0:
            raise ValueError("episode_length must be positive")
        
        if self.action_interval <= 0:
            raise ValueError("action_interval must be positive")
        
        if self.max_aircraft <= 0:
            raise ValueError("max_aircraft must be positive")


def create_default_config(**overrides) -> OpenScopeConfig:
    """
    Create a default configuration with optional overrides.
    
    Args:
        **overrides: Configuration values to override
        
    Returns:
        OpenScopeConfig: Configured environment settings
        
    Example:
        >>> config = create_default_config(
        ...     airport="KJFK",
        ...     max_aircraft=10,
        ...     headless=False
        ... )
    """
    config = OpenScopeConfig()
    
    # Apply overrides
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
        elif hasattr(config.reward_config, key):
            setattr(config.reward_config, key, value)
        elif hasattr(config.action_config, key):
            setattr(config.action_config, key, value)
        elif hasattr(config.browser_config, key):
            setattr(config.browser_config, key, value)
        else:
            config.custom_config[key] = value
    
    return config


def validate_config(config: OpenScopeConfig) -> bool:
    """
    Validate configuration for consistency and correctness.
    
    Args:
        config: Configuration to validate
        
    Returns:
        bool: True if configuration is valid
        
    Raises:
        ValueError: If configuration is invalid
    """
    # Validate episode settings
    if config.episode_length <= 0:
        raise ValueError("episode_length must be positive")
    
    if config.action_interval <= 0:
        raise ValueError("action_interval must be positive")
    
    # Validate aircraft settings
    if config.max_aircraft <= 0:
        raise ValueError("max_aircraft must be positive")
    
    if config.max_aircraft > 50:
        raise ValueError("max_aircraft should not exceed 50 for performance reasons")
    
    # Validate URL
    if not config.game_url.startswith(('http://', 'https://')):
        raise ValueError("game_url must be a valid HTTP/HTTPS URL")
    
    # Validate airport code
    if len(config.airport) != 4:
        raise ValueError("airport must be a 4-character ICAO code")
    
    # Validate timewarp
    if config.timewarp < 1 or config.timewarp > 2000:
        raise ValueError("timewarp must be between 1 and 2000")
    
    return True
