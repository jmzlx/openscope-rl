"""
Custom exceptions for OpenScope RL environment.

This module defines specific exception classes for different types of errors
that can occur in the environment, providing better error handling and debugging.
"""


class OpenScopeError(Exception):
    """Base exception for all OpenScope RL environment errors."""
    pass


class BrowserError(OpenScopeError):
    """Exception raised for browser-related errors."""
    pass


class GameInterfaceError(OpenScopeError):
    """Exception raised for game interface communication errors."""
    pass


class StateProcessingError(OpenScopeError):
    """Exception raised for state processing and observation conversion errors."""
    pass


class RewardCalculationError(OpenScopeError):
    """Exception raised for reward calculation errors."""
    pass


class ConfigurationError(OpenScopeError):
    """Exception raised for configuration validation errors."""
    pass


class ActionError(OpenScopeError):
    """Exception raised for action execution errors."""
    pass


class EpisodeError(OpenScopeError):
    """Exception raised for episode management errors."""
    pass
