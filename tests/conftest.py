"""
Pytest configuration and shared fixtures for OpenScope RL tests.
"""

import pytest
import numpy as np
from typing import Dict, Any, List
from unittest.mock import MagicMock, Mock
import torch

from environment.config import create_default_config
from environment.spaces import create_observation_space, create_action_space


@pytest.fixture
def mock_game_state() -> Dict[str, Any]:
    """Create a mock game state for testing."""
    return {
        "aircraft": [
            {
                "callsign": "UAL123",
                "position": [10.5, -5.2],
                "altitude": 25000,
                "heading": 90.0,
                "speed": 300.0,
                "groundSpeed": 290.0,
                "assignedAltitude": 30000,
                "assignedHeading": 100.0,
                "assignedSpeed": 300.0,
                "isOnGround": False,
                "isTaxiing": False,
                "isEstablished": True,
                "category": "arrival",
            },
            {
                "callsign": "AAL456",
                "position": [-15.3, 8.7],
                "altitude": 35000,
                "heading": 270.0,
                "speed": 320.0,
                "groundSpeed": 310.0,
                "assignedAltitude": 35000,
                "assignedHeading": 270.0,
                "assignedSpeed": 320.0,
                "isOnGround": False,
                "isTaxiing": False,
                "isEstablished": False,
                "category": "departure",
            },
        ],
        "conflicts": [
            {
                "aircraft1": "UAL123",
                "aircraft2": "AAL456",
                "hasConflict": True,
                "hasViolation": False,
            }
        ],
        "score": 1000,
        "time": 100.5,
    }


@pytest.fixture
def mock_observation() -> Dict[str, np.ndarray]:
    """Create a mock observation for testing."""
    max_aircraft = 20
    aircraft_feature_dim = 14
    
    return {
        "aircraft": np.random.rand(max_aircraft, aircraft_feature_dim).astype(np.float32),
        "aircraft_mask": np.array([True] * 2 + [False] * (max_aircraft - 2), dtype=bool),
        "global_state": np.array([0.5, 0.1, 0.0, 1.0], dtype=np.float32),
        "conflict_matrix": np.zeros((max_aircraft, max_aircraft), dtype=np.float32),
    }


@pytest.fixture
def mock_action() -> Dict[str, int]:
    """Create a mock action for testing."""
    return {
        "aircraft_id": 0,
        "command_type": 0,
        "altitude": 10,
        "heading": 6,
        "speed": 4,
    }


@pytest.fixture
def test_config() -> Dict[str, Any]:
    """Create a test configuration."""
    return {
        "airport": "KLAS",
        "max_aircraft": 20,
        "timewarp": 5,
        "episode_length": 3600,
        "action_interval": 5.0,
        "headless": True,
    }


@pytest.fixture
def default_config():
    """Create a default OpenScopeConfig for testing."""
    return create_default_config(
        airport="KLAS",
        max_aircraft=20,
        timewarp=5,
        headless=True,
    )


@pytest.fixture
def mock_page():
    """Create a mock Playwright page for testing."""
    page = MagicMock()
    
    # Mock evaluate to return mock game state (for state extraction)
    def mock_evaluate(script, arg=None):
        # Return mock game state structure
        return {
            "aircraft": [],
            "conflicts": [],
            "score": 1000,
            "time": 0.0,
        }
    
    page.evaluate = Mock(side_effect=mock_evaluate)
    # Mock fill() and press() for command execution (Playwright native methods)
    page.fill = Mock()
    page.press = Mock()
    page.url = "http://localhost:3003"
    page.is_closed.return_value = False
    page.title.return_value = "OpenScope"
    
    return page


@pytest.fixture
def observation_space():
    """Create observation space for testing."""
    return create_observation_space(max_aircraft=20)


@pytest.fixture
def action_space():
    """Create action space for testing."""
    return create_action_space(max_aircraft=20)


@pytest.fixture
def mock_model_input(mock_observation):
    """Create a mock model input (converted from observation)."""
    return {
        "aircraft": torch.from_numpy(mock_observation["aircraft"]).unsqueeze(0),
        "aircraft_mask": torch.from_numpy(mock_observation["aircraft_mask"]).unsqueeze(0),
        "global_state": torch.from_numpy(mock_observation["global_state"]).unsqueeze(0),
        "conflict_matrix": torch.from_numpy(mock_observation["conflict_matrix"]).unsqueeze(0),
    }

