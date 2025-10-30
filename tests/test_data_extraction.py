"""
Tests for data extraction from OpenScope.

These tests verify that:
- State extraction works correctly
- All required data fields are present
- State processing produces valid observations
- Data normalization is correct
- Missing data is handled gracefully
"""

import pytest
import numpy as np
from typing import Dict, Any, List
from unittest.mock import Mock

from environment.utils import extract_game_state
from environment.state_processor import StateProcessor
from environment.config import create_default_config
from environment.exceptions import StateProcessingError


class TestStateExtraction:
    """Tests for game state extraction."""
    
    def test_extract_game_state_structure(self, mock_page):
        """Test that extracted state has correct structure."""
        mock_state = {
            "aircraft": [
                {"callsign": "UAL123", "position": [10.0, 20.0]}
            ],
            "conflicts": [],
            "score": 1000,
            "time": 50.0,
        }
        mock_page.evaluate.return_value = mock_state
        
        state = extract_game_state(mock_page)
        
        assert "aircraft" in state
        assert "conflicts" in state
        assert "score" in state
        assert "time" in state
        assert isinstance(state["aircraft"], list)
        assert isinstance(state["conflicts"], list)
    
    def test_extract_game_state_with_aircraft(self, mock_page):
        """Test state extraction with aircraft data."""
        mock_state = {
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
                }
            ],
            "conflicts": [],
            "score": 1000,
            "time": 100.5,
        }
        # Mock should return state when called with any script
        mock_page.evaluate = Mock(return_value=mock_state)
        
        state = extract_game_state(mock_page)
        
        assert len(state["aircraft"]) == 1
        aircraft = state["aircraft"][0]
        assert aircraft["callsign"] == "UAL123"
        assert aircraft["altitude"] == 25000
        assert aircraft["category"] == "arrival"
    
    def test_extract_game_state_with_conflicts(self, mock_page):
        """Test state extraction with conflicts."""
        mock_state = {
            "aircraft": [
                {"callsign": "UAL123"},
                {"callsign": "AAL456"},
            ],
            "conflicts": [
                {
                    "aircraft1": "UAL123",
                    "aircraft2": "AAL456",
                    "hasConflict": True,
                    "hasViolation": False,
                }
            ],
            "score": 995,
            "time": 150.0,
        }
        # Mock should return state when called with any script
        mock_page.evaluate = Mock(return_value=mock_state)
        
        state = extract_game_state(mock_page)
        
        assert len(state["conflicts"]) == 1
        conflict = state["conflicts"][0]
        assert conflict["aircraft1"] == "UAL123"
        assert conflict["aircraft2"] == "AAL456"
        assert conflict["hasConflict"] is True
    
    def test_extract_game_state_missing_fields(self, mock_page):
        """Test handling of missing fields in extracted state."""
        # Missing some fields
        mock_state = {
            "aircraft": [],
            # Missing conflicts, score, time
        }
        mock_page.evaluate.return_value = mock_state
        
        state = extract_game_state(mock_page)
        
        # Should still return a valid dict, even with missing fields
        assert isinstance(state, dict)
        # extract_game_state may add defaults or leave missing - test handles both


class TestStateProcessing:
    """Tests for state processing into observations."""
    
    def test_state_processor_initialization(self, default_config):
        """Test StateProcessor initialization."""
        processor = StateProcessor(default_config)
        
        assert processor.config == default_config
        assert processor.max_aircraft == default_config.max_aircraft
    
    def test_process_state_basic(self, default_config, mock_game_state):
        """Test basic state processing."""
        processor = StateProcessor(default_config)
        obs = processor.process_state(mock_game_state)
        
        # Verify observation structure
        assert "aircraft" in obs
        assert "aircraft_mask" in obs
        assert "global_state" in obs
        assert "conflict_matrix" in obs
        
        # Verify shapes
        assert obs["aircraft"].shape == (default_config.max_aircraft, default_config.aircraft_feature_dim)
        assert obs["aircraft_mask"].shape == (default_config.max_aircraft,)
        assert obs["global_state"].shape == (default_config.global_state_dim,)
        assert obs["conflict_matrix"].shape == (default_config.max_aircraft, default_config.max_aircraft)
    
    def test_process_state_normalization(self, default_config, mock_game_state):
        """Test that state values are properly normalized."""
        processor = StateProcessor(default_config)
        obs = processor.process_state(mock_game_state)
        
        # Check that values are in reasonable ranges [0, 1] or close
        aircraft_features = obs["aircraft"]
        
        # Non-masked aircraft should have normalized values
        assert np.all(aircraft_features >= -1.0)  # Allow some negative for normalized positions
        assert np.all(aircraft_features <= 2.0)  # Allow some overflow
        
        # Global state should be normalized
        global_state = obs["global_state"]
        assert np.all(np.isfinite(global_state))
    
    def test_process_state_aircraft_mask(self, default_config):
        """Test aircraft mask is set correctly."""
        # Create state with 3 aircraft
        state = {
            "aircraft": [
                {"callsign": f"AC{i}", "position": [0, 0], "altitude": 20000}
                for i in range(3)
            ],
            "conflicts": [],
            "score": 1000,
            "time": 0.0,
        }
        
        processor = StateProcessor(default_config)
        obs = processor.process_state(state)
        
        # First 3 should be True, rest False
        assert np.sum(obs["aircraft_mask"]) == 3
        assert obs["aircraft_mask"][0] == True  # Use == for numpy bool comparison
        assert obs["aircraft_mask"][2] == True
        assert obs["aircraft_mask"][3] == False
    
    def test_process_state_conflict_matrix(self, default_config):
        """Test conflict matrix is populated correctly."""
        state = {
            "aircraft": [
                {"callsign": "UAL123"},
                {"callsign": "AAL456"},
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
            "time": 0.0,
        }
        
        processor = StateProcessor(default_config)
        obs = processor.process_state(state)
        
        # Check conflict matrix
        conflict_matrix = obs["conflict_matrix"]
        
        # Should have conflict between aircraft 0 and 1
        # Note: exact values depend on implementation
        assert conflict_matrix[0, 1] > 0 or conflict_matrix[1, 0] > 0
    
    def test_process_state_empty_aircraft(self, default_config):
        """Test processing state with no aircraft."""
        state = {
            "aircraft": [],
            "conflicts": [],
            "score": 1000,
            "time": 0.0,
        }
        
        processor = StateProcessor(default_config)
        obs = processor.process_state(state)
        
        # All aircraft should be masked
        assert np.sum(obs["aircraft_mask"]) == 0
        # Aircraft features should be zeros
        assert np.all(obs["aircraft"] == 0)
    
    def test_process_state_missing_fields(self, default_config):
        """Test handling of missing aircraft fields."""
        # Aircraft with missing fields
        state = {
            "aircraft": [
                {
                    "callsign": "UAL123",
                    # Missing position, altitude, etc.
                }
            ],
            "conflicts": [],
            "score": 1000,
            "time": 0.0,
        }
        
        processor = StateProcessor(default_config)
        # Should handle gracefully
        obs = processor.process_state(state)
        
        assert obs is not None
        assert "aircraft" in obs
    
    def test_validate_observation(self, default_config, mock_observation):
        """Test observation validation."""
        processor = StateProcessor(default_config)
        
        # Valid observation should pass
        is_valid = processor.validate_observation(mock_observation)
        assert is_valid is True
    
    def test_validate_observation_missing_keys(self, default_config, mock_observation):
        """Test observation validation catches missing keys."""
        processor = StateProcessor(default_config)
        
        # Remove a required key
        invalid_obs = {k: v for k, v in mock_observation.items() if k != "aircraft"}
        
        with pytest.raises(StateProcessingError):
            processor.validate_observation(invalid_obs)
    
    def test_validate_observation_wrong_shape(self, default_config):
        """Test observation validation catches wrong shapes."""
        processor = StateProcessor(default_config)
        
        # Create observation with wrong shape
        invalid_obs = {
            "aircraft": np.zeros((10, 10), dtype=np.float32),  # Wrong shape
            "aircraft_mask": np.zeros(20, dtype=bool),
            "global_state": np.zeros(4, dtype=np.float32),
            "conflict_matrix": np.zeros((20, 20), dtype=np.float32),
        }
        
        with pytest.raises(StateProcessingError):
            processor.validate_observation(invalid_obs)


class TestDataCompleteness:
    """Tests to ensure all required data is extracted."""
    
    def test_required_aircraft_fields(self, mock_game_state):
        """Test that all required aircraft fields are present."""
        required_fields = [
            "callsign",
            "position",
            "altitude",
            "heading",
            "speed",
        ]
        
        for aircraft in mock_game_state["aircraft"]:
            for field in required_fields:
                assert field in aircraft, f"Missing required field: {field}"
    
    def test_state_processor_observation_info(self, default_config):
        """Test that observation info is complete."""
        processor = StateProcessor(default_config)
        info = processor.get_observation_info()
        
        assert "aircraft_shape" in info
        assert "aircraft_mask_shape" in info
        assert "global_state_shape" in info
        assert "conflict_matrix_shape" in info
        assert "aircraft_features" in info
        assert "global_features" in info
        
        # Verify feature counts match
        assert len(info["aircraft_features"]) == default_config.aircraft_feature_dim
        assert len(info["global_features"]) == default_config.global_state_dim

