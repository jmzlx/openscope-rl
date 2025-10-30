"""
Tests for model data requirements.

These tests verify that:
- Observations are compatible with model inputs
- All required data fields are present for models
- Data types and shapes match model expectations
- Models can process observations correctly
"""

import pytest
import numpy as np
import torch
from typing import Dict, Any

from models.networks import ATCActorCritic
from models.config import create_default_network_config
from environment.state_processor import StateProcessor
from environment.config import create_default_config
from environment.exceptions import StateProcessingError


class TestModelObservationCompatibility:
    """Tests for compatibility between observations and models."""
    
    def test_observation_to_model_input_conversion(self, mock_observation):
        """Test converting observation to model input format."""
        # Model expects torch tensors with batch dimension
        aircraft_tensor = torch.from_numpy(mock_observation["aircraft"]).unsqueeze(0)
        mask_tensor = torch.from_numpy(mock_observation["aircraft_mask"]).unsqueeze(0)
        global_tensor = torch.from_numpy(mock_observation["global_state"]).unsqueeze(0)
        conflict_tensor = torch.from_numpy(mock_observation["conflict_matrix"]).unsqueeze(0)
        
        model_input = {
            "aircraft": aircraft_tensor,
            "aircraft_mask": mask_tensor,
            "global_state": global_tensor,
            "conflict_matrix": conflict_tensor,
        }
        
        # Verify shapes
        batch_size = 1
        max_aircraft = mock_observation["aircraft"].shape[0]
        assert model_input["aircraft"].shape == (batch_size, max_aircraft, 14)
        assert model_input["aircraft_mask"].shape == (batch_size, max_aircraft)
        assert model_input["global_state"].shape == (batch_size, 4)
    
    def test_model_forward_pass(self, mock_observation):
        """Test that model can process a valid observation."""
        config = create_default_network_config(max_aircraft=20)
        model = ATCActorCritic(config)
        model.eval()
        
        # Convert observation to model input
        model_input = {
            "aircraft": torch.from_numpy(mock_observation["aircraft"]).unsqueeze(0),
            "aircraft_mask": torch.from_numpy(mock_observation["aircraft_mask"]).unsqueeze(0),
            "global_state": torch.from_numpy(mock_observation["global_state"]).unsqueeze(0),
            "conflict_matrix": torch.from_numpy(mock_observation["conflict_matrix"]).unsqueeze(0),
        }
        
        # Forward pass
        with torch.no_grad():
            action_logits, value = model(model_input)
        
        # Verify output structure
        assert isinstance(action_logits, dict)
        assert "aircraft_id" in action_logits
        assert "command_type" in action_logits
        assert "altitude" in action_logits
        assert "heading" in action_logits
        assert "speed" in action_logits
        
        # Verify value output
        assert value.shape == (1, 1)  # (batch_size, 1)
    
    def test_model_action_sampling(self, mock_observation):
        """Test that model can sample actions from observation."""
        config = create_default_network_config(max_aircraft=20)
        model = ATCActorCritic(config)
        model.eval()
        
        model_input = {
            "aircraft": torch.from_numpy(mock_observation["aircraft"]).unsqueeze(0),
            "aircraft_mask": torch.from_numpy(mock_observation["aircraft_mask"]).unsqueeze(0),
            "global_state": torch.from_numpy(mock_observation["global_state"]).unsqueeze(0),
            "conflict_matrix": torch.from_numpy(mock_observation["conflict_matrix"]).unsqueeze(0),
        }
        
        with torch.no_grad():
            action, log_prob, entropy, value = model.get_action_and_value(model_input)
        
        # Verify action structure
        assert isinstance(action, dict)
        assert "aircraft_id" in action
        assert "command_type" in action
        assert "altitude" in action
        assert "heading" in action
        assert "speed" in action
        
        # Verify action values are in valid ranges
        assert 0 <= action["aircraft_id"].item() <= 20
        assert 0 <= action["command_type"].item() < 5
        assert 0 <= action["altitude"].item() < 18
        assert 0 <= action["heading"].item() < 13
        assert 0 <= action["speed"].item() < 8


class TestObservationDataCompleteness:
    """Tests to ensure observations contain all data required by models."""
    
    def test_required_observation_keys(self, mock_observation):
        """Test that observation has all required keys for models."""
        required_keys = ["aircraft", "aircraft_mask", "global_state", "conflict_matrix"]
        
        for key in required_keys:
            assert key in mock_observation, f"Missing required observation key: {key}"
    
    def test_observation_data_types(self, mock_observation):
        """Test that observation data types match model expectations."""
        # Aircraft features should be float32
        assert mock_observation["aircraft"].dtype == np.float32
        
        # Aircraft mask should be bool
        assert mock_observation["aircraft_mask"].dtype == np.bool_ or mock_observation["aircraft_mask"].dtype == bool
        
        # Global state should be float32
        assert mock_observation["global_state"].dtype == np.float32
        
        # Conflict matrix should be float32
        assert mock_observation["conflict_matrix"].dtype == np.float32
    
    def test_observation_shapes(self, mock_observation):
        """Test that observation shapes match model expectations."""
        max_aircraft = 20
        aircraft_feature_dim = 14
        global_state_dim = 4
        
        assert mock_observation["aircraft"].shape == (max_aircraft, aircraft_feature_dim)
        assert mock_observation["aircraft_mask"].shape == (max_aircraft,)
        assert mock_observation["global_state"].shape == (global_state_dim,)
        assert mock_observation["conflict_matrix"].shape == (max_aircraft, max_aircraft)
    
    def test_observation_no_nan_or_inf(self, mock_observation):
        """Test that observation doesn't contain NaN or Inf values."""
        assert np.all(np.isfinite(mock_observation["aircraft"]))
        assert np.all(np.isfinite(mock_observation["global_state"]))
        assert np.all(np.isfinite(mock_observation["conflict_matrix"]))


class TestEndToEndDataFlow:
    """Tests for complete data flow from state to model."""
    
    def test_state_to_model_pipeline(self, default_config, mock_game_state):
        """Test complete pipeline from game state to model input."""
        # Process state to observation
        processor = StateProcessor(default_config)
        obs = processor.process_state(mock_game_state)
        
        # Validate observation
        processor.validate_observation(obs)
        
        # Convert to model input
        model_input = {
            "aircraft": torch.from_numpy(obs["aircraft"]).unsqueeze(0).float(),
            "aircraft_mask": torch.from_numpy(obs["aircraft_mask"]).unsqueeze(0).bool(),
            "global_state": torch.from_numpy(obs["global_state"]).unsqueeze(0).float(),
            "conflict_matrix": torch.from_numpy(obs["conflict_matrix"]).unsqueeze(0).float(),
        }
        
        # Create model and process
        model_config = create_default_network_config(max_aircraft=default_config.max_aircraft)
        model = ATCActorCritic(model_config)
        model.eval()
        
        with torch.no_grad():
            action_logits, value = model(model_input)
        
        # Verify outputs
        assert action_logits is not None
        assert value is not None
    
    def test_batch_processing(self, default_config):
        """Test processing multiple observations in a batch."""
        processor = StateProcessor(default_config)
        model_config = create_default_network_config(max_aircraft=default_config.max_aircraft)
        model = ATCActorCritic(model_config)
        model.eval()
        
        # Create batch of observations
        batch_size = 4
        observations = []
        
        for i in range(batch_size):
            state = {
                "aircraft": [
                    {
                        "callsign": f"AC{i}",
                        "position": [i * 10, i * 5],
                        "altitude": 20000 + i * 1000,
                        "heading": i * 30,
                        "speed": 250 + i * 10,
                        "groundSpeed": 250 + i * 10,
                        "assignedAltitude": 20000 + i * 1000,
                        "assignedHeading": i * 30,
                        "assignedSpeed": 250 + i * 10,
                        "isOnGround": False,
                        "isTaxiing": False,
                        "isEstablished": True,
                        "category": "arrival" if i % 2 == 0 else "departure",
                    }
                ],
                "conflicts": [],
                "score": 1000,
                "time": i * 10.0,
            }
            obs = processor.process_state(state)
            observations.append(obs)
        
        # Stack into batch
        batch_input = {
            "aircraft": torch.stack([
                torch.from_numpy(obs["aircraft"]).float()
                for obs in observations
            ]),
            "aircraft_mask": torch.stack([
                torch.from_numpy(obs["aircraft_mask"]).bool()
                for obs in observations
            ]),
            "global_state": torch.stack([
                torch.from_numpy(obs["global_state"]).float()
                for obs in observations
            ]),
            "conflict_matrix": torch.stack([
                torch.from_numpy(obs["conflict_matrix"]).float()
                for obs in observations
            ]),
        }
        
        # Process batch
        with torch.no_grad():
            action_logits, value = model(batch_input)
        
        # Verify batch outputs
        assert value.shape[0] == batch_size
        assert action_logits["aircraft_id"].shape[0] == batch_size


class TestModelDataValidation:
    """Tests for validating data passed to models."""
    
    def test_model_observation_validation(self):
        """Test that model validates observation structure."""
        config = create_default_network_config(max_aircraft=20)
        model = ATCActorCritic(config)
        
        # Missing key should raise error
        invalid_input = {
            "aircraft": torch.randn(1, 20, 14),
            "aircraft_mask": torch.ones(1, 20, dtype=torch.bool),
            # Missing global_state
        }
        
        with pytest.raises((ValueError, KeyError)):
            model(invalid_input)
    
    def test_model_shape_validation(self):
        """Test that model validates observation shapes."""
        config = create_default_network_config(max_aircraft=20)
        model = ATCActorCritic(config)
        
        # Wrong shape should raise error
        invalid_input = {
            "aircraft": torch.randn(1, 10, 14),  # Wrong max_aircraft
            "aircraft_mask": torch.ones(1, 20, dtype=torch.bool),
            "global_state": torch.randn(1, 4),
            "conflict_matrix": torch.randn(1, 20, 20),
        }
        
        with pytest.raises((ValueError, RuntimeError)):
            model(invalid_input)

