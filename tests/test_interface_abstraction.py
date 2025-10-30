"""
Tests for the interface abstraction layer.

These tests verify that:
- Interface abstraction works correctly
- OpenScopeAdapter wraps PlaywrightEnv properly
- MockATCEnvironment works for testing models
- Models can work with any interface implementation
"""

import pytest
import numpy as np
import torch
from typing import Dict, Any

from environment.interface import (
    ATCEnvironmentInterface,
    OpenScopeAdapter,
    MockATCEnvironment,
)
from models.networks import ATCActorCritic
from models.config import create_default_network_config


class TestInterfaceAbstraction:
    """Tests for interface abstraction."""
    
    def test_mock_environment_implements_interface(self):
        """Test that MockATCEnvironment implements the interface."""
        env = MockATCEnvironment(max_aircraft=20)
        
        assert isinstance(env, ATCEnvironmentInterface)
        
        # Test reset
        obs, info = env.reset()
        assert "aircraft" in obs
        assert "aircraft_mask" in obs
        assert "global_state" in obs
        assert "conflict_matrix" in obs
        
        # Test step
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert "raw_state" in info
        
        # Test close
        env.close()  # Should not raise
    
    def test_open_scope_adapter(self):
        """Test OpenScopeAdapter wraps environment correctly."""
        # Create a mock PlaywrightEnv-like object
        class MockPlaywrightEnv:
            def __init__(self):
                from environment.spaces import create_observation_space, create_action_space
                self.observation_space = create_observation_space(20)
                self.action_space = create_action_space(20)
            
            def reset(self, seed=None, options=None):
                obs = {
                    "aircraft": np.zeros((20, 14), dtype=np.float32),
                    "aircraft_mask": np.zeros(20, dtype=bool),
                    "global_state": np.zeros(4, dtype=np.float32),
                    "conflict_matrix": np.zeros((20, 20), dtype=np.float32),
                }
                return obs, {}
            
            def step(self, action):
                obs = {
                    "aircraft": np.zeros((20, 14), dtype=np.float32),
                    "aircraft_mask": np.zeros(20, dtype=bool),
                    "global_state": np.zeros(4, dtype=np.float32),
                    "conflict_matrix": np.zeros((20, 20), dtype=np.float32),
                }
                return obs, 0.0, False, False, {}
            
            def close(self):
                pass
        
        mock_env = MockPlaywrightEnv()
        adapter = OpenScopeAdapter(mock_env)
        
        assert isinstance(adapter, ATCEnvironmentInterface)
        assert adapter.get_observation_space() == mock_env.observation_space
        assert adapter.get_action_space() == mock_env.action_space
        
        obs, info = adapter.reset()
        assert "aircraft" in obs
        
        action = adapter.get_action_space().sample()
        obs, reward, terminated, truncated, info = adapter.step(action)
        
        adapter.close()


class TestModelWithInterface:
    """Tests for using models with interface abstraction."""
    
    def test_model_works_with_mock_environment(self):
        """Test that models work with MockATCEnvironment."""
        # Create mock environment
        env = MockATCEnvironment(max_aircraft=20)
        
        # Create model
        config = create_default_network_config(max_aircraft=20)
        model = ATCActorCritic(config)
        model.eval()
        
        # Reset environment
        obs, info = env.reset()
        
        # Convert observation to model input
        model_input = {
            "aircraft": torch.from_numpy(obs["aircraft"]).unsqueeze(0).float(),
            "aircraft_mask": torch.from_numpy(obs["aircraft_mask"]).unsqueeze(0).bool(),
            "global_state": torch.from_numpy(obs["global_state"]).unsqueeze(0).float(),
            "conflict_matrix": torch.from_numpy(obs["conflict_matrix"]).unsqueeze(0).float(),
        }
        
        # Forward pass
        with torch.no_grad():
            action_logits, value = model(model_input)
        
        # Sample action
        with torch.no_grad():
            action, log_prob, entropy, value = model.get_action_and_value(model_input)
        
        # Convert action to environment format
        env_action = {
            "aircraft_id": action["aircraft_id"].item(),
            "command_type": action["command_type"].item(),
            "altitude": action["altitude"].item(),
            "heading": action["heading"].item(),
            "speed": action["speed"].item(),
        }
        
        # Step environment
        next_obs, reward, terminated, truncated, next_info = env.step(env_action)
        
        assert "aircraft" in next_obs
        assert isinstance(reward, float)
    
    def test_interface_separation(self):
        """Test that interface properly separates concerns."""
        # Models should only depend on ATCEnvironmentInterface, not PlaywrightEnv
        env = MockATCEnvironment(max_aircraft=20)
        
        # Verify we can swap implementations
        assert isinstance(env, ATCEnvironmentInterface)
        
        # Test that interface methods work
        obs_space = env.get_observation_space()
        action_space = env.get_action_space()
        
        assert obs_space is not None
        assert action_space is not None
        
        # Test environment interaction
        obs, info = env.reset()
        action = action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        # All should work without knowing implementation details
        assert obs is not None
        assert isinstance(reward, float)

