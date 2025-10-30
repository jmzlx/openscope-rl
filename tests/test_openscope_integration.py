"""
Tests for OpenScope integration.

These tests verify that:
- Browser management works correctly
- Game interface can connect to OpenScope
- Command execution works
- State extraction works
- The environment can be initialized and reset
"""

import pytest
import time
from unittest.mock import MagicMock, Mock, patch
from typing import Dict, Any

from environment.utils import BrowserManager
from environment.game_interface import OpenScopeInterface
from environment.exceptions import BrowserError, GameInterfaceError
from environment.config import BrowserConfig, create_default_config


class TestBrowserManager:
    """Tests for BrowserManager."""
    
    def test_browser_manager_initialization(self, default_config):
        """Test that BrowserManager can be initialized."""
        browser_config = default_config.browser_config
        manager = BrowserManager(browser_config)
        
        assert manager.config == browser_config
        assert not manager.is_initialized
        assert manager.page is None
    
    def test_browser_initialization_integration(self, default_config):
        """Test that BrowserManager can actually initialize a browser."""
        browser_config = default_config.browser_config
        browser_config.headless = True
        
        manager = BrowserManager(browser_config)
        try:
            manager.initialize()
            assert manager.is_initialized
            assert manager.page is not None
        finally:
            manager.cleanup()
    
    def test_browser_manager_context_manager(self, default_config):
        """Test BrowserManager as context manager."""
        browser_config = default_config.browser_config
        
        # Note: This test doesn't actually start a browser
        # To test full integration, use the skipped test above
        with BrowserManager(browser_config) as manager:
            # In normal usage, browser would be initialized here
            assert manager.config == browser_config
    
    def test_navigate_to_game_url(self, mock_page):
        """Test navigation to game URL."""
        config = create_default_config()
        manager = BrowserManager(config.browser_config)
        manager.page = mock_page
        
        # Mock response object with status attribute
        mock_response = MagicMock()
        mock_response.status = 200
        mock_page.goto.return_value = mock_response
        mock_page.url = "http://localhost:3003"
        mock_page.title.return_value = "OpenScope"
        
        manager.navigate_to_game("http://localhost:3003")
        
        assert mock_page.goto.called
        assert mock_page.wait_for_load_state.called
    
    def test_browser_cleanup(self, default_config):
        """Test browser cleanup."""
        manager = BrowserManager(default_config.browser_config)
        
        # Mock browser components
        manager.playwright = MagicMock()
        manager.browser = MagicMock()
        manager.page = MagicMock()
        manager._is_initialized = True
        
        manager.cleanup()
        
        assert manager.page is None
        assert manager.browser is None
        assert manager.playwright is None
        assert not manager.is_initialized


class TestGameInterface:
    """Tests for OpenScopeInterface."""
    
    def test_game_interface_initialization(self, default_config):
        """Test that OpenScopeInterface can be initialized."""
        interface = OpenScopeInterface(default_config)
        
        assert interface.config == default_config
        assert not interface.is_initialized
        assert interface.page is None
    
    def test_game_interface_setup(self, default_config, mock_page):
        """Test game setup (airport and timewarp)."""
        interface = OpenScopeInterface(default_config)
        interface.initialize(mock_page)
        
        assert interface.is_initialized
        # Verify setup commands were executed via mock
        assert mock_page.evaluate.called
    
    def test_execute_action_no_op(self, default_config, mock_page):
        """Test executing a no-op action."""
        interface = OpenScopeInterface(default_config)
        interface.initialize(mock_page)
        
        # No-op action: aircraft_id == max_aircraft
        action = {
            "aircraft_id": default_config.max_aircraft,
            "command_type": 0,
            "altitude": 0,
            "heading": 0,
            "speed": 0,
        }
        
        result = interface.execute_action(action, [])
        assert result is None
    
    def test_execute_action_altitude(self, default_config, mock_page):
        """Test executing an altitude command."""
        interface = OpenScopeInterface(default_config)
        interface.initialize(mock_page)
        
        aircraft_data = [{
            "callsign": "UAL123",
            "altitude": 25000,
        }]
        
        action = {
            "aircraft_id": 0,
            "command_type": 0,  # ALTITUDE
            "altitude": 10,  # FL 100
            "heading": 0,
            "speed": 0,
        }
        
        # Mock the evaluate to track commands
        command_executed = []
        def capture_command(script, arg=None):
            if arg:
                command_executed.append(arg)
        
        mock_page.evaluate = Mock(side_effect=capture_command)
        
        result = interface.execute_action(action, aircraft_data)
        
        # Should execute a command like "UAL123 c 100"
        assert result is not None
        assert "UAL123" in result
        assert "c" in result
    
    def test_get_game_state(self, default_config, mock_page):
        """Test getting game state."""
        interface = OpenScopeInterface(default_config)
        interface.initialize(mock_page)
        
        # Mock state extraction
        mock_state = {
            "aircraft": [],
            "conflicts": [],
            "score": 1000,
            "time": 0.0,
        }
        mock_page.evaluate.return_value = mock_state
        
        state = interface.get_game_state()
        
        assert state == mock_state
        assert mock_page.evaluate.called
    
    def test_reset_game(self, default_config, mock_page):
        """Test resetting the game."""
        interface = OpenScopeInterface(default_config)
        interface.initialize(mock_page)
        
        interface.reset_game()
        
        # Should call clear and re-setup
        assert mock_page.evaluate.called
    
    def test_is_game_ready(self, default_config, mock_page):
        """Test game readiness check."""
        interface = OpenScopeInterface(default_config)
        interface.initialize(mock_page)
        
        # Mock game ready response
        mock_page.evaluate = Mock(return_value={
            "ready": True,
            "hasAircraftController": True,
            "hasAircraftList": True,
            "hasScoreElement": True,
        })
        
        is_ready = interface.is_game_ready()
        
        assert is_ready is True
        assert mock_page.evaluate.called


class TestOpenScopeIntegration:
    """Integration tests for full OpenScope environment."""
    
    def test_full_integration(self):
        """Test full integration with real OpenScope server."""
        # This test requires:
        # 1. OpenScope server running at localhost:3003
        # 2. Browser installation (playwright install chromium)
        from environment import PlaywrightEnv
        
        env = PlaywrightEnv(
            game_url="http://localhost:3003",
            airport="KLAS",
            max_aircraft=5,
            timewarp=1,
            headless=True,
            episode_length=60,
        )
        
        try:
            # Reset environment
            obs, info = env.reset()
            
            # Verify observation structure
            assert "aircraft" in obs
            assert "aircraft_mask" in obs
            assert "global_state" in obs
            assert "conflict_matrix" in obs
            
            # Take a random action
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Verify observation still valid
            assert "aircraft" in obs
            
            # Verify info contains expected keys
            assert "raw_state" in info
            
        finally:
            env.close()
    
    def test_environment_configuration(self, test_config):
        """Test environment configuration."""
        from environment import PlaywrightEnv
        
        env = PlaywrightEnv(**test_config)
        
        assert env.config.airport == test_config["airport"]
        assert env.config.max_aircraft == test_config["max_aircraft"]
        assert env.config.timewarp == test_config["timewarp"]
        
        env.close()

