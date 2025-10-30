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
        # Verify setup commands were executed using Playwright's fill() and press()
        assert mock_page.fill.called
        assert mock_page.press.called
    
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
        
        # Track command execution using fill() and press()
        command_filled = []
        def capture_fill(selector, value):
            if selector == '#command':
                command_filled.append(value)
        
        mock_page.fill = Mock(side_effect=capture_fill)
        
        result = interface.execute_action(action, aircraft_data)
        
        # Should execute a command like "UAL123 c 100"
        assert result is not None
        assert "UAL123" in result
        assert "c" in result
        # Verify command was filled and Enter was pressed
        assert mock_page.fill.called
        assert mock_page.press.called
        assert any("UAL123" in cmd for cmd in command_filled)
    
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
        
        # Should call clear and re-setup using fill() and press()
        assert mock_page.fill.called
        assert mock_page.press.called
    
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
    
    def test_event_detection_and_scoring(self):
        """Test that all scoring events are detected and match the actual score.
        
        This test verifies:
        1. Event hook is properly installed and captures events
        2. All event types that affect score are being tracked
        3. Calculated score from events matches actual game score
        
        This is critical for reward calculation accuracy in RL training.
        """
        from environment import PlaywrightEnv
        from environment.constants import GAME_EVENT_SCORES
        from environment.utils import extract_optimal_game_state
        import time
        
        env = PlaywrightEnv(
            game_url="http://localhost:3003",
            airport="KLAS",
            max_aircraft=15,
            timewarp=1000,  # High timewarp to generate many events quickly
            headless=True,
            episode_length=300,
        )
        
        try:
            # Reset environment to initialize
            obs, info = env.reset()
            page = env.browser_manager.page
            
            # Check that event hook is installed
            hook_status = page.evaluate("""() => ({
                hookInstalled: !!window._rlDrainEvents,
                gcExists: !!window.gameController,
                gcWrapped: !!(window.gameController && window.gameController._eventsRecordNewOriginal),
            })""")
            
            assert hook_status["hookInstalled"], "Event hook not installed (_rlDrainEvents missing)"
            assert hook_status["gcExists"], "GameController not exposed on window"
            assert hook_status["gcWrapped"], "GameController.events_recordNew not wrapped"
            
            # Get initial score
            initial_state = extract_optimal_game_state(page)
            initial_score = initial_state.get("score", 0)
            
            # Let the game run for 20 seconds (simulated time: ~5.5 hours at 1000x)
            # This should generate multiple events (arrivals, departures, etc.)
            time.sleep(20)
            
            # Extract final state with events
            final_state = extract_optimal_game_state(page)
            final_score = final_state.get("score", 0)
            event_counts = final_state.get("event_counts", {})
            
            # Verify we captured events
            assert event_counts, "No events captured during test"
            assert len(event_counts) > 0, "Event counts dictionary is empty"
            
            # Calculate expected score change from events
            calculated_score_change = 0
            for event_type, count in event_counts.items():
                if event_type in GAME_EVENT_SCORES:
                    points_per_event = GAME_EVENT_SCORES[event_type]
                    calculated_score_change += points_per_event * count
                else:
                    # Warn about unknown event types
                    print(f"Warning: Unknown event type '{event_type}' not in GAME_EVENT_SCORES")
            
            actual_score_change = final_score - initial_score
            
            # Print diagnostic info
            print(f"\n=== Event Detection Test Results ===")
            print(f"Initial score: {initial_score}")
            print(f"Final score: {final_score}")
            print(f"Actual score change: {actual_score_change}")
            print(f"Calculated score change from events: {calculated_score_change}")
            print(f"\nEvent counts:")
            for event_type, count in sorted(event_counts.items(), key=lambda x: -x[1]):
                points = GAME_EVENT_SCORES.get(event_type, 0)
                total_points = points * count
                print(f"  {event_type:35s}: {count:4d} × {points:5d} = {total_points:7d} pts")
            print(f"\nScore match: {'✓ PASS' if abs(actual_score_change - calculated_score_change) < 10 else '✗ FAIL'}")
            
            # Assert that calculated score matches actual score
            # Allow small tolerance for timing/rounding issues
            score_tolerance = 10  # Allow ±10 points difference
            assert abs(actual_score_change - calculated_score_change) <= score_tolerance, \
                f"Score mismatch: actual change={actual_score_change}, " \
                f"calculated from events={calculated_score_change}, " \
                f"difference={abs(actual_score_change - calculated_score_change)}"
            
            # Verify we're tracking major event types
            # At minimum, we expect departures and arrivals in a busy airport
            expected_event_types = {'ARRIVAL', 'DEPARTURE'}
            captured_event_types = set(event_counts.keys())
            common_events = expected_event_types & captured_event_types
            
            assert len(common_events) > 0, \
                f"Expected to capture at least one of {expected_event_types}, " \
                f"but only captured {captured_event_types}"
            
            print(f"\n✅ Event detection test PASSED")
            print(f"   - Hook properly installed and wrapping events_recordNew")
            print(f"   - Captured {len(event_counts)} different event types")
            print(f"   - Score calculation matches actual score (±{score_tolerance} pts)")
            
        finally:
            env.close()

