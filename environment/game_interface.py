"""
Game interface module for OpenScope RL environment.

This module provides the OpenScopeInterface class for communicating with the
OpenScope game, handling command execution and state extraction.
"""

import logging
import time
from typing import Any, Dict, List, Optional
from playwright.sync_api import Page

from .config import OpenScopeConfig
from .exceptions import GameInterfaceError
from .utils import execute_command, extract_game_state


logger = logging.getLogger(__name__)


class OpenScopeInterface:
    """
    Interface for communicating with the OpenScope game.
    
    This class handles all communication with the OpenScope game including
    command execution, state extraction, and game initialization.
    """
    
    def __init__(self, config: OpenScopeConfig):
        """
        Initialize OpenScope interface.
        
        Args:
            config: OpenScope configuration
        """
        self.config = config
        self.page: Optional[Page] = None
        self._is_initialized = False
    
    def initialize(self, page: Page) -> None:
        """
        Initialize the interface with a browser page.
        
        Args:
            page: Playwright page object
            
        Raises:
            GameInterfaceError: If initialization fails
        """
        try:
            self.page = page
            self._setup_game()
            self._is_initialized = True
            logger.info("OpenScope interface initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenScope interface: {e}")
            raise GameInterfaceError(f"Interface initialization failed: {e}") from e
    
    def _setup_game(self) -> None:
        """Setup the game with airport and timewarp settings."""
        if not self.page:
            raise GameInterfaceError("Page not available")
        
        # Set airport
        logger.info(f"Setting airport to {self.config.airport}")
        execute_command(self.page, f"airport {self.config.airport}")
        
        # Wait for airport setup
        time.sleep(self.config.browser_config.airport_setup_delay)
        
        # Set timewarp
        logger.info(f"Setting timewarp to {self.config.timewarp}x")
        execute_command(self.page, f"timewarp {self.config.timewarp}")
        
        # Wait for timewarp setup
        time.sleep(self.config.browser_config.timewarp_setup_delay)
        
        logger.info("Game setup completed")
    
    def execute_action(self, action: Dict[str, int], aircraft_data: List[Dict[str, Any]]) -> Optional[str]:
        """
        Execute an action in the game.
        
        Args:
            action: Action dictionary with aircraft_id, command_type, etc.
            aircraft_data: Current aircraft data from game state
            
        Returns:
            Command string that was executed, or None if no action taken
            
        Raises:
            GameInterfaceError: If action execution fails
        """
        if not self._is_initialized:
            raise GameInterfaceError("Interface not initialized")
        
        aircraft_id = action["aircraft_id"]
        
        # Check for "no action": aircraft_id == max_aircraft is the dedicated "no-op" action
        # This allows the agent to explicitly choose to do nothing in a cycle
        max_aircraft = self.config.max_aircraft
        if aircraft_id == max_aircraft:
            logger.debug(f"No action taken: aircraft_id {aircraft_id} (explicit no-op)")
            return None
        
        # Safety check: also handle out-of-bounds as no-op (shouldn't happen with proper masking)
        if aircraft_id < 0 or aircraft_id >= len(aircraft_data):
            logger.debug(f"No action taken: aircraft_id {aircraft_id} (out of range [0, {len(aircraft_data)})")
            return None
        
        # aircraft_id now valid: selects aircraft at that index
        aircraft = aircraft_data[aircraft_id]
        callsign = aircraft["callsign"]
        command_type = self.config.action_config.command_types[action["command_type"]]
        
        command = self._build_command(action, aircraft, command_type)
        
        if command:
            try:
                execute_command(self.page, command)
                logger.debug(f"Executed command: {command}")
                return command
            except Exception as e:
                logger.error(f"Failed to execute command '{command}': {e}")
                raise GameInterfaceError(f"Command execution failed: {e}") from e
        
        return None
    
    def _build_command(self, action: Dict[str, int], aircraft: Dict[str, Any],
                      command_type) -> Optional[str]:
        """
        Build command string from action and aircraft data.

        Args:
            action: Action dictionary
            aircraft: Aircraft data
            command_type: Type of command to build (CommandType enum)

        Returns:
            Command string or None if no valid command
        """
        from .config import CommandType

        callsign = aircraft["callsign"]

        if command_type == CommandType.ALTITUDE:
            alt_fl = self.config.action_config.altitude_values[action["altitude"]]
            return f"{callsign} c {alt_fl}"

        elif command_type == CommandType.HEADING:
            hdg_change = self.config.action_config.heading_changes[action["heading"]]
            current_heading = aircraft.get("heading", 0)
            # Round current heading to avoid float artifacts, wrap to [0, 360),
            # and map 0 to 360 to satisfy OpenScope's 001-360 requirement
            new_hdg = int((int(round(current_heading)) + hdg_change) % 360)
            if new_hdg == 0:
                new_hdg = 360
            return f"{callsign} fh {new_hdg:03d}"

        elif command_type == CommandType.SPEED:
            spd = self.config.action_config.speed_values[action["speed"]]
            return f"{callsign} sp {spd}"

        elif command_type == CommandType.ILS:
            target_runway = aircraft.get("targetRunway") or aircraft.get("target_runway")
            if target_runway:
                return f"{callsign} i {target_runway}"
            else:
                logger.debug(f"No target runway for aircraft {callsign}, ILS command skipped")
                return None

        elif command_type == CommandType.DIRECT:
            # Direct command - could be implemented based on specific needs
            logger.debug(f"Direct command for {callsign} - not implemented")
            return None

        else:
            logger.warning(f"Unknown command type: {command_type}")
            return None
    
    def get_game_state(self) -> Dict[str, Any]:
        """
        Get current game state.
        
        Returns:
            Dictionary containing aircraft data, conflicts, score, and time
            
        Raises:
            GameInterfaceError: If state extraction fails
        """
        if not self._is_initialized:
            raise GameInterfaceError("Interface not initialized")
        
        try:
            state = extract_game_state(self.page)
            logger.debug(f"Extracted state: {len(state.get('aircraft', []))} aircraft")
            return state
            
        except Exception as e:
            logger.error(f"Failed to get game state: {e}")
            raise GameInterfaceError(f"State extraction failed: {e}") from e
    
    def reset_game(self) -> None:
        """
        Reset the game to initial state.
        
        Raises:
            GameInterfaceError: If reset fails
        """
        if not self._is_initialized:
            raise GameInterfaceError("Interface not initialized")
        
        try:
            logger.info("Resetting game")
            execute_command(self.page, "clear")

            time.sleep(0.5)  # Reduced from 1.0 second

            # Re-setup game
            self._setup_game()
            logger.info("Game reset completed")
            
        except Exception as e:
            logger.error(f"Failed to reset game: {e}")
            raise GameInterfaceError(f"Game reset failed: {e}") from e
    
    def wait_for_update(self, num_frames: int = 1) -> None:
        """
        Wait for game to process updates.
        
        Args:
            num_frames: Number of frames to wait
        """
        if not self._is_initialized:
            raise GameInterfaceError("Interface not initialized")
        
        time.sleep(num_frames * self.config.browser_config.game_update_delay)
    
    def is_game_ready(self) -> bool:
        """
        Check if the game is ready for interaction.
        
        Returns:
            True if game is ready
        """
        if not self._is_initialized or not self.page:
            return False
        
        try:
            # Check if page is still valid
            if self.page.is_closed():
                logger.warning("Page is closed during game readiness check")
                return False
            
            # Check if aircraft controller is available
            # Note: OpenScope only exposes window.aircraftController, not window.gameController
            result = self.page.evaluate("""
                (function() {
                    const hasAircraftController = !!window.aircraftController;
                    const hasAircraftList = hasAircraftController &&
                                           window.aircraftController.aircraft &&
                                           window.aircraftController.aircraft.list;
                    const hasScoreElement = !!document.querySelector('#score');

                    return {
                        ready: hasAircraftController && hasAircraftList && hasScoreElement,
                        hasAircraftController: hasAircraftController,
                        hasAircraftList: hasAircraftList,
                        hasScoreElement: hasScoreElement,
                        url: window.location.href
                    };
                })();
            """)

            if result and result.get('ready'):
                logger.debug("Game is ready")
                return True
            else:
                logger.debug(f"Game not ready: {result}")
                return False
            
        except Exception as e:
            logger.warning(f"Failed to check game readiness: {e}")
            return False
    
    def get_game_info(self) -> Dict[str, Any]:
        """
        Get general game information.

        Returns:
            Dictionary with game information
        """
        if not self._is_initialized:
            return {}

        try:
            info = self.page.evaluate("""
                (function() {
                    return {
                        hasAircraftController: !!window.aircraftController,
                        hasScoreElement: !!document.querySelector('#score'),
                        gameUrl: window.location.href,
                        timestamp: Date.now()
                    };
                })();
            """)
            return info

        except Exception as e:
            logger.warning(f"Failed to get game info: {e}")
            # Return minimal safe info with error indicator rather than empty dict
            return {
                "hasAircraftController": False,
                "hasScoreElement": False,
                "error": str(e)
            }
    
    @property
    def is_initialized(self) -> bool:
        """Check if interface is initialized."""
        return self._is_initialized and self.page is not None