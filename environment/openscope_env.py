"""
OpenScope Gymnasium Environment
Interfaces with the openScope ATC simulator for RL training
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Any
import time
from playwright.sync_api import sync_playwright, Page


class OpenScopeEnv(gym.Env):
    """
    Gymnasium environment for openScope ATC simulator
    
    Observation Space:
        - Aircraft states (variable number, padded to max_aircraft)
        - Global state (time, runway config, etc.)
        - Conflict matrix
    
    Action Space:
        - Aircraft selection (discrete)
        - Command type (discrete)
        - Command parameters (discrete)
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 1}
    
    def __init__(
        self,
        game_url: str = "http://localhost:3003",
        airport: str = "KLAS",
        timewarp: int = 5,
        max_aircraft: int = 20,
        episode_length: int = 3600,
        action_interval: float = 5.0,
        headless: bool = True,
        config: Optional[Dict] = None
    ):
        super().__init__()
        
        self.game_url = game_url
        self.airport = airport
        self.timewarp = timewarp
        self.max_aircraft = max_aircraft
        self.episode_length = episode_length
        self.action_interval = action_interval
        self.headless = headless
        self.config = config or {}
        
        # Browser automation
        self.playwright = None
        self.browser = None
        self.page: Optional[Page] = None
        
        # Episode state
        self.current_step = 0
        self.episode_score = 0
        self.last_action_time = 0
        
        # Define observation space
        aircraft_features = 32  # [x, y, alt, hdg, spd, phase, ...]
        self.observation_space = spaces.Dict({
            "aircraft": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(max_aircraft, aircraft_features),
                dtype=np.float32
            ),
            "aircraft_mask": spaces.Box(
                low=0,
                high=1,
                shape=(max_aircraft,),
                dtype=np.bool_
            ),
            "global_state": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(16,),
                dtype=np.float32
            ),
            "conflict_matrix": spaces.Box(
                low=0,
                high=1,
                shape=(max_aircraft, max_aircraft),
                dtype=np.float32
            )
        })
        
        # Define action space (hierarchical)
        # Action: [aircraft_id, command_type, param1, param2]
        self.action_space = spaces.Dict({
            "aircraft_id": spaces.Discrete(max_aircraft + 1),  # +1 for "no action"
            "command_type": spaces.Discrete(5),  # altitude, heading, speed, ils, direct
            "altitude": spaces.Discrete(18),  # Index into altitude list
            "heading": spaces.Discrete(13),  # Index into heading change list
            "speed": spaces.Discrete(8),  # Index into speed list
        })
        
        # Action mappings
        self.altitude_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 
                                100, 110, 120, 130, 140, 150, 160, 170, 180]
        self.heading_changes = [-90, -60, -45, -30, -20, -10, 0, 10, 20, 30, 45, 60, 90]
        self.speed_values = [180, 200, 220, 240, 260, 280, 300, 320]
        self.command_types = ["altitude", "heading", "speed", "ils", "direct"]
        
    def _init_browser(self):
        """Initialize Playwright browser"""
        if self.playwright is None:
            self.playwright = sync_playwright().start()
            self.browser = self.playwright.chromium.launch(headless=self.headless)
            self.page = self.browser.new_page()
            
            # Load the game
            self.page.goto(self.game_url)
            self.page.wait_for_load_state("networkidle")
            
            # Set airport
            self._execute_command(f"airport {self.airport}")
            time.sleep(2)
            
            # Set timewarp
            self._execute_command(f"timewarp {self.timewarp}")
            time.sleep(0.5)
    
    def _execute_command(self, command: str):
        """Execute a command in the openScope game"""
        if self.page is None:
            return
        
        # JavaScript to execute command
        js_code = f"""
        (function() {{
            const input = document.querySelector('#command');
            if (input) {{
                input.value = '{command}';
                input.dispatchEvent(new KeyboardEvent('keydown', {{
                    key: 'Enter',
                    code: 'Enter',
                    keyCode: 13
                }}));
            }}
        }})();
        """
        self.page.evaluate(js_code)
    
    def _get_game_state(self) -> Dict[str, Any]:
        """Extract current game state from the browser"""
        if self.page is None:
            return {}
        
        # JavaScript to extract game state
        js_code = """
        (function() {
            if (!window.aircraftController) return null;
            
            const aircraft = [];
            const aircraftList = window.aircraftController.aircraft.list;
            
            for (let i = 0; i < aircraftList.length; i++) {
                const ac = aircraftList[i];
                aircraft.push({
                    callsign: ac.callsign,
                    position: ac.relativePosition,
                    altitude: ac.altitude,
                    heading: ac.heading,
                    speed: ac.speed,
                    groundSpeed: ac.groundSpeed,
                    assignedAltitude: ac.mcp.altitude,
                    assignedHeading: ac.mcp.heading,
                    assignedSpeed: ac.mcp.speed,
                    category: ac.category,
                    isOnGround: ac.isOnGround(),
                    isTaxiing: ac.isTaxiing(),
                    isEstablished: ac.isEstablishedOnCourse ? ac.isEstablishedOnCourse() : false,
                    targetRunway: ac.fms.arrivalRunwayModel ? ac.fms.arrivalRunwayModel.name : null
                });
            }
            
            const conflicts = [];
            if (window.aircraftController.conflicts) {
                window.aircraftController.conflicts.forEach(conflict => {
                    conflicts.push({
                        aircraft1: conflict.aircraft[0].callsign,
                        aircraft2: conflict.aircraft[1].callsign,
                        distance: conflict.distance,
                        altitude: conflict.altitude,
                        hasConflict: conflict.hasConflict(),
                        hasViolation: conflict.hasViolation()
                    });
                });
            }
            
            return {
                aircraft: aircraft,
                conflicts: conflicts,
                score: window.gameController ? window.gameController.game.score : 0,
                time: window.TimeKeeper ? window.TimeKeeper.accumulatedDeltaTime : 0
            };
        })();
        """
        
        state = self.page.evaluate(js_code)
        return state if state else {}
    
    def _state_to_observation(self, state: Dict) -> Dict[str, np.ndarray]:
        """Convert game state to gym observation"""
        aircraft_data = state.get("aircraft", [])
        conflicts = state.get("conflicts", [])
        
        # Aircraft features
        aircraft_obs = np.zeros((self.max_aircraft, 32), dtype=np.float32)
        aircraft_mask = np.zeros(self.max_aircraft, dtype=bool)
        
        for i, ac in enumerate(aircraft_data[:self.max_aircraft]):
            aircraft_mask[i] = True
            
            # Normalize features
            pos = ac.get("position", [0, 0])
            aircraft_obs[i, :] = [
                pos[0] / 100.0,  # Normalized x position
                pos[1] / 100.0,  # Normalized y position
                ac.get("altitude", 0) / 50000.0,  # Normalized altitude
                ac.get("heading", 0) / 360.0,  # Normalized heading
                ac.get("speed", 0) / 500.0,  # Normalized speed
                ac.get("groundSpeed", 0) / 600.0,  # Normalized ground speed
                ac.get("assignedAltitude", 0) / 50000.0,
                ac.get("assignedHeading", 0) / 360.0,
                ac.get("assignedSpeed", 0) / 500.0,
                1.0 if ac.get("isOnGround", False) else 0.0,
                1.0 if ac.get("isTaxiing", False) else 0.0,
                1.0 if ac.get("isEstablished", False) else 0.0,
                1.0 if ac.get("category") == "arrival" else 0.0,
                1.0 if ac.get("category") == "departure" else 0.0,
                # Additional features can be added
                *([0.0] * 18)  # Padding to 32 features
            ]
        
        # Global state
        global_state = np.array([
            state.get("time", 0) / 3600.0,  # Normalized time
            len(aircraft_data) / self.max_aircraft,  # Aircraft count ratio
            len(conflicts) / 10.0,  # Conflict count (normalized)
            state.get("score", 0) / 1000.0,  # Normalized score
            *([0.0] * 12)  # Padding to 16 features
        ], dtype=np.float32)
        
        # Conflict matrix
        conflict_matrix = np.zeros((self.max_aircraft, self.max_aircraft), dtype=np.float32)
        callsign_to_idx = {ac.get("callsign"): i for i, ac in enumerate(aircraft_data[:self.max_aircraft])}
        
        for conflict in conflicts:
            cs1 = conflict.get("aircraft1")
            cs2 = conflict.get("aircraft2")
            if cs1 in callsign_to_idx and cs2 in callsign_to_idx:
                i, j = callsign_to_idx[cs1], callsign_to_idx[cs2]
                # Encode conflict severity
                severity = 0.5 if conflict.get("hasConflict") else 0.0
                severity = 1.0 if conflict.get("hasViolation") else severity
                conflict_matrix[i, j] = severity
                conflict_matrix[j, i] = severity
        
        return {
            "aircraft": aircraft_obs,
            "aircraft_mask": aircraft_mask,
            "global_state": global_state,
            "conflict_matrix": conflict_matrix
        }
    
    def _calculate_reward(self, state: Dict, prev_state: Dict) -> float:
        """Calculate reward based on state change"""
        reward = 0.0
        
        # Get score change
        current_score = state.get("score", 0)
        prev_score = prev_state.get("score", 0)
        score_delta = current_score - prev_score
        
        # Main reward from game score
        reward += score_delta
        
        # Shaped rewards
        conflicts = state.get("conflicts", [])
        aircraft = state.get("aircraft", [])
        
        # Penalty for conflicts
        for conflict in conflicts:
            if conflict.get("hasViolation"):
                reward -= 5.0  # Additional penalty beyond game score
            elif conflict.get("hasConflict"):
                reward -= 1.0  # Warning
        
        # Small timestep penalty to encourage efficiency
        reward -= 0.01
        
        # Bonus for maintaining safe operations
        if len(conflicts) == 0 and len(aircraft) > 0:
            reward += 0.05
        
        return reward
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """Reset the environment"""
        super().reset(seed=seed)
        
        # Initialize browser if needed
        if self.page is None:
            self._init_browser()
        
        # Reset game
        self._execute_command(f"airport {self.airport}")
        time.sleep(2)
        self._execute_command(f"timewarp {self.timewarp}")
        time.sleep(0.5)
        
        # Get initial state
        self.current_step = 0
        self.episode_score = 0
        self.prev_state = self._get_game_state()
        self.last_action_time = time.time()
        
        observation = self._state_to_observation(self.prev_state)
        info = {"raw_state": self.prev_state}
        
        return observation, info
    
    def step(self, action: Dict) -> Tuple[Dict, float, bool, bool, Dict]:
        """Execute action and return next state"""
        # Parse action
        aircraft_id = action["aircraft_id"]
        
        # Execute action if not "no action"
        if aircraft_id < len(self.prev_state.get("aircraft", [])):
            aircraft = self.prev_state["aircraft"][aircraft_id]
            callsign = aircraft["callsign"]
            command_type = self.command_types[action["command_type"]]
            
            # Build command string
            command = f"{callsign} "
            
            if command_type == "altitude":
                alt = self.altitude_values[action["altitude"]]
                command += f"c {alt}"
            elif command_type == "heading":
                hdg_change = self.heading_changes[action["heading"]]
                current_hdg = aircraft.get("heading", 0)
                new_hdg = int((current_hdg + hdg_change) % 360)
                command += f"fh {new_hdg:03d}"
            elif command_type == "speed":
                spd = self.speed_values[action["speed"]]
                command += f"sp {spd}"
            elif command_type == "ils":
                if aircraft.get("targetRunway"):
                    command += f"i {aircraft['targetRunway']}"
                else:
                    command = None
            elif command_type == "direct":
                # Simplified: just continue on route
                command = None
            
            if command:
                self._execute_command(command)
        
        # Wait for action interval
        time.sleep(self.action_interval / self.timewarp)
        
        # Get new state
        state = self._get_game_state()
        observation = self._state_to_observation(state)
        
        # Calculate reward
        reward = self._calculate_reward(state, self.prev_state)
        
        # Check termination
        self.current_step += 1
        game_time = state.get("time", 0)
        
        terminated = False
        truncated = game_time >= self.episode_length or self.current_step >= 1000
        
        # Check for critical failures (multiple collisions)
        if state.get("score", 0) < -2000:
            terminated = True
            reward -= 100  # Extra penalty
        
        info = {
            "raw_state": state,
            "score": state.get("score", 0),
            "aircraft_count": len(state.get("aircraft", [])),
            "conflict_count": len(state.get("conflicts", []))
        }
        
        self.prev_state = state
        
        return observation, reward, terminated, truncated, info
    
    def close(self):
        """Clean up resources"""
        if self.page:
            self.page.close()
        if self.browser:
            self.browser.close()
        if self.playwright:
            self.playwright.stop()
        
        self.page = None
        self.browser = None
        self.playwright = None
    
    def render(self):
        """Render the environment (game renders itself in browser)"""
        pass

