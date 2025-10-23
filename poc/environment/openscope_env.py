"""
OpenScope Gymnasium Environment
Clean POC implementation for RL training with openScope ATC simulator
"""

import threading
import time
from queue import Queue
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from playwright.sync_api import Browser, Page, Playwright, sync_playwright


class BrowserThread:
    """Dedicated thread for Playwright to avoid Jupyter asyncio conflicts"""

    def __init__(self):
        self.queue = Queue()
        self.result_queue = Queue()
        self.thread = None
        self.playwright: Optional[Playwright] = None
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None
        self.running = False

    def start(self):
        """Start the browser thread"""
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        """Main loop for the browser thread"""
        while self.running:
            try:
                command = self.queue.get(timeout=0.1)
                if command is None:  # Shutdown signal
                    break

                method, args, kwargs = command
                try:
                    result = getattr(self, f"_{method}")(*args, **kwargs)
                    self.result_queue.put((True, result))
                except Exception as e:
                    self.result_queue.put((False, e))
            except:
                continue

    def _init_browser(self, game_url, airport, timewarp, headless):
        """Initialize browser (runs in thread)"""
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=headless)
        self.page = self.browser.new_page()

        # Capture JS errors
        self.page.on("pageerror", lambda err: print(f"❌ JS Error: {err}"))

        # Inject time tracking
        self.page.add_init_script("""
            (function() {
                let startTime = null;
                let accumulatedGameTime = 0;
                const originalRAF = window.requestAnimationFrame;
                window.requestAnimationFrame = function(callback) {
                    return originalRAF.call(window, function(timestamp) {
                        if (startTime === null) startTime = timestamp;
                        accumulatedGameTime = (timestamp - startTime) / 1000;
                        return callback(timestamp);
                    });
                };
                window._getRLGameTime = () => accumulatedGameTime;
            })();
        """)

        # Load game
        self.page.goto(game_url)
        self.page.wait_for_load_state("networkidle")
        time.sleep(2)

        # Set airport and timewarp
        self._execute_command(f"airport {airport}")
        time.sleep(10)
        self._execute_command(f"timewarp {timewarp}")
        time.sleep(0.5)

    def _execute_command(self, command: str):
        """Execute command (runs in thread)"""
        if self.page:
            self.page.evaluate(f"""
                (function() {{
                    const input = document.querySelector('#command');
                    if (input) {{
                        input.value = '{command}';
                        input.dispatchEvent(new KeyboardEvent('keydown', {{
                            key: 'Enter', code: 'Enter', keyCode: 13
                        }}));
                    }}
                }})();
            """)

    def _get_game_state(self) -> dict[str, Any]:
        """Get game state (runs in thread)"""
        if not self.page:
            return {}

        raw_state = self.page.evaluate("""
            (function() {
                if (!window.aircraftController) return null;

                const aircraft = [];
                for (const ac of window.aircraftController.aircraft.list) {
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
                    window.aircraftController.conflicts.forEach(c => {
                        conflicts.push({
                            aircraft1: c.aircraft[0].callsign,
                            aircraft2: c.aircraft[1].callsign,
                            distance: c.distance,
                            altitude: c.altitude,
                            hasConflict: c.hasConflict(),
                            hasViolation: c.hasViolation()
                        });
                    });
                }

                let gameTime = 0;
                if (window._getRLGameTime) {
                    try { gameTime = window._getRLGameTime(); } catch (e) {}
                }

                return {
                    aircraft: aircraft,
                    conflicts: conflicts,
                    score: window.gameController ? window.gameController.game.score : 0,
                    time: gameTime
                };
            })();
        """)

        return raw_state if raw_state else {}

    def _close_browser(self):
        """Close browser (runs in thread)"""
        if self.page:
            self.page.close()
        if self.browser:
            self.browser.close()
        if self.playwright:
            self.playwright.stop()

    def call(self, method: str, *args, **kwargs):
        """Call a method in the browser thread and wait for result"""
        self.queue.put((method, args, kwargs))
        success, result = self.result_queue.get()
        if not success:
            raise result
        return result

    def stop(self):
        """Stop the browser thread"""
        self.running = False
        self.queue.put(None)
        if self.thread:
            self.thread.join(timeout=5)


class OpenScopeEnv(gym.Env):
    """
    Gymnasium environment for openScope ATC simulator

    Uses threading for Playwright to avoid Jupyter asyncio conflicts
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
        config: Optional[dict[str, Any]] = None,
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

        # Browser thread for Jupyter compatibility
        self.browser_thread: Optional[BrowserThread] = None

        # Episode state
        self.current_step = 0
        self.simulated_time = 0.0
        self.prev_state: dict[str, Any] = {}

        # Observation space
        self.observation_space = spaces.Dict({
            "aircraft": spaces.Box(
                low=-np.inf, high=np.inf, shape=(max_aircraft, 14), dtype=np.float32
            ),
            "aircraft_mask": spaces.Box(
                low=0, high=1, shape=(max_aircraft,), dtype=np.bool_
            ),
            "global_state": spaces.Box(
                low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
            ),
            "conflict_matrix": spaces.Box(
                low=0, high=1, shape=(max_aircraft, max_aircraft), dtype=np.float32
            ),
        })

        # Action space
        self.action_space = spaces.Dict({
            "aircraft_id": spaces.Discrete(max_aircraft + 1),
            "command_type": spaces.Discrete(5),
            "altitude": spaces.Discrete(18),
            "heading": spaces.Discrete(13),
            "speed": spaces.Discrete(8),
        })

        # Action mappings
        self.altitude_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180]
        self.heading_changes = [-90, -60, -45, -30, -20, -10, 0, 10, 20, 30, 45, 60, 90]
        self.speed_values = [180, 200, 220, 240, 260, 280, 300, 320]
        self.command_types = ["altitude", "heading", "speed", "ils", "direct"]

    def _init_browser(self):
        """Initialize browser thread"""
        if self.browser_thread is None:
            self.browser_thread = BrowserThread()
            self.browser_thread.start()
            time.sleep(0.5)  # Let thread start
            self.browser_thread.call('init_browser', self.game_url, self.airport, self.timewarp, self.headless)

    def _execute_command(self, command: str):
        """Execute command via browser thread"""
        if self.browser_thread:
            self.browser_thread.call('execute_command', command)

    def _get_game_state(self) -> dict[str, Any]:
        """Get game state via browser thread"""
        if not self.browser_thread:
            return {}
        state = self.browser_thread.call('get_game_state')
        # Fallback to simulated time if needed
        if state.get('time', 0) == 0:
            state['time'] = self.simulated_time
        return state

    def _state_to_observation(self, state: dict[str, Any]) -> dict[str, np.ndarray]:
        """Convert game state to gym observation"""
        aircraft_data = state.get("aircraft", [])
        conflicts = state.get("conflicts", [])

        # Aircraft features
        aircraft_obs = np.zeros((self.max_aircraft, 14), dtype=np.float32)
        aircraft_mask = np.zeros(self.max_aircraft, dtype=bool)

        for i, ac in enumerate(aircraft_data[:self.max_aircraft]):
            aircraft_mask[i] = True
            pos = ac.get("position", [0, 0])
            aircraft_obs[i, :] = [
                pos[0] / 100.0,                          # x position
                pos[1] / 100.0,                          # y position
                ac.get("altitude", 0) / 50000.0,         # altitude
                ac.get("heading", 0) / 360.0,            # heading
                ac.get("speed", 0) / 500.0,              # speed
                ac.get("groundSpeed", 0) / 600.0,        # ground speed
                ac.get("assignedAltitude", 0) / 50000.0, # assigned altitude
                ac.get("assignedHeading", 0) / 360.0,    # assigned heading
                ac.get("assignedSpeed", 0) / 500.0,      # assigned speed
                1.0 if ac.get("isOnGround", False) else 0.0,
                1.0 if ac.get("isTaxiing", False) else 0.0,
                1.0 if ac.get("isEstablished", False) else 0.0,
                1.0 if ac.get("category") == "arrival" else 0.0,
                1.0 if ac.get("category") == "departure" else 0.0,
            ]

        # Global state
        global_state = np.array([
            state.get("time", 0) / 3600.0,
            len(aircraft_data) / self.max_aircraft,
            len(conflicts) / 10.0,
            state.get("score", 0) / 1000.0,
        ], dtype=np.float32)

        # Conflict matrix
        conflict_matrix = np.zeros((self.max_aircraft, self.max_aircraft), dtype=np.float32)
        callsign_to_idx = {
            ac.get("callsign"): i
            for i, ac in enumerate(aircraft_data[:self.max_aircraft])
        }

        for c in conflicts:
            cs1, cs2 = c.get("aircraft1"), c.get("aircraft2")
            if cs1 in callsign_to_idx and cs2 in callsign_to_idx:
                i, j = callsign_to_idx[cs1], callsign_to_idx[cs2]
                severity = 1.0 if c.get("hasViolation") else (0.5 if c.get("hasConflict") else 0.0)
                conflict_matrix[i, j] = severity
                conflict_matrix[j, i] = severity

        return {
            "aircraft": aircraft_obs,
            "aircraft_mask": aircraft_mask,
            "global_state": global_state,
            "conflict_matrix": conflict_matrix,
        }

    def _calculate_reward(self, state: dict[str, Any], prev_state: dict[str, Any]) -> float:
        """Calculate reward based on state change"""
        reward = state.get("score", 0) - prev_state.get("score", 0)

        # Shaped rewards
        reward_config = self.config.get("rewards", {})
        conflicts = state.get("conflicts", [])
        aircraft = state.get("aircraft", [])

        for c in conflicts:
            if c.get("hasViolation"):
                reward += reward_config.get("separation_loss", -200) / 10.0
            elif c.get("hasConflict"):
                reward += reward_config.get("conflict_warning", -2.0)

        reward += reward_config.get("timestep_penalty", -0.01)

        if len(conflicts) == 0 and len(aircraft) > 0:
            reward += reward_config.get("safe_separation_bonus", 0.05)

        return reward

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Reset environment"""
        super().reset(seed=seed)

        # Initialize browser if needed
        if self.browser_thread is None:
            self._init_browser()

        # Reset game
        self._execute_command("clear")
        time.sleep(1)
        self._execute_command(f"airport {self.airport}")
        time.sleep(2)
        self._execute_command(f"timewarp {self.timewarp}")
        time.sleep(0.5)

        # Get initial state
        self.current_step = 0
        self.simulated_time = 0.0
        self.prev_state = self._get_game_state()

        observation = self._state_to_observation(self.prev_state)
        info = {"raw_state": self.prev_state}

        return observation, info

    def step(
        self, action: dict[str, int]
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Execute action and return next state"""
        aircraft_id = action["aircraft_id"]

        # Execute action if valid
        if aircraft_id < len(self.prev_state.get("aircraft", [])):
            aircraft = self.prev_state["aircraft"][aircraft_id]
            callsign = aircraft["callsign"]
            command_type = self.command_types[action["command_type"]]

            command = None
            if command_type == "altitude":
                alt_fl = self.altitude_values[action["altitude"]]
                command = f"{callsign} c {alt_fl}"
            elif command_type == "heading":
                hdg_change = self.heading_changes[action["heading"]]
                new_hdg = int((aircraft.get("heading", 0) + hdg_change) % 360)
                command = f"{callsign} fh {new_hdg:03d}"
            elif command_type == "speed":
                spd = self.speed_values[action["speed"]]
                command = f"{callsign} sp {spd}"
            elif command_type == "ils" and aircraft.get("targetRunway"):
                command = f"{callsign} i {aircraft['targetRunway']}"

            if command:
                self._execute_command(command)

        # Wait for action interval
        time.sleep(self.action_interval / self.timewarp)
        self.simulated_time += self.action_interval

        # Get new state
        state = self._get_game_state()
        observation = self._state_to_observation(state)
        reward = self._calculate_reward(state, self.prev_state)

        # Check termination
        self.current_step += 1
        terminated = state.get("score", 0) < -2000
        truncated = state.get("time", 0) >= self.episode_length or self.current_step >= 1000

        if terminated:
            reward -= 100

        info = {
            "raw_state": state,
            "score": state.get("score", 0),
            "aircraft_count": len(state.get("aircraft", [])),
            "conflict_count": len(state.get("conflicts", [])),
        }

        self.prev_state = state
        return observation, reward, terminated, truncated, info

    def close(self):
        """Clean up resources"""
        if self.browser_thread:
            self.browser_thread.call('close_browser')
            self.browser_thread.stop()
            self.browser_thread = None

    def render(self):
        """Render (game renders itself in browser)"""
        pass
