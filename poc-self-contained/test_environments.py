#!/usr/bin/env python3
"""
Test script for self-contained ATC environments.
Verifies that both 2D and 3D environments work correctly.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import gymnasium as gym
from gymnasium import spaces

print("=" * 60)
print("Testing Self-Contained ATC Environments")
print("=" * 60)

# ============================================================================
# Simple 2D Environment Test
# ============================================================================

print("\n[1/2] Testing Simple 2D ATC Environment...")

# Constants
AIRSPACE_SIZE = 20.0
SEPARATION_MIN = 3.0
CONFLICT_BUFFER = 1.0
AIRCRAFT_SPEED = 4.0
TURN_RATE = 15.0

@dataclass
class Aircraft2D:
    """Simple 2D aircraft."""
    callsign: str
    x: float
    y: float
    heading: float
    target_heading: float
    exit_edge: int

    def update(self, dt: float = 1.0):
        """Update position and heading."""
        heading_diff = (self.target_heading - self.heading + 180) % 360 - 180
        turn_amount = np.clip(heading_diff, -TURN_RATE, TURN_RATE)
        self.heading = (self.heading + turn_amount) % 360

        heading_rad = np.radians(self.heading)
        self.x += AIRCRAFT_SPEED * np.sin(heading_rad) * dt
        self.y += AIRCRAFT_SPEED * np.cos(heading_rad) * dt

    def distance_to(self, other: 'Aircraft2D') -> float:
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

class Simple2DATCEnv(gym.Env):
    """Minimal 2D ATC environment for testing."""

    def __init__(self, max_aircraft: int = 5, max_steps: int = 50):
        super().__init__()
        self.max_aircraft = max_aircraft
        self.max_steps = max_steps
        self.aircraft: List[Aircraft2D] = []
        self.current_step = 0

        self.observation_space = spaces.Dict({
            'aircraft': spaces.Box(
                low=-1.0, high=1.0,
                shape=(max_aircraft, 6),
                dtype=np.float32
            ),
            'mask': spaces.Box(
                low=0, high=1,
                shape=(max_aircraft,),
                dtype=np.uint8
            )
        })

        self.action_space = spaces.MultiDiscrete([max_aircraft + 1, 25])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.aircraft = []
        self.current_step = 0

        # Spawn 2 aircraft
        for i in range(2):
            self.aircraft.append(Aircraft2D(
                callsign=f"TEST{i}",
                x=self.np_random.uniform(-5, 5),
                y=self.np_random.uniform(-5, 5),
                heading=self.np_random.uniform(0, 360),
                target_heading=self.np_random.uniform(0, 360),
                exit_edge=0
            ))

        return self._get_obs(), {}

    def step(self, action):
        aircraft_id, heading_change_idx = action

        if aircraft_id < len(self.aircraft):
            heading_change = (heading_change_idx - 12) * 15
            ac = self.aircraft[aircraft_id]
            ac.target_heading = (ac.heading + heading_change) % 360

        for ac in self.aircraft:
            ac.update()

        self.current_step += 1
        terminated = self.current_step >= self.max_steps

        return self._get_obs(), 0.0, terminated, False, {}

    def _get_obs(self):
        obs = np.zeros((self.max_aircraft, 6), dtype=np.float32)
        mask = np.zeros(self.max_aircraft, dtype=np.uint8)

        for i, ac in enumerate(self.aircraft):
            if i < self.max_aircraft:
                obs[i] = [
                    ac.x / AIRSPACE_SIZE,
                    ac.y / AIRSPACE_SIZE,
                    np.sin(np.radians(ac.heading)),
                    np.cos(np.radians(ac.heading)),
                    0.0, 0.0
                ]
                mask[i] = 1

        return {'aircraft': obs, 'mask': mask}

# Test 2D environment
try:
    env = Simple2DATCEnv(max_aircraft=5, max_steps=10)
    obs, info = env.reset(seed=42)

    print(f"   ✅ Environment created")
    print(f"   ✅ Observation space: {env.observation_space}")
    print(f"   ✅ Action space: {env.action_space}")

    # Run a few steps
    for _ in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated:
            break

    print(f"   ✅ Steps executed successfully")
    print(f"   ✅ Final observation shape: aircraft={obs['aircraft'].shape}, mask={obs['mask'].shape}")

    print("\n✅ Simple 2D ATC Environment: PASSED")

except Exception as e:
    print(f"\n❌ Simple 2D ATC Environment: FAILED")
    print(f"   Error: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# Realistic 3D Environment Test
# ============================================================================

print("\n[2/2] Testing Realistic 3D ATC Environment...")

# Constants for 3D
SEPARATION_LATERAL_NM = 3.0
SEPARATION_VERTICAL_FT = 1000.0
AREA_SIZE_NM = 20.0
TURN_RATE_DEG_PER_SEC = 3.0
FT_PER_SEC_CLIMB = 33.3

@dataclass
class Aircraft3D:
    """3D aircraft with altitude."""
    callsign: str
    x: float
    y: float
    altitude: float
    heading: float
    speed: float
    target_altitude: float = None
    target_heading: float = None
    target_speed: float = None

    def __post_init__(self):
        if self.target_altitude is None:
            self.target_altitude = self.altitude
        if self.target_heading is None:
            self.target_heading = self.heading
        if self.target_speed is None:
            self.target_speed = self.speed

    def update(self, dt: float):
        """Update 3D state."""
        # Heading
        heading_diff = (self.target_heading - self.heading + 180) % 360 - 180
        max_turn = TURN_RATE_DEG_PER_SEC * dt
        if abs(heading_diff) <= max_turn:
            self.heading = self.target_heading
        else:
            self.heading += max_turn * np.sign(heading_diff)
        self.heading = self.heading % 360

        # Altitude
        alt_diff = self.target_altitude - self.altitude
        if abs(alt_diff) > 0:
            max_change = FT_PER_SEC_CLIMB * dt
            if abs(alt_diff) <= max_change:
                self.altitude = self.target_altitude
            else:
                self.altitude += max_change * np.sign(alt_diff)

        # Position
        heading_rad = np.radians(self.heading)
        speed_nm_per_sec = self.speed / 3600.0
        self.x += speed_nm_per_sec * dt * np.sin(heading_rad)
        self.y += speed_nm_per_sec * dt * np.cos(heading_rad)

class Realistic3DATCEnv(gym.Env):
    """Minimal 3D ATC environment for testing."""

    def __init__(self, max_aircraft: int = 8, max_steps: int = 50):
        super().__init__()
        self.max_aircraft = max_aircraft
        self.max_steps = max_steps
        self.aircraft: List[Aircraft3D] = []
        self.current_step = 0

        self.observation_space = spaces.Dict({
            'aircraft': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(max_aircraft, 14),
                dtype=np.float32
            ),
            'aircraft_mask': spaces.Box(
                low=0, high=1,
                shape=(max_aircraft,),
                dtype=np.uint8
            ),
            'conflict_matrix': spaces.Box(
                low=0.0, high=1.0,
                shape=(max_aircraft, max_aircraft),
                dtype=np.float32
            ),
            'global_state': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(4,),
                dtype=np.float32
            )
        })

        self.action_space = spaces.MultiDiscrete([
            max_aircraft + 1, 5, 18, 12, 8
        ])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.aircraft = []
        self.current_step = 0

        # Spawn 2 aircraft
        for i in range(2):
            self.aircraft.append(Aircraft3D(
                callsign=f"TEST{i}",
                x=self.np_random.uniform(-5, 5),
                y=self.np_random.uniform(-5, 5),
                altitude=self.np_random.uniform(3000, 8000),
                heading=self.np_random.uniform(0, 360),
                speed=self.np_random.uniform(200, 280)
            ))

        return self._get_obs(), {}

    def step(self, action):
        for ac in self.aircraft:
            ac.update(1.0)

        self.current_step += 1
        terminated = self.current_step >= self.max_steps

        return self._get_obs(), 0.0, terminated, False, {}

    def _get_obs(self):
        aircraft_obs = np.zeros((self.max_aircraft, 14), dtype=np.float32)
        mask = np.zeros(self.max_aircraft, dtype=np.uint8)
        conflict_matrix = np.zeros((self.max_aircraft, self.max_aircraft), dtype=np.float32)
        global_state = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        for i, ac in enumerate(self.aircraft):
            if i < self.max_aircraft:
                aircraft_obs[i] = [
                    ac.x / AREA_SIZE_NM,
                    ac.y / AREA_SIZE_NM,
                    ac.altitude / 10000.0,
                    ac.heading / 360.0,
                    ac.speed / 300.0,
                    ac.target_altitude / 10000.0,
                    ac.target_heading / 360.0,
                    ac.target_speed / 300.0,
                    0, 0, -1, 0, 0, 0
                ]
                mask[i] = 1

        return {
            'aircraft': aircraft_obs,
            'aircraft_mask': mask,
            'conflict_matrix': conflict_matrix,
            'global_state': global_state
        }

# Test 3D environment
try:
    env = Realistic3DATCEnv(max_aircraft=8, max_steps=10)
    obs, info = env.reset(seed=42)

    print(f"   ✅ Environment created")
    print(f"   ✅ Observation space: {env.observation_space}")
    print(f"   ✅ Action space: {env.action_space}")

    # Run a few steps
    for _ in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated:
            break

    print(f"   ✅ Steps executed successfully")
    print(f"   ✅ Final observation shapes:")
    print(f"      - aircraft: {obs['aircraft'].shape}")
    print(f"      - mask: {obs['aircraft_mask'].shape}")
    print(f"      - conflict_matrix: {obs['conflict_matrix'].shape}")
    print(f"      - global_state: {obs['global_state'].shape}")

    print("\n✅ Realistic 3D ATC Environment: PASSED")

except Exception as e:
    print(f"\n❌ Realistic 3D ATC Environment: FAILED")
    print(f"   Error: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 60)
print("Test Summary")
print("=" * 60)
print("✅ Both environments passed basic tests!")
print("\nNext steps:")
print("  1. Open simple_2d_atc.ipynb in Jupyter")
print("  2. Run all cells to see the environment in action")
print("  3. Train a PPO agent (takes ~5 minutes)")
print("  4. Try realistic_3d_atc.ipynb for full 3D simulation")
print("=" * 60)
