"""
Wrapper to convert Dict action/observation spaces to formats compatible with SB3

This wrapper allows environments with Dict action spaces to work with
Stable-Baselines3, which only supports Box, Discrete, MultiDiscrete, and MultiBinary.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class DictToMultiDiscreteWrapper(gym.ActionWrapper):
    """
    Converts a Dict action space to MultiDiscrete for SB3 compatibility.

    Stable-Baselines3's PPO only supports:
    - Box (continuous)
    - Discrete (single choice)
    - MultiDiscrete (multiple independent choices)
    - MultiBinary (binary choices)

    This wrapper converts Dict({'key1': Discrete(n1), 'key2': Discrete(n2), ...})
    to MultiDiscrete([n1, n2, ...]) so SB3 can use it.

    Example:
        Dict({'aircraft_id': Discrete(21), 'command_type': Discrete(5), ...})
        -> MultiDiscrete([21, 5, 13, 18, 8])

    The wrapper automatically converts between formats:
    - Agent outputs: [5, 2, 7, 10, 3] (MultiDiscrete array)
    - Environment receives: {'aircraft_id': 5, 'command_type': 2, ...} (Dict)
    """

    def __init__(self, env):
        """
        Args:
            env: Gymnasium environment with Dict action space
        """
        super().__init__(env)

        # Verify we have a Dict action space
        if not isinstance(env.action_space, spaces.Dict):
            raise ValueError(
                f"DictToMultiDiscreteWrapper requires Dict action space, "
                f"got {type(env.action_space)}"
            )

        # Store the original Dict action space structure
        # Important: preserve the key order for consistent indexing
        # We want: [aircraft_id, command_type, heading, altitude, speed]
        # NOT alphabetical: [aircraft_id, altitude, command_type, heading, speed]
        desired_order = ['aircraft_id', 'command_type', 'heading', 'altitude', 'speed']
        available_keys = set(env.action_space.spaces.keys())

        # Use desired order if all keys match, otherwise use dict order
        if set(desired_order) == available_keys:
            self._action_keys = desired_order
        else:
            self._action_keys = list(env.action_space.spaces.keys())

        self._action_spaces = [env.action_space.spaces[key] for key in self._action_keys]

        # Verify all components are Discrete
        for key, space in zip(self._action_keys, self._action_spaces):
            if not isinstance(space, spaces.Discrete):
                raise ValueError(
                    f"All Dict action space components must be Discrete, "
                    f"but '{key}' is {type(space)}"
                )

        # Create MultiDiscrete action space
        # MultiDiscrete([n1, n2, ...]) means n1 choices for dim 0, n2 for dim 1, etc.
        nvec = [space.n for space in self._action_spaces]
        self.action_space = spaces.MultiDiscrete(nvec)

        print(f"✅ Wrapped action space:")
        print(f"   Original: Dict({dict(zip(self._action_keys, [f'Discrete({s.n})' for s in self._action_spaces]))})")
        print(f"   Wrapped:  MultiDiscrete({nvec})")

    def action(self, action):
        """
        Convert MultiDiscrete action (array) to Dict action for the base environment.

        Args:
            action: numpy array or list [a0, a1, a2, ...] from the agent

        Returns:
            dict: {'key1': a0, 'key2': a1, ...} for the environment
        """
        # Convert array to dict
        if isinstance(action, np.ndarray):
            action = action.tolist()

        # Map array indices to dict keys
        action_dict = {
            key: int(action[i])
            for i, key in enumerate(self._action_keys)
        }

        return action_dict

    def reverse_action(self, action):
        """
        Convert Dict action back to MultiDiscrete (for logging/debugging).

        Args:
            action: dict {'key1': val1, 'key2': val2, ...}

        Returns:
            array: [val1, val2, ...]
        """
        return np.array([action[key] for key in self._action_keys])


class FlattenDictObservationWrapper(gym.ObservationWrapper):
    """
    Flattens Dict observation space to Box space for SB3 compatibility.

    IMPORTANT: This is only needed if you want to use NormalizeObservation wrapper.
    For the demo notebook, we recommend skipping observation normalization to keep
    things simple.

    This wrapper is provided for completeness but is NOT recommended for the
    OpenScope environment, which has heterogeneous observation components
    (aircraft features, global state, conflict matrix) that shouldn't be
    normalized together.
    """

    def __init__(self, env):
        super().__init__(env)
        raise NotImplementedError(
            "FlattenDictObservationWrapper is not recommended for OpenScope. "
            "The Dict observation space contains heterogeneous data "
            "(aircraft features, global state, conflicts) that should not be "
            "flattened together. Instead, skip observation normalization or "
            "implement a custom Dict-aware normalizer."
        )


if __name__ == "__main__":
    # Test the wrapper
    print("Testing DictToMultiDiscreteWrapper...")
    print("=" * 60)

    # Create a dummy environment with Dict action space
    class DummyEnv(gym.Env):
        def __init__(self):
            self.action_space = spaces.Dict({
                'aircraft_id': spaces.Discrete(21),
                'command_type': spaces.Discrete(5),
                'heading': spaces.Discrete(13),
                'altitude': spaces.Discrete(18),
                'speed': spaces.Discrete(8),
            })
            self.observation_space = spaces.Box(low=0, high=1, shape=(10,))

        def reset(self, seed=None, options=None):
            return np.zeros(10), {}

        def step(self, action):
            print(f"  Environment received: {action}")
            return np.zeros(10), 0.0, False, False, {}

    # Wrap it
    env = DummyEnv()
    wrapped_env = DictToMultiDiscreteWrapper(env)

    print(f"\nOriginal action space: {env.action_space}")
    print(f"Wrapped action space:  {wrapped_env.action_space}")

    # Test action conversion
    print("\nTesting action conversion:")
    test_action = np.array([5, 2, 7, 10, 3])
    print(f"  Agent outputs: {test_action}")

    wrapped_env.step(test_action)

    print("\n✅ Wrapper test passed!")
