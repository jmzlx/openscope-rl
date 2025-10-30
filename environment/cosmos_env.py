"""
Cosmos-based OpenScope Environment.

This module provides a Gymnasium-compatible environment that uses a fine-tuned
NVIDIA Cosmos model to simulate OpenScope dynamics. This is much faster than
running the actual browser-based game, enabling rapid RL training.

The environment:
1. Uses Cosmos to predict next video frames given current frame + action
2. Extracts game state from generated frames (using vision model or shadow state)
3. Computes rewards using a learned reward model
4. Provides standard Gymnasium interface for RL algorithms
"""

import logging
from typing import Any, Dict, Optional, Tuple
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import cv2

from environment.config import OpenScopeConfig, create_default_config, CommandType
from environment.spaces import create_observation_space, create_action_space
from environment.state_processor import StateProcessor
from environment.metrics import EpisodeMetrics

logger = logging.getLogger(__name__)


class RewardModel(nn.Module):
    """
    Learned reward model for Cosmos environment.

    This model predicts rewards from state + action, trained on
    collected OpenScope episodes.
    """

    def __init__(
        self,
        state_dim: int = 128,
        action_dim: int = 5,
        hidden_dim: int = 256,
    ):
        """
        Initialize reward model.

        Args:
            state_dim: State representation dimension
            action_dim: Action dimension
            hidden_dim: Hidden layer dimension
        """
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Predict reward for state-action pair.

        Args:
            state: State representation (batch_size, state_dim)
            action: Action representation (batch_size, action_dim)

        Returns:
            Predicted reward (batch_size, 1)
        """
        x = torch.cat([state, action], dim=-1)
        reward = self.encoder(x)
        return reward


class StateExtractor(nn.Module):
    """
    Extract game state from video frames.

    This model uses a CNN to extract aircraft positions, conflicts, etc.
    from rendered frames. An alternative approach would be to maintain
    a shadow state synchronized with Cosmos predictions.
    """

    def __init__(
        self,
        frame_shape: Tuple[int, int, int] = (3, 720, 1280),
        state_dim: int = 128,
    ):
        """
        Initialize state extractor.

        Args:
            frame_shape: Input frame shape (C, H, W)
            state_dim: Output state dimension
        """
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(frame_shape[0], 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, state_dim),
        )

    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        """
        Extract state representation from frame.

        Args:
            frame: Input frame (batch_size, C, H, W)

        Returns:
            State representation (batch_size, state_dim)
        """
        return self.encoder(frame)


class CosmosOpenScopeEnv(gym.Env):
    """
    OpenScope environment simulated by Cosmos world model.

    This environment is much faster than the real OpenScope environment
    because it doesn't require browser automation or JavaScript execution.
    It uses a fine-tuned Cosmos model to predict next frames, and extracts
    game state from those frames.

    Example:
        >>> env = CosmosOpenScopeEnv(
        ...     cosmos_model_path="cosmos_finetuned/best_model.pt",
        ...     reward_model_path="reward_model/best_model.pt",
        ...     airport="KJFK",
        ...     max_aircraft=10
        ... )
        >>> obs, info = env.reset()
        >>> action = env.action_space.sample()
        >>> obs, reward, terminated, truncated, info = env.step(action)
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(
        self,
        cosmos_model_path: str,
        reward_model_path: Optional[str] = None,
        state_extractor_path: Optional[str] = None,
        airport: str = "KLAS",
        max_aircraft: int = 10,
        episode_length: int = 1800,
        action_interval: float = 5.0,
        frame_size: Tuple[int, int] = (1280, 720),
        device: Optional[str] = None,
        use_shadow_state: bool = True,  # Use shadow state instead of vision model
    ):
        """
        Initialize Cosmos-based environment.

        Args:
            cosmos_model_path: Path to fine-tuned Cosmos model
            reward_model_path: Path to reward model (None = use heuristic rewards)
            state_extractor_path: Path to state extractor model (if not using shadow state)
            airport: ICAO airport code
            max_aircraft: Maximum number of aircraft
            episode_length: Maximum episode length in seconds
            action_interval: Time between actions in seconds
            frame_size: Frame size (width, height)
            device: Device to run models on (None = auto-detect)
            use_shadow_state: Whether to use shadow state (faster) or vision model
        """
        super().__init__()

        self.airport = airport
        self.max_aircraft = max_aircraft
        self.episode_length = episode_length
        self.action_interval = action_interval
        self.frame_size = frame_size
        self.use_shadow_state = use_shadow_state

        # Device setup
        from .utils import get_device
        if device is None:
            self.device = torch.device(get_device())
        else:
            self.device = torch.device(get_device(device))

        logger.info(f"CosmosOpenScopeEnv using device: {self.device}")

        # Load models
        self.cosmos_model = self._load_cosmos_model(cosmos_model_path)
        self.reward_model = self._load_reward_model(reward_model_path)

        if not use_shadow_state:
            self.state_extractor = self._load_state_extractor(state_extractor_path)
        else:
            self.state_extractor = None

        # Configuration
        self.config = create_default_config(
            airport=airport,
            max_aircraft=max_aircraft,
            episode_length=episode_length,
            action_interval=action_interval,
        )

        # Define spaces
        self.observation_space = create_observation_space(max_aircraft)
        self.action_space = create_action_space(max_aircraft)

        # State processor (for converting shadow state to observation)
        self.state_processor = StateProcessor(self.config)

        # Episode tracking
        self.episode_metrics = EpisodeMetrics()

        # Episode state
        self.current_frame: Optional[np.ndarray] = None
        self.shadow_state: Dict[str, Any] = {}
        self.current_step = 0
        self.simulated_time = 0.0

        logger.info(f"CosmosOpenScopeEnv initialized: {airport}, {max_aircraft} aircraft")

    def _load_cosmos_model(self, model_path: str):
        """Load fine-tuned Cosmos model."""
        model_path = Path(model_path)

        if not model_path.exists():
            logger.warning(f"Cosmos model not found: {model_path}")
            logger.warning("Using placeholder model")
            return self._create_placeholder_cosmos_model()

        try:
            checkpoint = torch.load(model_path, map_location=self.device)

            # This would be the actual Cosmos model loading
            # from nvidia_cosmos import CosmosWFM
            # model = CosmosWFM.from_checkpoint(checkpoint)

            # For now, use placeholder
            model = self._create_placeholder_cosmos_model()
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(self.device)
            model.eval()

            logger.info(f"Loaded Cosmos model from: {model_path}")
            return model

        except Exception as e:
            logger.error(f"Failed to load Cosmos model: {e}")
            return self._create_placeholder_cosmos_model()

    def _create_placeholder_cosmos_model(self):
        """Create placeholder Cosmos model for testing."""

        class PlaceholderCosmosModel(nn.Module):
            def __init__(self):
                super().__init__()
                # Simple identity model (returns input with slight noise)
                self.conv = nn.Conv2d(3, 3, 3, padding=1)

            def forward(self, current_frame, action=None):
                # Add slight variation to current frame
                return self.conv(current_frame) + current_frame * 0.95

        model = PlaceholderCosmosModel()
        model.to(self.device)
        model.eval()
        return model

    def _load_reward_model(self, model_path: Optional[str]):
        """Load reward model."""
        if model_path is None:
            logger.info("No reward model provided, using heuristic rewards")
            return None

        model_path = Path(model_path)

        if not model_path.exists():
            logger.warning(f"Reward model not found: {model_path}, using heuristic")
            return None

        try:
            model = RewardModel()
            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(self.device)
            model.eval()

            logger.info(f"Loaded reward model from: {model_path}")
            return model

        except Exception as e:
            logger.error(f"Failed to load reward model: {e}")
            return None

    def _load_state_extractor(self, model_path: Optional[str]):
        """Load state extractor model."""
        if model_path is None:
            logger.info("No state extractor provided, creating new one")
            model = StateExtractor()
            model.to(self.device)
            model.eval()
            return model

        model_path = Path(model_path)

        if not model_path.exists():
            logger.warning(f"State extractor not found: {model_path}, creating new")
            model = StateExtractor()
            model.to(self.device)
            model.eval()
            return model

        try:
            model = StateExtractor()
            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(self.device)
            model.eval()

            logger.info(f"Loaded state extractor from: {model_path}")
            return model

        except Exception as e:
            logger.error(f"Failed to load state extractor: {e}")
            model = StateExtractor()
            model.to(self.device)
            model.eval()
            return model

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Reset environment to initial state.

        Args:
            seed: Random seed
            options: Additional options

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)

        # Reset episode state
        self.current_step = 0
        self.simulated_time = 0.0
        self.episode_metrics.start_episode()

        # Initialize with a blank frame (or load initial frame from dataset)
        self.current_frame = self._generate_initial_frame()

        # Initialize shadow state
        self.shadow_state = self._initialize_shadow_state()

        # Get observation
        observation = self.state_processor.process_state(self.shadow_state)

        info = {
            "simulated_time": self.simulated_time,
            "current_step": self.current_step,
            "shadow_state": self.shadow_state,
        }

        logger.debug("Environment reset")
        return observation, info

    def step(
        self,
        action: Dict[str, int]
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Execute action and return next state.

        Args:
            action: Action dictionary

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Convert action to tensor
        action_tensor = self._encode_action(action)

        # Convert current frame to tensor
        frame_tensor = torch.from_numpy(self.current_frame).float() / 255.0
        frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
        frame_tensor = frame_tensor.to(self.device)

        # Predict next frame using Cosmos
        with torch.no_grad():
            next_frame_tensor = self.cosmos_model(frame_tensor, action_tensor)

        # Convert back to numpy
        next_frame = next_frame_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        next_frame = (next_frame * 255.0).clip(0, 255).astype(np.uint8)

        # Extract state from frame
        if self.use_shadow_state:
            # Update shadow state based on action
            self.shadow_state = self._update_shadow_state(self.shadow_state, action)
        else:
            # Extract state using vision model
            with torch.no_grad():
                state_representation = self.state_extractor(next_frame_tensor)
            self.shadow_state = self._state_representation_to_dict(state_representation)

        # Get observation
        observation = self.state_processor.process_state(self.shadow_state)

        # Calculate reward
        reward = self._calculate_reward(self.shadow_state, action)

        # Update episode state
        self.current_frame = next_frame
        self.current_step += 1
        self.simulated_time += self.action_interval
        self.episode_metrics.add_reward(reward)
        self.episode_metrics.increment_step()

        # Check termination
        terminated = self._check_terminated(self.shadow_state)
        truncated = (
            self.simulated_time >= self.episode_length or
            self.current_step >= 1000
        )

        info = {
            "simulated_time": self.simulated_time,
            "current_step": self.current_step,
            "shadow_state": self.shadow_state,
            "episode_metrics": self.episode_metrics.to_dict(),
        }

        return observation, reward, terminated, truncated, info

    def render(self) -> np.ndarray:
        """
        Render the current frame.

        Returns:
            Current frame as RGB array
        """
        if self.current_frame is None:
            return np.zeros((self.frame_size[1], self.frame_size[0], 3), dtype=np.uint8)
        return self.current_frame

    def close(self):
        """Clean up resources."""
        # Nothing to clean up for Cosmos environment
        pass

    def _generate_initial_frame(self) -> np.ndarray:
        """Generate initial frame for episode start."""
        # For now, return a blank frame
        # In practice, you would load an initial frame from the dataset
        # or use Cosmos to generate one
        return np.zeros((self.frame_size[1], self.frame_size[0], 3), dtype=np.uint8)

    def _initialize_shadow_state(self) -> Dict[str, Any]:
        """Initialize shadow state for new episode."""
        # Create initial state with a few aircraft
        num_initial_aircraft = np.random.randint(1, min(4, self.max_aircraft + 1))

        aircraft = []
        for i in range(num_initial_aircraft):
            aircraft.append({
                "callsign": f"N{1000 + i}",
                "position": [np.random.uniform(-10, 10), np.random.uniform(-10, 10)],
                "altitude": np.random.randint(100, 400) * 100,
                "heading": np.random.uniform(0, 360),
                "speed": np.random.uniform(200, 300),
                "category": "arrival" if np.random.rand() > 0.5 else "departure",
            })

        return {
            "aircraft": aircraft,
            "conflicts": [],
            "score": 1000,
            "time": 0.0,
        }

    def _update_shadow_state(
        self,
        state: Dict[str, Any],
        action: Dict[str, int]
    ) -> Dict[str, Any]:
        """
        Update shadow state based on action.

        This maintains a simplified physics model synchronized with Cosmos.
        """
        # Make a copy
        new_state = {
            "aircraft": [ac.copy() for ac in state["aircraft"]],
            "conflicts": [],
            "score": state["score"],
            "time": state["time"] + self.action_interval,
        }

        # Apply action to selected aircraft
        aircraft_id = action.get("aircraft_id", 0)
        if aircraft_id < len(new_state["aircraft"]):
            aircraft = new_state["aircraft"][aircraft_id]

            command_type_idx = action.get("command_type", 0)
            if command_type_idx == 0:  # ALTITUDE
                target_altitude = action.get("altitude_value", 100) * 100
                aircraft["altitude"] = target_altitude

            elif command_type_idx == 1:  # HEADING
                heading_change = action.get("heading_change", 0)
                aircraft["heading"] = (aircraft["heading"] + heading_change) % 360

            elif command_type_idx == 2:  # SPEED
                target_speed = action.get("speed_value", 250)
                aircraft["speed"] = target_speed

        # Simple physics update for all aircraft
        for aircraft in new_state["aircraft"]:
            # Update position based on heading and speed
            heading_rad = np.radians(aircraft["heading"])
            distance = aircraft["speed"] * self.action_interval / 3600.0  # Convert to nautical miles
            aircraft["position"][0] += distance * np.sin(heading_rad)
            aircraft["position"][1] += distance * np.cos(heading_rad)

        # Check for conflicts (simple distance check)
        conflicts = []
        for i in range(len(new_state["aircraft"])):
            for j in range(i + 1, len(new_state["aircraft"])):
                ac1 = new_state["aircraft"][i]
                ac2 = new_state["aircraft"][j]

                # Horizontal separation
                dx = ac1["position"][0] - ac2["position"][0]
                dy = ac1["position"][1] - ac2["position"][1]
                horizontal_distance = np.sqrt(dx**2 + dy**2)

                # Vertical separation
                vertical_distance = abs(ac1["altitude"] - ac2["altitude"]) / 100.0  # In flight levels

                if horizontal_distance < 5.0 and vertical_distance < 10.0:
                    conflicts.append({
                        "aircraft": [ac1["callsign"], ac2["callsign"]],
                        "hasViolation": horizontal_distance < 3.0 and vertical_distance < 5.0,
                    })

        new_state["conflicts"] = conflicts

        # Update score based on conflicts
        for conflict in conflicts:
            if conflict["hasViolation"]:
                new_state["score"] -= 100
            else:
                new_state["score"] -= 5

        return new_state

    def _encode_action(self, action: Dict[str, int]) -> torch.Tensor:
        """Encode action dictionary as tensor."""
        action_components = [
            action.get("aircraft_id", 0),
            action.get("command_type", 0),
            action.get("altitude_value", 0),
            action.get("heading_change", 0),
            action.get("speed_value", 0),
        ]

        action_tensor = torch.tensor(action_components, dtype=torch.float32)
        action_tensor = action_tensor / torch.tensor([20.0, 5.0, 18.0, 13.0, 8.0])
        action_tensor = action_tensor.unsqueeze(0).to(self.device)

        return action_tensor

    def _calculate_reward(self, state: Dict[str, Any], action: Dict[str, int]) -> float:
        """Calculate reward for current state."""
        if self.reward_model is not None:
            # Use learned reward model
            with torch.no_grad():
                # Extract state representation
                state_repr = self._extract_state_representation(state)
                action_tensor = self._encode_action(action)

                reward_tensor = self.reward_model(state_repr, action_tensor)
                return reward_tensor.item()
        else:
            # Use heuristic reward
            reward = 0.0

            # Timestep penalty
            reward -= 0.01

            # Conflict penalty
            for conflict in state.get("conflicts", []):
                if conflict.get("hasViolation", False):
                    reward -= 200.0
                else:
                    reward -= 2.0

            # Score-based reward
            score_change = state.get("score", 0) - 1000  # Relative to initial score
            reward += score_change * 0.01

            return reward

    def _extract_state_representation(self, state: Dict[str, Any]) -> torch.Tensor:
        """Extract state representation for reward model."""
        # Simple encoding: count aircraft, conflicts, etc.
        features = [
            len(state.get("aircraft", [])),
            len(state.get("conflicts", [])),
            state.get("score", 0) / 1000.0,
            state.get("time", 0) / 1800.0,
        ]

        # Pad to state_dim
        features += [0.0] * (128 - len(features))

        state_tensor = torch.tensor(features[:128], dtype=torch.float32)
        state_tensor = state_tensor.unsqueeze(0).to(self.device)

        return state_tensor

    def _state_representation_to_dict(self, state_repr: torch.Tensor) -> Dict[str, Any]:
        """Convert state representation to dictionary (when using vision model)."""
        # This would parse the state representation into a structured dict
        # For now, return a minimal state
        return {
            "aircraft": [],
            "conflicts": [],
            "score": 1000,
            "time": self.simulated_time,
        }

    def _check_terminated(self, state: Dict[str, Any]) -> bool:
        """Check if episode should terminate."""
        # Terminate on very low score
        return state.get("score", 0) < -1000
