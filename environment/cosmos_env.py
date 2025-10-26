"""
OpenScope environment simulated by Cosmos.

This module implements a Gymnasium-compatible environment that uses a fine-tuned
Cosmos World Foundation Model to simulate OpenScope ATC dynamics, enabling much
faster RL training compared to browser-based simulation.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces

try:
    from nvidia_cosmos import CosmosWFM
    COSMOS_AVAILABLE = True
except ImportError:
    COSMOS_AVAILABLE = False


logger = logging.getLogger(__name__)


class StateExtractor(nn.Module):
    """
    Extract game state from Cosmos-generated frames.

    This uses a vision model to detect and localize aircraft from rendered frames.
    """

    def __init__(self, frame_size: Tuple[int, int] = (512, 512), max_aircraft: int = 10):
        """
        Initialize state extractor.

        Args:
            frame_size: Size of input frames (H, W)
            max_aircraft: Maximum number of aircraft to track
        """
        super().__init__()
        self.frame_size = frame_size
        self.max_aircraft = max_aircraft

        # Simple CNN for state extraction
        # In practice, could use a pre-trained vision model (YOLO, SAM, etc.)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        # Output heads for aircraft features
        # Each aircraft: [x, y, heading, speed, altitude, ...]
        self.aircraft_predictor = nn.Linear(256, max_aircraft * 14)

    def forward(self, frame: torch.Tensor) -> np.ndarray:
        """
        Extract state from frame.

        Args:
            frame: Frame tensor [C, H, W]

        Returns:
            State array [max_aircraft, 14]
        """
        # Add batch dimension
        if frame.dim() == 3:
            frame = frame.unsqueeze(0)

        # Encode
        features = self.encoder(frame)
        features = features.view(features.size(0), -1)

        # Predict aircraft states
        aircraft_states = self.aircraft_predictor(features)
        aircraft_states = aircraft_states.view(-1, self.max_aircraft, 14)

        # Remove batch dimension and convert to numpy
        state = aircraft_states[0].detach().cpu().numpy()

        return state


class RewardModel(nn.Module):
    """
    Learned reward model for Cosmos environment.

    Predicts rewards from state-action pairs, trained on collected episodes.
    """

    def __init__(self, state_dim: int = 140, action_dim: int = 4, hidden_dim: int = 256):
        """
        Initialize reward model.

        Args:
            state_dim: State dimensionality
            action_dim: Action dimensionality
            hidden_dim: Hidden layer size
        """
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Predict reward.

        Args:
            state: State tensor [B, state_dim]
            action: Action tensor [B, action_dim]

        Returns:
            Predicted reward [B, 1]
        """
        # Concatenate state and action
        state_action = torch.cat([state, action], dim=-1)

        # Predict reward
        reward = self.network(state_action)

        return reward


class CosmosOpenScopeEnv(gym.Env):
    """
    OpenScope environment using Cosmos as world model.

    This is MUCH faster than real OpenScope (no browser, no JavaScript!).
    Enables rapid RL training with 10-100x speedup.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 1}

    def __init__(
        self,
        cosmos_model_path: str = "cosmos-openscope-finetuned",
        max_aircraft: int = 10,
        max_steps: int = 300,
        frame_size: Tuple[int, int] = (512, 512),
        device: Optional[str] = None,
        render_mode: Optional[str] = None,
    ):
        """
        Initialize Cosmos-based environment.

        Args:
            cosmos_model_path: Path to fine-tuned Cosmos model
            max_aircraft: Maximum number of aircraft
            max_steps: Maximum steps per episode
            frame_size: Size of frames (H, W)
            device: Device to run on ('cuda' or 'cpu')
            render_mode: Rendering mode
        """
        super().__init__()

        if not COSMOS_AVAILABLE:
            logger.warning("nvidia-cosmos not installed. Using placeholder implementation.")

        self.max_aircraft = max_aircraft
        self.max_steps = max_steps
        self.frame_size = frame_size
        self.render_mode = render_mode

        # Device setup
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"Initializing Cosmos environment on {self.device}")

        # Load fine-tuned Cosmos model
        self.cosmos_model = self._load_cosmos_model(cosmos_model_path)

        # State extractor (vision model)
        self.state_extractor = StateExtractor(frame_size, max_aircraft).to(self.device)

        # Reward model
        self.reward_model = RewardModel().to(self.device)

        # Current frame
        self.current_frame = None
        self.current_state = None
        self.step_count = 0

        # Define observation space (same as PlaywrightEnv)
        # Each aircraft has 14 features: [x, y, heading, speed, altitude, ...]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(max_aircraft, 14),
            dtype=np.float32,
        )

        # Define action space (MultiDiscrete for ATC commands)
        # [aircraft_id, command_type, param1, param2]
        self.action_space = spaces.MultiDiscrete([max_aircraft, 10, 100, 100])

        logger.info("Cosmos environment initialized successfully")

    def _load_cosmos_model(self, model_path: str):
        """Load fine-tuned Cosmos model."""
        if COSMOS_AVAILABLE:
            # NOTE: Actual API may differ
            # model = CosmosWFM.from_pretrained(model_path)
            # return model.to(self.device).eval()
            pass

        # Placeholder model
        logger.warning("Using placeholder Cosmos model")

        class DummyCosmosModel(nn.Module):
            def __init__(self, frame_size):
                super().__init__()
                self.frame_size = frame_size

            def generate(self, current_frame, action_embedding):
                # Return same frame with slight noise
                noise = torch.randn_like(current_frame) * 0.01
                return torch.clamp(current_frame + noise, 0, 1)

        return DummyCosmosModel(self.frame_size).to(self.device).eval()

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset environment to initial state.

        Args:
            seed: Random seed
            options: Additional options

        Returns:
            Initial observation and info dict
        """
        super().reset(seed=seed)

        # Generate or load initial frame
        self.current_frame = self._get_initial_frame()
        self.step_count = 0

        # Extract state from initial frame
        self.current_state = self._extract_state(self.current_frame)

        info = {
            'step': self.step_count,
            'cosmos_generated': False,  # Initial frame is not generated
        }

        return self.current_state.copy(), info

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take action in Cosmos-simulated environment.

        Args:
            action: Action to take [aircraft_id, command_type, param1, param2]

        Returns:
            observation, reward, terminated, truncated, info
        """
        # Generate next frame using Cosmos
        next_frame = self._predict_next_frame(self.current_frame, action)

        # Extract state from generated frame
        next_state = self._extract_state(next_frame)

        # Compute reward using reward model
        reward = self._compute_reward(next_state, action)

        # Check termination
        self.step_count += 1
        terminated = self._check_done(next_state)
        truncated = self.step_count >= self.max_steps

        # Update current frame and state
        self.current_frame = next_frame
        self.current_state = next_state

        info = {
            'step': self.step_count,
            'cosmos_generated': True,
        }

        return next_state.copy(), reward, terminated, truncated, info

    def _predict_next_frame(
        self,
        current_frame: torch.Tensor,
        action: np.ndarray,
    ) -> torch.Tensor:
        """
        Use Cosmos to predict next frame.

        Args:
            current_frame: Current frame tensor
            action: Action taken

        Returns:
            Next frame tensor
        """
        with torch.no_grad():
            # Encode action
            action_tensor = torch.FloatTensor(action).to(self.device)
            action_embedding = self._encode_action(action_tensor)

            # Predict next frame
            # NOTE: Actual API may differ
            next_frame = self.cosmos_model.generate(
                current_frame,
                action_embedding=action_embedding,
            )

        return next_frame

    def _extract_state(self, frame: torch.Tensor) -> np.ndarray:
        """
        Extract game state from frame.

        Args:
            frame: Frame tensor [C, H, W]

        Returns:
            State array [max_aircraft, 14]
        """
        with torch.no_grad():
            state = self.state_extractor(frame)

        return state

    def _compute_reward(self, state: np.ndarray, action: np.ndarray) -> float:
        """
        Compute reward using learned reward model.

        Args:
            state: Current state
            action: Action taken

        Returns:
            Reward value
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state.flatten()).unsqueeze(0).to(self.device)
            action_tensor = torch.FloatTensor(action).unsqueeze(0).to(self.device)

            reward_tensor = self.reward_model(state_tensor, action_tensor)
            reward = reward_tensor.item()

        return reward

    def _encode_action(self, action: torch.Tensor) -> torch.Tensor:
        """
        Encode action for Cosmos.

        Args:
            action: Action tensor

        Returns:
            Action embedding
        """
        # Simple one-hot encoding + normalization
        # In practice, could use learned action encoder
        # For now, use simple approach
        return action.unsqueeze(0)  # Add batch dimension

    def _get_initial_frame(self) -> torch.Tensor:
        """
        Get initial frame for episode start.

        Returns:
            Initial frame tensor [C, H, W]
        """
        # Could be random frame from training data
        # For now, return random noise as placeholder
        frame = torch.rand(3, *self.frame_size).to(self.device)

        return frame

    def _check_done(self, state: np.ndarray) -> bool:
        """
        Check if episode is done.

        Args:
            state: Current state

        Returns:
            Whether episode is terminated
        """
        # Simple heuristic: done if no aircraft (all states are zero)
        if np.all(np.abs(state) < 1e-6):
            return True

        return False

    def render(self):
        """
        Render environment.

        Returns:
            RGB array if render_mode is 'rgb_array'
        """
        if self.render_mode == "rgb_array":
            # Convert frame to numpy array
            if self.current_frame is not None:
                frame = self.current_frame.permute(1, 2, 0).cpu().numpy()
                frame = (frame * 255).astype(np.uint8)
                return frame

        return None

    def close(self):
        """Clean up resources."""
        pass

    def train_reward_model(
        self,
        data_dir: str = "cosmos_data",
        epochs: int = 10,
        batch_size: int = 32,
        lr: float = 1e-3,
    ):
        """
        Train reward model on collected episodes.

        Args:
            data_dir: Directory containing collected data
            epochs: Number of training epochs
            batch_size: Batch size
            lr: Learning rate
        """
        logger.info("Training reward model...")

        # Load collected data
        data_path = Path(data_dir)
        metadata_files = sorted((data_path / "metadata").glob("episode_*.npz"))

        # Prepare training data
        states = []
        actions = []
        rewards = []

        for metadata_file in metadata_files:
            data = np.load(metadata_file, allow_pickle=True)
            obs = data['observations']
            act = data['actions']
            rew = data['rewards']

            # Flatten observations
            obs_flat = [o.flatten() for o in obs[:-1]]  # Exclude last observation
            states.extend(obs_flat)
            actions.extend(act)
            rewards.extend(rew)

        states = torch.FloatTensor(np.array(states))
        actions = torch.FloatTensor(np.array(actions))
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1)

        # Create dataset and dataloader
        dataset = torch.utils.data.TensorDataset(states, actions, rewards)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
        )

        # Train
        self.reward_model.train()
        optimizer = torch.optim.Adam(self.reward_model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        for epoch in range(epochs):
            total_loss = 0.0
            num_batches = 0

            for state_batch, action_batch, reward_batch in dataloader:
                state_batch = state_batch.to(self.device)
                action_batch = action_batch.to(self.device)
                reward_batch = reward_batch.to(self.device)

                # Forward pass
                pred_reward = self.reward_model(state_batch, action_batch)

                # Compute loss
                loss = criterion(pred_reward, reward_batch)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches
            logger.info(f"Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.4f}")

        self.reward_model.eval()
        logger.info("Reward model training complete!")


def test_cosmos_env():
    """Test Cosmos environment."""
    env = CosmosOpenScopeEnv()

    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial info: {info}")

    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        print(f"Step {i+1}: reward={reward:.2f}, done={terminated or truncated}")

        if terminated or truncated:
            break

    env.close()
    print("Test complete!")


if __name__ == "__main__":
    test_cosmos_env()
