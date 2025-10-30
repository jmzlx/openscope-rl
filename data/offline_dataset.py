"""
Offline dataset collection and management for Decision Transformer.

This module provides utilities for collecting episodes from various policies
and creating PyTorch datasets for offline RL training.
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class Episode:
    """Single episode data container."""

    observations: List[Dict[str, np.ndarray]] = field(default_factory=list)
    actions: List[Dict[str, int]] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)
    infos: List[Dict] = field(default_factory=list)

    # Computed fields
    returns_to_go: Optional[np.ndarray] = None
    total_return: Optional[float] = None
    length: Optional[int] = None

    def compute_returns_to_go(self):
        """Compute return-to-go for each timestep."""
        rtgs = []
        rtg = 0.0

        # Iterate backwards to compute cumulative future rewards
        for reward in reversed(self.rewards):
            rtg += reward
            rtgs.insert(0, rtg)

        self.returns_to_go = np.array(rtgs, dtype=np.float32)
        self.total_return = float(rtgs[0]) if rtgs else 0.0
        self.length = len(self.rewards)

    def to_dict(self) -> Dict:
        """Convert to dictionary for saving."""
        return {
            "observations": self.observations,
            "actions": self.actions,
            "rewards": self.rewards,
            "dones": self.dones,
            "infos": self.infos,
            "returns_to_go": self.returns_to_go,
            "total_return": self.total_return,
            "length": self.length,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Episode":
        """Load from dictionary."""
        episode = cls(
            observations=data["observations"],
            actions=data["actions"],
            rewards=data["rewards"],
            dones=data["dones"],
            infos=data["infos"],
        )
        episode.returns_to_go = data.get("returns_to_go")
        episode.total_return = data.get("total_return")
        episode.length = data.get("length")
        return episode


class OfflineDatasetCollector:
    """
    Collect episodes from various policies for offline RL.

    This class runs multiple policies in the environment and collects
    diverse trajectories for Decision Transformer training.

    Example:
        >>> from poc.atc_rl import Simple2DATCEnv
        >>> env = Simple2DATCEnv(max_aircraft=5)
        >>> collector = OfflineDatasetCollector(env)
        >>>
        >>> # Collect random episodes
        >>> random_episodes = collector.collect_random_episodes(num_episodes=500)
        >>>
        >>> # Collect heuristic episodes
        >>> heuristic_episodes = collector.collect_heuristic_episodes(num_episodes=300)
        >>>
        >>> # Save all episodes
        >>> all_episodes = random_episodes + heuristic_episodes
        >>> collector.save_episodes(all_episodes, "offline_data.pkl")
    """

    def __init__(self, env):
        """
        Initialize dataset collector.

        Args:
            env: Gymnasium environment (OpenScope or POC environment)
        """
        self.env = env

    def collect_random_episodes(
        self, num_episodes: int, max_steps: int = 1000, verbose: bool = True
    ) -> List[Episode]:
        """
        Collect episodes using random policy.

        Args:
            num_episodes: Number of episodes to collect
            max_steps: Maximum steps per episode
            verbose: Show progress bar

        Returns:
            List of episodes
        """
        episodes = []

        iterator = tqdm(range(num_episodes), desc="Random policy") if verbose else range(num_episodes)

        for _ in iterator:
            episode = self._collect_episode(
                policy_fn=lambda obs: self.env.action_space.sample(),
                max_steps=max_steps,
            )
            episodes.append(episode)

        logger.info(f"Collected {num_episodes} random episodes")
        return episodes

    def collect_heuristic_episodes(
        self, num_episodes: int, max_steps: int = 1000, verbose: bool = True
    ) -> List[Episode]:
        """
        Collect episodes using simple heuristic policy.

        The heuristic tries to:
        1. Maintain safe separation between aircraft
        2. Guide aircraft toward their exit points
        3. Adjust altitude to avoid conflicts

        Args:
            num_episodes: Number of episodes to collect
            max_steps: Maximum steps per episode
            verbose: Show progress bar

        Returns:
            List of episodes
        """
        episodes = []

        iterator = tqdm(range(num_episodes), desc="Heuristic policy") if verbose else range(num_episodes)

        for _ in iterator:
            episode = self._collect_episode(
                policy_fn=self._heuristic_policy,
                max_steps=max_steps,
            )
            episodes.append(episode)

        logger.info(f"Collected {num_episodes} heuristic episodes")
        return episodes

    def collect_custom_policy_episodes(
        self,
        policy_fn: Callable,
        num_episodes: int,
        max_steps: int = 1000,
        verbose: bool = True,
        description: str = "Custom policy",
    ) -> List[Episode]:
        """
        Collect episodes using custom policy function.

        Args:
            policy_fn: Function that takes observation and returns action
            num_episodes: Number of episodes to collect
            max_steps: Maximum steps per episode
            verbose: Show progress bar
            description: Description for progress bar

        Returns:
            List of episodes
        """
        episodes = []

        iterator = tqdm(range(num_episodes), desc=description) if verbose else range(num_episodes)

        for _ in iterator:
            episode = self._collect_episode(
                policy_fn=policy_fn,
                max_steps=max_steps,
            )
            episodes.append(episode)

        logger.info(f"Collected {num_episodes} episodes with {description}")
        return episodes

    def _collect_episode(self, policy_fn: Callable, max_steps: int) -> Episode:
        """
        Collect a single episode.

        Args:
            policy_fn: Policy function
            max_steps: Maximum steps

        Returns:
            Episode data
        """
        episode = Episode()

        obs, info = self.env.reset()
        episode.observations.append(obs)

        for step in range(max_steps):
            # Get action from policy
            action = policy_fn(obs)

            # Step environment
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            # Store transition
            episode.actions.append(action)
            episode.rewards.append(float(reward))
            episode.dones.append(bool(done))
            episode.infos.append(info)

            if not done:
                episode.observations.append(next_obs)

            obs = next_obs

            if done:
                break

        # Compute returns-to-go
        episode.compute_returns_to_go()

        return episode

    def _heuristic_policy(self, obs: Dict[str, np.ndarray]) -> Dict[str, int]:
        """
        Simple heuristic policy for ATC.

        Strategy:
        1. Find aircraft pairs that are too close (potential conflicts)
        2. Adjust altitude or heading to increase separation
        3. Bias toward no-action to avoid over-controlling

        Args:
            obs: Observation dictionary

        Returns:
            Action dictionary
        """
        aircraft = obs["aircraft"]
        aircraft_mask = obs["aircraft_mask"]
        conflict_matrix = obs.get("conflict_matrix", None)

        # Count active aircraft
        num_active = int(aircraft_mask.sum())

        # 70% chance of no action
        if np.random.random() < 0.7:
            return {
                "aircraft_id": 0,  # No action
                "command_type": 0,
                "altitude": 0,
                "heading": 0,
                "speed": 0,
            }

        if num_active == 0:
            return {
                "aircraft_id": 0,
                "command_type": 0,
                "altitude": 0,
                "heading": 0,
                "speed": 0,
            }

        # Find conflicts if available
        if conflict_matrix is not None:
            # Look for aircraft in conflict
            conflicts = np.where(conflict_matrix.sum(axis=1) > 0)[0]

            if len(conflicts) > 0:
                # Select random conflicting aircraft
                aircraft_idx = np.random.choice(conflicts)

                # Issue altitude change (safer than heading)
                return {
                    "aircraft_id": aircraft_idx + 1,  # +1 because 0 is no-action
                    "command_type": 0,  # Altitude command
                    "altitude": np.random.randint(0, 18),  # Random altitude
                    "heading": 0,
                    "speed": 0,
                }

        # Otherwise, select random active aircraft
        active_indices = np.where(aircraft_mask)[0]
        if len(active_indices) > 0:
            aircraft_idx = np.random.choice(active_indices)

            # Random command type (prefer altitude and heading)
            command_type = np.random.choice([0, 1], p=[0.6, 0.4])  # Altitude or heading

            return {
                "aircraft_id": aircraft_idx + 1,
                "command_type": command_type,
                "altitude": np.random.randint(0, 18),
                "heading": np.random.randint(0, 13),
                "speed": np.random.randint(0, 8),
            }

        # Fallback to no action
        return {
            "aircraft_id": 0,
            "command_type": 0,
            "altitude": 0,
            "heading": 0,
            "speed": 0,
        }

    @staticmethod
    def save_episodes(episodes: List[Episode], filepath: str):
        """
        Save episodes to disk.

        Args:
            episodes: List of episodes
            filepath: Path to save file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "episodes": [ep.to_dict() for ep in episodes],
            "num_episodes": len(episodes),
            "total_steps": sum(ep.length for ep in episodes),
            "avg_return": np.mean([ep.total_return for ep in episodes]),
            "std_return": np.std([ep.total_return for ep in episodes]),
        }

        with open(filepath, "wb") as f:
            pickle.dump(data, f)

        logger.info(f"Saved {len(episodes)} episodes to {filepath}")
        logger.info(f"  Total steps: {data['total_steps']}")
        logger.info(f"  Avg return: {data['avg_return']:.2f} ± {data['std_return']:.2f}")

    @staticmethod
    def load_episodes(filepath: str) -> List[Episode]:
        """
        Load episodes from disk.

        Args:
            filepath: Path to saved file

        Returns:
            List of episodes
        """
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        episodes = [Episode.from_dict(ep_data) for ep_data in data["episodes"]]

        logger.info(f"Loaded {len(episodes)} episodes from {filepath}")
        return episodes


class OfflineRLDataset(Dataset):
    """
    PyTorch Dataset for offline RL with Decision Transformer.

    This dataset samples context windows from episodes for training.
    Each sample contains:
    - Context of past (return-to-go, state, action) tuples
    - Target actions to predict

    Example:
        >>> episodes = OfflineDatasetCollector.load_episodes("data.pkl")
        >>> dataset = OfflineRLDataset(
        ...     episodes=episodes,
        ...     context_len=20,
        ...     max_aircraft=20,
        ...     state_dim=14
        ... )
        >>> dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    """

    def __init__(
        self,
        episodes: List[Episode],
        context_len: int,
        max_aircraft: int,
        state_dim: int,
        scale_returns: bool = True,
        return_scale: float = 1000.0,
    ):
        """
        Initialize dataset.

        Args:
            episodes: List of episodes
            context_len: Context window length
            max_aircraft: Maximum number of aircraft
            state_dim: State feature dimension
            scale_returns: Whether to scale returns (recommended)
            return_scale: Scale factor for returns
        """
        self.episodes = episodes
        self.context_len = context_len
        self.max_aircraft = max_aircraft
        self.state_dim = state_dim
        self.scale_returns = scale_returns
        self.return_scale = return_scale

        # Pre-compute valid starting indices for each episode
        self.episode_starts = []
        for ep_idx, episode in enumerate(episodes):
            # We can sample from any position in the episode
            for start_idx in range(episode.length):
                self.episode_starts.append((ep_idx, start_idx))

        logger.info(f"Created dataset with {len(self.episode_starts)} samples from {len(episodes)} episodes")

    def __len__(self) -> int:
        return len(self.episode_starts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single training sample.

        Returns:
            Dictionary containing:
            - returns: (context_len, 1)
            - states: (context_len, max_aircraft, state_dim)
            - aircraft_masks: (context_len, max_aircraft)
            - actions: Dictionary with each component (context_len,)
            - timesteps: (context_len,)
            - attention_mask: (context_len,)
        """
        ep_idx, start_idx = self.episode_starts[idx]
        episode = self.episodes[ep_idx]

        # Determine actual context length (might be shorter than context_len)
        end_idx = min(start_idx + self.context_len, episode.length)
        actual_len = end_idx - start_idx

        # Padding length
        pad_len = self.context_len - actual_len

        # Extract returns-to-go
        returns = episode.returns_to_go[start_idx:end_idx]
        if self.scale_returns:
            returns = returns / self.return_scale

        # Pad returns
        if pad_len > 0:
            returns = np.concatenate([np.zeros(pad_len), returns])

        # Extract states
        states = []
        aircraft_masks = []
        for i in range(start_idx, end_idx):
            obs = episode.observations[i]
            states.append(obs["aircraft"])
            aircraft_masks.append(obs["aircraft_mask"])

        # Pad states and masks
        if pad_len > 0:
            pad_state = np.zeros((pad_len, self.max_aircraft, self.state_dim), dtype=np.float32)
            pad_mask = np.zeros((pad_len, self.max_aircraft), dtype=bool)
            states = [pad_state] + states
            aircraft_masks = [pad_mask] + aircraft_masks

        states = np.stack(states, axis=0)
        aircraft_masks = np.stack(aircraft_masks, axis=0)

        # Extract actions (as dictionary)
        actions_dict = defaultdict(list)
        for i in range(start_idx, end_idx):
            action = episode.actions[i]
            for key, value in action.items():
                actions_dict[key].append(value)

        # Pad actions
        if pad_len > 0:
            for key in actions_dict.keys():
                actions_dict[key] = [0] * pad_len + actions_dict[key]

        # Convert to numpy arrays
        actions_dict = {key: np.array(values, dtype=np.int64) for key, values in actions_dict.items()}

        # Create timesteps
        timesteps = np.arange(start_idx, end_idx)
        if pad_len > 0:
            timesteps = np.concatenate([np.zeros(pad_len, dtype=np.int64), timesteps])

        # Create attention mask (1 for valid, 0 for padding)
        attention_mask = np.concatenate([np.zeros(pad_len), np.ones(actual_len)])

        # Convert to tensors
        return {
            "returns": torch.from_numpy(returns).unsqueeze(-1).float(),  # (context_len, 1)
            "states": torch.from_numpy(states).float(),  # (context_len, max_aircraft, state_dim)
            "aircraft_masks": torch.from_numpy(aircraft_masks).bool(),  # (context_len, max_aircraft)
            "actions": {key: torch.from_numpy(val).long() for key, val in actions_dict.items()},
            "timesteps": torch.from_numpy(timesteps).long(),  # (context_len,)
            "attention_mask": torch.from_numpy(attention_mask).float(),  # (context_len,)
        }


def create_dataloader(
    episodes: List[Episode],
    context_len: int,
    max_aircraft: int,
    state_dim: int,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    **dataset_kwargs,
) -> DataLoader:
    """
    Create DataLoader for offline RL training.

    Args:
        episodes: List of episodes
        context_len: Context window length
        max_aircraft: Maximum aircraft
        state_dim: State dimension
        batch_size: Batch size
        shuffle: Shuffle data
        num_workers: Number of worker processes
        **dataset_kwargs: Additional dataset arguments

    Returns:
        DataLoader instance
    """
    dataset = OfflineRLDataset(
        episodes=episodes,
        context_len=context_len,
        max_aircraft=max_aircraft,
        state_dim=state_dim,
        **dataset_kwargs,
    )

    # Custom collate function for dictionary actions
    def collate_fn(batch):
        collated = {}

        # Handle each key separately
        for key in batch[0].keys():
            if key == "actions":
                # Actions is a dict, need special handling
                action_keys = batch[0]["actions"].keys()
                collated["actions"] = {
                    action_key: torch.stack([sample["actions"][action_key] for sample in batch])
                    for action_key in action_keys
                }
            else:
                # Stack normally
                collated[key] = torch.stack([sample[key] for sample in batch])

        return collated

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
