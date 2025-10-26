"""
Cosmos Data Collector for OpenScope.

This module collects OpenScope gameplay data for training NVIDIA Cosmos world models.
It captures video frames, game states, actions, and rewards during episodes.
"""

import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import base64

import cv2
import numpy as np
from tqdm import tqdm

from environment import PlaywrightEnv, create_default_config

logger = logging.getLogger(__name__)


@dataclass
class FrameData:
    """Data for a single frame in an episode."""

    frame_idx: int
    timestamp: float
    simulated_time: float

    # Visual data
    screenshot: np.ndarray  # RGB image (H, W, 3)

    # Game state
    state: Dict[str, Any]  # Raw game state
    aircraft_count: int
    conflict_count: int
    score: int

    # Action taken
    action: Optional[Dict[str, int]]
    action_text: Optional[str]  # Human-readable action description

    # Reward
    reward: float


@dataclass
class EpisodeData:
    """Complete episode data for Cosmos training."""

    episode_id: str
    airport: str
    max_aircraft: int
    timewarp: int

    # Episode metadata
    start_time: float
    end_time: float
    duration: float

    # Episode statistics
    total_frames: int
    total_reward: float
    avg_aircraft: float
    max_aircraft_seen: int
    total_conflicts: int
    total_violations: int

    # Frame-by-frame data
    frames: List[FrameData]

    # Episode outcome
    terminated: bool
    truncated: bool
    final_score: int


class CosmosDataCollector:
    """
    Collect OpenScope gameplay data for Cosmos world model training.

    This collector captures:
    - Video frames (browser screenshots at each step)
    - Game states (aircraft positions, velocities, etc.)
    - Actions taken (command type, parameters)
    - Rewards received

    The collected data can be used to fine-tune NVIDIA Cosmos models to
    simulate OpenScope dynamics.

    Example:
        >>> collector = CosmosDataCollector(
        ...     output_dir="cosmos_data",
        ...     airport="KJFK",
        ...     max_aircraft=10
        ... )
        >>> collector.collect_dataset(num_episodes=100)
    """

    def __init__(
        self,
        output_dir: str = "cosmos_data",
        airport: str = "KLAS",
        max_aircraft: int = 10,
        timewarp: int = 10,
        headless: bool = False,  # Keep visible for debugging
        episode_length: int = 1800,  # 30 minutes
        frame_skip: int = 1,  # Capture every frame
        video_fps: int = 10,  # Output video FPS
        screenshot_width: int = 1280,
        screenshot_height: int = 720,
    ):
        """
        Initialize Cosmos data collector.

        Args:
            output_dir: Directory to save collected data
            airport: ICAO airport code
            max_aircraft: Maximum number of aircraft
            timewarp: Time acceleration factor
            headless: Whether to run browser in headless mode
            episode_length: Maximum episode length in seconds
            frame_skip: Capture every Nth frame (1 = all frames)
            video_fps: Output video frames per second
            screenshot_width: Screenshot width in pixels
            screenshot_height: Screenshot height in pixels
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.airport = airport
        self.max_aircraft = max_aircraft
        self.timewarp = timewarp
        self.headless = headless
        self.episode_length = episode_length
        self.frame_skip = frame_skip
        self.video_fps = video_fps
        self.screenshot_size = (screenshot_width, screenshot_height)

        logger.info(f"CosmosDataCollector initialized: {airport}, {max_aircraft} aircraft")
        logger.info(f"Output directory: {self.output_dir}")

    def collect_episode(
        self,
        episode_id: str,
        policy: str = "random",
        save_video: bool = True,
        save_frames: bool = False,
    ) -> EpisodeData:
        """
        Collect a single episode of OpenScope gameplay.

        Args:
            episode_id: Unique identifier for this episode
            policy: Policy to use ("random", "heuristic", "manual")
            save_video: Whether to save video file
            save_frames: Whether to save individual frame images

        Returns:
            EpisodeData: Complete episode data
        """
        logger.info(f"Starting episode collection: {episode_id}")
        start_time = time.time()

        # Create episode directory
        episode_dir = self.output_dir / episode_id
        episode_dir.mkdir(parents=True, exist_ok=True)

        # Create environment
        config = create_default_config(
            airport=self.airport,
            max_aircraft=self.max_aircraft,
            timewarp=self.timewarp,
            headless=self.headless,
            episode_length=self.episode_length,
        )

        env = PlaywrightEnv(**config.__dict__)

        try:
            # Reset environment
            obs, info = env.reset()

            # Initialize data collection
            frames: List[FrameData] = []
            total_reward = 0.0
            aircraft_counts: List[int] = []
            conflict_counts: List[int] = []
            violation_count = 0

            # Video writer (if saving video)
            video_writer = None
            if save_video:
                video_path = episode_dir / f"{episode_id}.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(
                    str(video_path),
                    fourcc,
                    self.video_fps,
                    self.screenshot_size
                )

            # Collect frames
            frame_idx = 0
            done = False

            with tqdm(desc=f"Collecting {episode_id}", unit="frame") as pbar:
                while not done:
                    # Capture screenshot
                    screenshot = self._capture_screenshot(env)

                    # Get current state
                    state = info.get("raw_state", {})
                    aircraft_count = len(state.get("aircraft", []))
                    conflict_count = len(state.get("conflicts", []))
                    score = state.get("score", 0)

                    aircraft_counts.append(aircraft_count)
                    conflict_counts.append(conflict_count)

                    # Count violations
                    for conflict in state.get("conflicts", []):
                        if conflict.get("hasViolation", False):
                            violation_count += 1

                    # Select action based on policy
                    action = self._select_action(env, obs, policy)
                    action_text = self._action_to_text(action, state)

                    # Store frame data (before taking action)
                    if frame_idx % self.frame_skip == 0:
                        frame_data = FrameData(
                            frame_idx=frame_idx,
                            timestamp=time.time() - start_time,
                            simulated_time=info.get("simulated_time", 0.0),
                            screenshot=screenshot,
                            state=state,
                            aircraft_count=aircraft_count,
                            conflict_count=conflict_count,
                            score=score,
                            action=action,
                            action_text=action_text,
                            reward=0.0,  # Will be updated after step
                        )
                        frames.append(frame_data)

                        # Save video frame
                        if video_writer is not None:
                            # Convert RGB to BGR for OpenCV
                            bgr_frame = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
                            video_writer.write(bgr_frame)

                        # Save individual frame (optional)
                        if save_frames:
                            frame_path = episode_dir / "frames" / f"frame_{frame_idx:06d}.png"
                            frame_path.parent.mkdir(exist_ok=True)
                            cv2.imwrite(str(frame_path), cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR))

                    # Execute action
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated

                    # Update reward for this frame
                    if frames and frame_idx % self.frame_skip == 0:
                        frames[-1].reward = reward

                    total_reward += reward
                    frame_idx += 1
                    pbar.update(1)

                    # Safety limit
                    if frame_idx >= 10000:  # Prevent runaway episodes
                        logger.warning(f"Episode {episode_id} reached frame limit")
                        break

            # Close video writer
            if video_writer is not None:
                video_writer.release()

            # Create episode data
            end_time = time.time()
            episode_data = EpisodeData(
                episode_id=episode_id,
                airport=self.airport,
                max_aircraft=self.max_aircraft,
                timewarp=self.timewarp,
                start_time=start_time,
                end_time=end_time,
                duration=end_time - start_time,
                total_frames=len(frames),
                total_reward=total_reward,
                avg_aircraft=np.mean(aircraft_counts) if aircraft_counts else 0.0,
                max_aircraft_seen=max(aircraft_counts) if aircraft_counts else 0,
                total_conflicts=sum(conflict_counts),
                total_violations=violation_count,
                frames=frames,
                terminated=terminated,
                truncated=truncated,
                final_score=info.get("score", 0),
            )

            # Save episode metadata (without frames to save memory)
            metadata = {
                "episode_id": episode_data.episode_id,
                "airport": episode_data.airport,
                "max_aircraft": episode_data.max_aircraft,
                "timewarp": episode_data.timewarp,
                "start_time": episode_data.start_time,
                "end_time": episode_data.end_time,
                "duration": episode_data.duration,
                "total_frames": episode_data.total_frames,
                "total_reward": episode_data.total_reward,
                "avg_aircraft": episode_data.avg_aircraft,
                "max_aircraft_seen": episode_data.max_aircraft_seen,
                "total_conflicts": episode_data.total_conflicts,
                "total_violations": episode_data.total_violations,
                "terminated": episode_data.terminated,
                "truncated": episode_data.truncated,
                "final_score": episode_data.final_score,
            }

            metadata_path = episode_dir / "metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            # Save frame actions and rewards (without screenshots)
            actions_rewards = []
            for frame in frames:
                actions_rewards.append({
                    "frame_idx": frame.frame_idx,
                    "timestamp": frame.timestamp,
                    "simulated_time": frame.simulated_time,
                    "action": frame.action,
                    "action_text": frame.action_text,
                    "reward": frame.reward,
                    "aircraft_count": frame.aircraft_count,
                    "conflict_count": frame.conflict_count,
                    "score": frame.score,
                })

            actions_path = episode_dir / "actions_rewards.json"
            with open(actions_path, "w") as f:
                json.dump(actions_rewards, f, indent=2)

            logger.info(f"Episode {episode_id} collected: {len(frames)} frames, "
                       f"reward={total_reward:.2f}, duration={end_time - start_time:.2f}s")

            return episode_data

        finally:
            env.close()

    def collect_dataset(
        self,
        num_episodes: int = 100,
        traffic_levels: Optional[List[int]] = None,
        policies: Optional[List[str]] = None,
        save_video: bool = True,
        save_frames: bool = False,
    ) -> List[EpisodeData]:
        """
        Collect a dataset of multiple episodes with diverse scenarios.

        Args:
            num_episodes: Number of episodes to collect
            traffic_levels: List of aircraft counts to try (None = use default)
            policies: List of policies to use (None = random only)
            save_video: Whether to save video files
            save_frames: Whether to save individual frame images

        Returns:
            List of EpisodeData for all collected episodes
        """
        if traffic_levels is None:
            traffic_levels = [2, 5, 10]  # Start with lower traffic

        if policies is None:
            policies = ["random"]

        logger.info(f"Starting dataset collection: {num_episodes} episodes")
        logger.info(f"Traffic levels: {traffic_levels}")
        logger.info(f"Policies: {policies}")

        episodes: List[EpisodeData] = []

        # Distribute episodes across traffic levels and policies
        episodes_per_config = num_episodes // (len(traffic_levels) * len(policies))
        extra_episodes = num_episodes % (len(traffic_levels) * len(policies))

        episode_idx = 0

        for traffic in traffic_levels:
            for policy in policies:
                # Update max aircraft for this configuration
                self.max_aircraft = traffic

                # Collect episodes for this configuration
                num_for_config = episodes_per_config + (1 if episode_idx < extra_episodes else 0)

                for i in range(num_for_config):
                    episode_id = f"ep_{episode_idx:04d}_traffic{traffic}_{policy}"

                    try:
                        episode_data = self.collect_episode(
                            episode_id=episode_id,
                            policy=policy,
                            save_video=save_video,
                            save_frames=save_frames,
                        )
                        episodes.append(episode_data)

                    except Exception as e:
                        logger.error(f"Failed to collect episode {episode_id}: {e}")

                    episode_idx += 1

        # Save dataset summary
        summary = {
            "num_episodes": len(episodes),
            "total_frames": sum(ep.total_frames for ep in episodes),
            "total_duration": sum(ep.duration for ep in episodes),
            "avg_reward": np.mean([ep.total_reward for ep in episodes]),
            "traffic_levels": traffic_levels,
            "policies": policies,
        }

        summary_path = self.output_dir / "dataset_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Dataset collection completed: {len(episodes)} episodes, "
                   f"{summary['total_frames']} frames, "
                   f"{summary['total_duration']:.2f}s total duration")

        return episodes

    def _capture_screenshot(self, env: PlaywrightEnv) -> np.ndarray:
        """
        Capture a screenshot from the environment.

        Args:
            env: OpenScope environment

        Returns:
            Screenshot as RGB numpy array (H, W, 3)
        """
        try:
            # Get page from browser manager
            page = env.browser_manager.page

            # Capture screenshot as PNG bytes
            screenshot_bytes = page.screenshot(
                type="png",
                full_page=False,
            )

            # Convert to numpy array
            nparr = np.frombuffer(screenshot_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Resize to target size
            if img_rgb.shape[:2] != self.screenshot_size[::-1]:  # (H, W) vs (W, H)
                img_rgb = cv2.resize(
                    img_rgb,
                    self.screenshot_size,
                    interpolation=cv2.INTER_AREA
                )

            return img_rgb

        except Exception as e:
            logger.error(f"Failed to capture screenshot: {e}")
            # Return black frame as fallback
            return np.zeros((self.screenshot_size[1], self.screenshot_size[0], 3), dtype=np.uint8)

    def _select_action(
        self,
        env: PlaywrightEnv,
        obs: Dict[str, np.ndarray],
        policy: str = "random"
    ) -> Dict[str, int]:
        """
        Select an action based on the specified policy.

        Args:
            env: OpenScope environment
            obs: Current observation
            policy: Policy name ("random", "heuristic", "manual")

        Returns:
            Action dictionary
        """
        if policy == "random":
            return env.action_space.sample()

        elif policy == "heuristic":
            # TODO: Implement simple heuristic policy
            # For now, use random
            return env.action_space.sample()

        elif policy == "manual":
            # TODO: Implement manual control
            # For now, use random
            return env.action_space.sample()

        else:
            logger.warning(f"Unknown policy: {policy}, using random")
            return env.action_space.sample()

    def _action_to_text(
        self,
        action: Dict[str, int],
        state: Dict[str, Any]
    ) -> str:
        """
        Convert action dictionary to human-readable text.

        Args:
            action: Action dictionary
            state: Current game state

        Returns:
            Human-readable action description
        """
        try:
            from environment.config import CommandType

            # Get action components
            aircraft_id = action.get("aircraft_id", 0)
            command_type_idx = action.get("command_type", 0)

            # Get command type
            command_types = [CommandType.ALTITUDE, CommandType.HEADING, CommandType.SPEED,
                           CommandType.ILS, CommandType.DIRECT]
            command_type = command_types[command_type_idx] if command_type_idx < len(command_types) else CommandType.ALTITUDE

            # Get aircraft callsign
            aircraft_list = state.get("aircraft", [])
            callsign = aircraft_list[aircraft_id]["callsign"] if aircraft_id < len(aircraft_list) else f"A{aircraft_id}"

            # Format action text
            if command_type == CommandType.ALTITUDE:
                altitude = action.get("altitude_value", 0) * 100  # Convert to feet
                return f"{callsign} climb/descend to {altitude} feet"

            elif command_type == CommandType.HEADING:
                heading_change = action.get("heading_change", 0)
                return f"{callsign} turn {heading_change} degrees"

            elif command_type == CommandType.SPEED:
                speed = action.get("speed_value", 250)
                return f"{callsign} speed {speed} knots"

            elif command_type == CommandType.ILS:
                return f"{callsign} cleared ILS approach"

            elif command_type == CommandType.DIRECT:
                return f"{callsign} proceed direct"

            else:
                return f"{callsign} {command_type.value}"

        except Exception as e:
            logger.debug(f"Failed to convert action to text: {e}")
            return "Unknown action"


class CosmosDataset:
    """
    Dataset wrapper for loading and managing Cosmos training data.

    This class provides utilities for loading collected episodes,
    creating train/val splits, and iterating over the data.
    """

    def __init__(self, data_dir: str):
        """
        Initialize dataset from directory.

        Args:
            data_dir: Directory containing collected episodes
        """
        self.data_dir = Path(data_dir)
        self.episodes: List[Dict[str, Any]] = []

        # Load dataset summary
        summary_path = self.data_dir / "dataset_summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                self.summary = json.load(f)
        else:
            self.summary = {}

        # Load all episode metadata
        self._load_episodes()

        logger.info(f"Loaded dataset: {len(self.episodes)} episodes, "
                   f"{self.summary.get('total_frames', 0)} frames")

    def _load_episodes(self):
        """Load metadata for all episodes."""
        episode_dirs = [d for d in self.data_dir.iterdir() if d.is_dir() and d.name.startswith("ep_")]

        for episode_dir in sorted(episode_dirs):
            metadata_path = episode_dir / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path) as f:
                    metadata = json.load(f)
                    metadata["episode_dir"] = str(episode_dir)
                    self.episodes.append(metadata)

    def get_episode(self, episode_id: str) -> Dict[str, Any]:
        """Get episode metadata by ID."""
        for episode in self.episodes:
            if episode["episode_id"] == episode_id:
                return episode
        raise ValueError(f"Episode not found: {episode_id}")

    def load_video(self, episode_id: str) -> Optional[np.ndarray]:
        """
        Load video for an episode.

        Args:
            episode_id: Episode identifier

        Returns:
            Video as numpy array (T, H, W, 3) or None if not found
        """
        episode = self.get_episode(episode_id)
        episode_dir = Path(episode["episode_dir"])
        video_path = episode_dir / f"{episode_id}.mp4"

        if not video_path.exists():
            logger.warning(f"Video not found: {video_path}")
            return None

        # Load video using OpenCV
        cap = cv2.VideoCapture(str(video_path))
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

        cap.release()

        return np.array(frames) if frames else None

    def load_actions_rewards(self, episode_id: str) -> List[Dict[str, Any]]:
        """
        Load actions and rewards for an episode.

        Args:
            episode_id: Episode identifier

        Returns:
            List of frame data with actions and rewards
        """
        episode = self.get_episode(episode_id)
        episode_dir = Path(episode["episode_dir"])
        actions_path = episode_dir / "actions_rewards.json"

        if not actions_path.exists():
            logger.warning(f"Actions/rewards not found: {actions_path}")
            return []

        with open(actions_path) as f:
            return json.load(f)

    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        if not self.episodes:
            return {}

        return {
            "num_episodes": len(self.episodes),
            "total_frames": sum(ep["total_frames"] for ep in self.episodes),
            "avg_frames_per_episode": np.mean([ep["total_frames"] for ep in self.episodes]),
            "avg_reward": np.mean([ep["total_reward"] for ep in self.episodes]),
            "avg_aircraft": np.mean([ep["avg_aircraft"] for ep in self.episodes]),
            "total_conflicts": sum(ep["total_conflicts"] for ep in self.episodes),
            "total_violations": sum(ep["total_violations"] for ep in self.episodes),
        }

    def create_splits(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Create train/val/test splits.

        Args:
            train_ratio: Fraction for training set
            val_ratio: Fraction for validation set
            test_ratio: Fraction for test set
            seed: Random seed

        Returns:
            Tuple of (train_ids, val_ids, test_ids)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Split ratios must sum to 1.0"

        episode_ids = [ep["episode_id"] for ep in self.episodes]

        # Shuffle with seed
        rng = np.random.RandomState(seed)
        rng.shuffle(episode_ids)

        # Split
        n_train = int(len(episode_ids) * train_ratio)
        n_val = int(len(episode_ids) * val_ratio)

        train_ids = episode_ids[:n_train]
        val_ids = episode_ids[n_train:n_train + n_val]
        test_ids = episode_ids[n_train + n_val:]

        logger.info(f"Created splits: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")

        return train_ids, val_ids, test_ids
