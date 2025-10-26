"""
Collect OpenScope gameplay data for Cosmos training.

This module collects episodes with synchronized video frames and game state data
for training NVIDIA Cosmos World Foundation Models on OpenScope ATC dynamics.
"""

import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
from PIL import Image
import cv2

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from environment import PlaywrightEnv, create_default_config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class CosmosDataCollector:
    """
    Collect OpenScope episodes with video + state data for Cosmos training.

    Captures:
    - Video frames (browser screenshots at 1 FPS)
    - Game states (aircraft positions, conflicts, etc.)
    - Actions taken (command issued)
    - Rewards received
    """

    def __init__(self, save_dir: str = "cosmos_data", frame_size: tuple = (512, 512)):
        """
        Initialize data collector.

        Args:
            save_dir: Directory to save collected data
            frame_size: Size to resize frames to (width, height)
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.frame_size = frame_size

        # Create subdirectories
        (self.save_dir / "videos").mkdir(exist_ok=True)
        (self.save_dir / "metadata").mkdir(exist_ok=True)
        (self.save_dir / "frames").mkdir(exist_ok=True)

        logger.info(f"Data collector initialized. Save directory: {self.save_dir}")

    def collect_episode(
        self,
        env: PlaywrightEnv,
        episode_id: int,
        policy: str = 'random',
        max_steps: int = 300
    ) -> Dict[str, Any]:
        """
        Collect single episode with video.

        Args:
            env: Initialized PlaywrightEnv
            episode_id: Episode identifier
            policy: Policy to use ('random' or 'heuristic')
            max_steps: Maximum steps per episode

        Returns:
            Episode data dictionary
        """
        frames = []
        states = []
        actions = []
        rewards = []
        observations = []

        logger.info(f"Starting episode {episode_id} with {policy} policy")

        obs, info = env.reset()
        done = False
        step = 0

        while not done and step < max_steps:
            # Capture screenshot from browser
            frame = self._capture_frame(env)
            frames.append(frame)

            # Store game state and observation
            states.append(info.get('raw_state', {}))
            observations.append(obs.copy())

            # Get action
            if policy == 'random':
                action = env.action_space.sample()
            elif policy == 'heuristic':
                action = self._heuristic_policy(obs, info)
            else:
                raise ValueError(f"Unknown policy: {policy}")

            actions.append(action)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            rewards.append(reward)

            done = terminated or truncated
            step += 1

            # Capture at 1 FPS (sleep to avoid capturing too fast)
            time.sleep(1.0)

        # Capture final frame
        if not done:
            frame = self._capture_frame(env)
            frames.append(frame)
            states.append(info.get('raw_state', {}))
            observations.append(obs.copy())

        # Save episode
        episode_data = {
            'frames': frames,
            'states': states,
            'observations': observations,
            'actions': actions,
            'rewards': rewards,
            'episode_return': sum(rewards),
            'episode_length': len(frames),
            'policy': policy,
        }

        # Save video
        video_path = self._save_video(frames, episode_id)
        episode_data['video_path'] = str(video_path)

        # Save metadata
        metadata_path = self.save_dir / "metadata" / f"episode_{episode_id}.npz"
        np.savez_compressed(
            metadata_path,
            observations=np.array(observations, dtype=object),
            actions=np.array(actions),
            rewards=np.array(rewards),
            states=np.array(states, dtype=object),
            episode_return=episode_data['episode_return'],
            episode_length=episode_data['episode_length'],
            policy=episode_data['policy'],
        )
        episode_data['metadata_path'] = str(metadata_path)

        logger.info(
            f"Episode {episode_id} complete: {len(frames)} frames, "
            f"return = {episode_data['episode_return']:.2f}"
        )

        return episode_data

    def _capture_frame(self, env: PlaywrightEnv) -> np.ndarray:
        """
        Capture screenshot from browser.

        Args:
            env: PlaywrightEnv instance

        Returns:
            Frame as numpy array (RGB)
        """
        try:
            # Get page screenshot as bytes
            screenshot_bytes = env.browser_manager.page.screenshot()

            # Convert to PIL Image
            from io import BytesIO
            image = Image.open(BytesIO(screenshot_bytes))

            # Resize to standard size
            image = image.resize(self.frame_size, Image.Resampling.LANCZOS)

            # Convert to numpy array
            frame = np.array(image)

            # Ensure RGB format
            if frame.shape[2] == 4:  # RGBA
                frame = frame[:, :, :3]

            return frame

        except Exception as e:
            logger.error(f"Failed to capture frame: {e}")
            # Return blank frame as fallback
            return np.zeros((*self.frame_size, 3), dtype=np.uint8)

    def _save_video(self, frames: List[np.ndarray], episode_id: int) -> Path:
        """
        Save frames as MP4 video.

        Args:
            frames: List of frames
            episode_id: Episode identifier

        Returns:
            Path to saved video
        """
        video_path = self.save_dir / "videos" / f"episode_{episode_id}.mp4"

        try:
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = 1  # 1 FPS (we capture at 1 frame per second)
            height, width = frames[0].shape[:2]

            writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

            for frame in frames:
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                writer.write(frame_bgr)

            writer.release()
            logger.debug(f"Video saved to {video_path}")

        except Exception as e:
            logger.error(f"Failed to save video: {e}")

        return video_path

    def _heuristic_policy(self, obs: np.ndarray, info: Dict[str, Any]) -> np.ndarray:
        """
        Simple rule-based policy for better data quality.

        Args:
            obs: Current observation
            info: Info dict from environment

        Returns:
            Action to take
        """
        # TODO: Implement actual heuristics based on observation
        # For now, use random policy as placeholder
        #
        # Potential heuristics:
        # - If conflict predicted, issue heading change
        # - If near exit, continue straight
        # - If near runway, issue ILS approach
        # - Maintain safe separation distances

        # Placeholder: sample random action
        from environment import PlaywrightEnv
        # This is a simplified version - in practice would need access to action space
        return np.array([0, 0, 0, 0])  # No-op action

    def collect_dataset(
        self,
        num_episodes: int = 100,
        airport: str = "KSFO",
        headless: bool = False,
        mix_policies: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Collect full dataset of episodes.

        Args:
            num_episodes: Number of episodes to collect
            airport: Airport code to use
            headless: Whether to run browser in headless mode
            mix_policies: Whether to mix random and heuristic policies

        Returns:
            List of episode data dictionaries
        """
        logger.info(f"Starting dataset collection: {num_episodes} episodes")
        logger.info(f"Airport: {airport}, Headless: {headless}")

        # Create environment
        config = create_default_config(
            airport=airport,
            headless=headless,
            timewarp=1  # Real-time for better video quality
        )
        env = PlaywrightEnv(**config.__dict__)

        episodes = []

        try:
            for i in range(num_episodes):
                # Mix of policies for diversity
                if mix_policies:
                    policy = 'random' if i % 2 == 0 else 'heuristic'
                else:
                    policy = 'random'

                try:
                    episode = self.collect_episode(env, episode_id=i, policy=policy)
                    episodes.append(episode)

                    # Log progress
                    if (i + 1) % 10 == 0:
                        avg_return = np.mean([ep['episode_return'] for ep in episodes[-10:]])
                        avg_length = np.mean([ep['episode_length'] for ep in episodes[-10:]])
                        logger.info(
                            f"Progress: {i+1}/{num_episodes} episodes | "
                            f"Last 10 avg return: {avg_return:.2f} | "
                            f"Last 10 avg length: {avg_length:.0f} frames"
                        )

                except Exception as e:
                    logger.error(f"Failed to collect episode {i}: {e}")
                    continue

        finally:
            env.close()

        # Save summary
        self._save_summary(episodes)

        logger.info(f"\nDataset collection complete!")
        logger.info(f"  Total episodes: {len(episodes)}")
        logger.info(f"  Total frames: {sum(ep['episode_length'] for ep in episodes)}")
        logger.info(f"  Average frames/episode: {np.mean([ep['episode_length'] for ep in episodes]):.0f}")
        logger.info(f"  Average return: {np.mean([ep['episode_return'] for ep in episodes]):.2f}")
        logger.info(f"  Saved to: {self.save_dir}")

        return episodes

    def _save_summary(self, episodes: List[Dict[str, Any]]) -> None:
        """
        Save dataset summary.

        Args:
            episodes: List of episode data
        """
        summary = {
            'num_episodes': len(episodes),
            'total_frames': sum(ep['episode_length'] for ep in episodes),
            'avg_episode_length': np.mean([ep['episode_length'] for ep in episodes]),
            'avg_episode_return': np.mean([ep['episode_return'] for ep in episodes]),
            'std_episode_return': np.std([ep['episode_return'] for ep in episodes]),
            'min_episode_return': np.min([ep['episode_return'] for ep in episodes]),
            'max_episode_return': np.max([ep['episode_return'] for ep in episodes]),
        }

        summary_path = self.save_dir / "dataset_summary.npz"
        np.savez(summary_path, **summary)
        logger.info(f"Summary saved to {summary_path}")


def main():
    """Main entry point for data collection."""
    import argparse

    parser = argparse.ArgumentParser(description="Collect OpenScope data for Cosmos training")
    parser.add_argument("--num-episodes", type=int, default=100, help="Number of episodes to collect")
    parser.add_argument("--airport", type=str, default="KSFO", help="Airport code")
    parser.add_argument("--headless", action="store_true", help="Run browser in headless mode")
    parser.add_argument("--save-dir", type=str, default="cosmos_data", help="Save directory")
    parser.add_argument("--no-mix", action="store_true", help="Use only random policy")

    args = parser.parse_args()

    collector = CosmosDataCollector(save_dir=args.save_dir)
    collector.collect_dataset(
        num_episodes=args.num_episodes,
        airport=args.airport,
        headless=args.headless,
        mix_policies=not args.no_mix
    )


if __name__ == "__main__":
    main()
