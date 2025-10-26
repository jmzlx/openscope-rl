"""
Train PPO in Cosmos-simulated environment.

This module implements PPO training in the fast Cosmos-based environment,
then transfers the learned policy to the real OpenScope environment.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
    CheckpointCallback,
)
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from environment.cosmos_env import CosmosOpenScopeEnv
from environment import PlaywrightEnv, create_default_config


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class TransferCallback(BaseCallback):
    """
    Callback for periodically evaluating policy transfer to real OpenScope.
    """

    def __init__(
        self,
        eval_freq: int = 10000,
        n_eval_episodes: int = 5,
        real_env_config: Optional[Dict[str, Any]] = None,
        verbose: int = 1,
    ):
        """
        Initialize transfer callback.

        Args:
            eval_freq: Evaluate transfer every N steps
            n_eval_episodes: Number of episodes for evaluation
            real_env_config: Configuration for real environment
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.real_env_config = real_env_config or {}
        self.transfer_rewards = []

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            logger.info(f"\nEvaluating policy transfer at step {self.n_calls}...")

            # Create real environment
            config = create_default_config(**self.real_env_config)
            real_env = PlaywrightEnv(**config.__dict__)

            try:
                episode_rewards = []

                for episode in range(self.n_eval_episodes):
                    obs, info = real_env.reset()
                    done = False
                    episode_reward = 0.0

                    while not done:
                        # Use trained policy
                        action, _states = self.model.predict(obs, deterministic=True)
                        obs, reward, terminated, truncated, info = real_env.step(action)
                        episode_reward += reward
                        done = terminated or truncated

                    episode_rewards.append(episode_reward)
                    logger.info(f"  Transfer episode {episode+1}: reward = {episode_reward:.2f}")

                avg_reward = np.mean(episode_rewards)
                self.transfer_rewards.append((self.n_calls, avg_reward))

                logger.info(f"Transfer evaluation complete: avg reward = {avg_reward:.2f}")

            finally:
                real_env.close()

        return True


class CosmosRLTrainer:
    """Train PPO in Cosmos environment and transfer to real OpenScope."""

    def __init__(
        self,
        cosmos_model_path: str = "cosmos-openscope-finetuned",
        output_dir: str = "cosmos_rl_models",
        n_envs: int = 4,
    ):
        """
        Initialize trainer.

        Args:
            cosmos_model_path: Path to fine-tuned Cosmos model
            output_dir: Directory to save trained models
            n_envs: Number of parallel environments
        """
        self.cosmos_model_path = cosmos_model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.n_envs = n_envs

        logger.info(f"Cosmos RL Trainer initialized")
        logger.info(f"  Cosmos model: {cosmos_model_path}")
        logger.info(f"  Output dir: {output_dir}")
        logger.info(f"  Parallel envs: {n_envs}")

    def make_cosmos_env(self, rank: int = 0):
        """
        Create Cosmos environment.

        Args:
            rank: Environment rank for seeding

        Returns:
            Wrapped environment
        """
        def _init():
            env = CosmosOpenScopeEnv(cosmos_model_path=self.cosmos_model_path)
            env = Monitor(env)
            env.reset(seed=rank)
            return env

        return _init

    def train(
        self,
        total_timesteps: int = 10_000_000,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        eval_freq: int = 50000,
        save_freq: int = 100000,
        transfer_eval_freq: int = 100000,
    ) -> PPO:
        """
        Train PPO in Cosmos environment.

        Args:
            total_timesteps: Total training timesteps
            learning_rate: Learning rate
            n_steps: Steps per rollout
            batch_size: Minibatch size
            n_epochs: Epochs per update
            eval_freq: Evaluation frequency
            save_freq: Checkpoint save frequency
            transfer_eval_freq: Transfer evaluation frequency

        Returns:
            Trained PPO model
        """
        logger.info("Setting up training...")

        # Create vectorized environments
        env = DummyVecEnv([self.make_cosmos_env(i) for i in range(self.n_envs)])

        # Normalize observations and rewards
        env = VecNormalize(env, norm_obs=True, norm_reward=True)

        # Create evaluation environment
        eval_env = DummyVecEnv([self.make_cosmos_env(self.n_envs)])
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)

        # Create PPO model
        logger.info("Creating PPO model...")
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            tensorboard_log=str(self.output_dir / "tensorboard"),
        )

        # Setup callbacks
        callbacks = []

        # Evaluation callback
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(self.output_dir / "best_model"),
            log_path=str(self.output_dir / "eval_logs"),
            eval_freq=eval_freq,
            deterministic=True,
            render=False,
        )
        callbacks.append(eval_callback)

        # Checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=save_freq,
            save_path=str(self.output_dir / "checkpoints"),
            name_prefix="cosmos_ppo",
        )
        callbacks.append(checkpoint_callback)

        # Transfer evaluation callback
        transfer_callback = TransferCallback(
            eval_freq=transfer_eval_freq,
            n_eval_episodes=3,
            real_env_config={'headless': True, 'timewarp': 10},
        )
        callbacks.append(transfer_callback)

        # Train
        logger.info(f"Starting training for {total_timesteps:,} timesteps...")
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True,
        )

        # Save final model
        final_model_path = self.output_dir / "final_model"
        model.save(final_model_path)
        env.save(self.output_dir / "vec_normalize.pkl")

        logger.info(f"Training complete! Model saved to {final_model_path}")

        # Save transfer evaluation results
        if transfer_callback.transfer_rewards:
            transfer_results = np.array(transfer_callback.transfer_rewards)
            np.save(self.output_dir / "transfer_rewards.npy", transfer_results)
            logger.info(f"Transfer rewards saved to {self.output_dir / 'transfer_rewards.npy'}")

        return model

    def evaluate_transfer(
        self,
        model_path: str,
        n_episodes: int = 10,
        render: bool = False,
    ) -> Dict[str, float]:
        """
        Evaluate trained policy on real OpenScope environment.

        Args:
            model_path: Path to trained model
            n_episodes: Number of evaluation episodes
            render: Whether to render episodes

        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating policy transfer on real OpenScope...")
        logger.info(f"  Model: {model_path}")
        logger.info(f"  Episodes: {n_episodes}")

        # Load model
        model = PPO.load(model_path)

        # Create real environment
        config = create_default_config(
            headless=not render,
            timewarp=1 if render else 10,
        )
        env = PlaywrightEnv(**config.__dict__)

        episode_rewards = []
        episode_lengths = []

        try:
            for episode in range(n_episodes):
                obs, info = env.reset()
                done = False
                episode_reward = 0.0
                episode_length = 0

                while not done:
                    # Use trained policy
                    action, _states = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)

                    episode_reward += reward
                    episode_length += 1
                    done = terminated or truncated

                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)

                logger.info(
                    f"Episode {episode+1}/{n_episodes}: "
                    f"reward = {episode_reward:.2f}, "
                    f"length = {episode_length}"
                )

        finally:
            env.close()

        # Compute statistics
        results = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'mean_length': np.mean(episode_lengths),
        }

        logger.info("\nTransfer Evaluation Results:")
        for key, value in results.items():
            logger.info(f"  {key}: {value:.2f}")

        return results

    def compare_sample_efficiency(
        self,
        cosmos_model_path: str,
        baseline_model_path: Optional[str] = None,
        n_episodes: int = 10,
    ) -> Dict[str, Any]:
        """
        Compare sample efficiency of Cosmos training vs baseline.

        Args:
            cosmos_model_path: Path to Cosmos-trained model
            baseline_model_path: Path to baseline model (trained on real env)
            n_episodes: Number of evaluation episodes

        Returns:
            Comparison results
        """
        logger.info("Comparing sample efficiency...")

        # Evaluate Cosmos-trained policy
        cosmos_results = self.evaluate_transfer(cosmos_model_path, n_episodes)

        # Evaluate baseline if provided
        if baseline_model_path:
            baseline_results = self.evaluate_transfer(baseline_model_path, n_episodes)

            comparison = {
                'cosmos': cosmos_results,
                'baseline': baseline_results,
                'improvement': {
                    'mean_reward': (
                        (cosmos_results['mean_reward'] - baseline_results['mean_reward'])
                        / baseline_results['mean_reward'] * 100
                    ),
                },
            }
        else:
            comparison = {
                'cosmos': cosmos_results,
                'baseline': None,
                'improvement': None,
            }

        return comparison


def main():
    """Main entry point for training."""
    import argparse

    parser = argparse.ArgumentParser(description="Train PPO in Cosmos environment")
    parser.add_argument(
        "--cosmos-model",
        type=str,
        default="cosmos-openscope-finetuned",
        help="Path to fine-tuned Cosmos model",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="cosmos_rl_models",
        help="Output directory",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=10_000_000,
        help="Total training timesteps",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=4,
        help="Number of parallel environments",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only evaluate existing model",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to model for evaluation",
    )

    args = parser.parse_args()

    trainer = CosmosRLTrainer(
        cosmos_model_path=args.cosmos_model,
        output_dir=args.output_dir,
        n_envs=args.n_envs,
    )

    if args.eval_only:
        if not args.model_path:
            raise ValueError("--model-path required for evaluation")
        trainer.evaluate_transfer(args.model_path, n_episodes=10, render=False)
    else:
        trainer.train(total_timesteps=args.timesteps)


if __name__ == "__main__":
    main()
