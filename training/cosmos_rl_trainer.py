"""
RL Training in Cosmos-Simulated Environment.

This module trains RL policies (e.g., PPO) in the fast Cosmos-simulated
OpenScope environment. Training is much faster than in the real environment
because there's no browser overhead.

After training in Cosmos, the policy can be transferred to the real OpenScope
environment for evaluation and fine-tuning.
"""

import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

from environment.cosmos_env import CosmosOpenScopeEnv

logger = logging.getLogger(__name__)


@dataclass
class CosmosRLConfig:
    """Configuration for RL training in Cosmos environment."""

    # Environment settings
    cosmos_model_path: str
    reward_model_path: Optional[str] = None
    state_extractor_path: Optional[str] = None
    airport: str = "KLAS"
    max_aircraft: int = 10
    episode_length: int = 1800

    # Training settings
    total_timesteps: int = 10_000_000  # 10M steps
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Environment parallelization
    n_envs: int = 8  # Number of parallel environments
    use_subproc: bool = True  # Use SubprocVecEnv for better performance

    # Network architecture
    policy_type: str = "MultiInputPolicy"
    net_arch: Dict[str, list] = None

    # Checkpointing
    checkpoint_freq: int = 100_000  # Save every 100k steps
    eval_freq: int = 50_000  # Evaluate every 50k steps
    n_eval_episodes: int = 10

    # Output
    output_dir: str = "cosmos_rl_trained"
    tensorboard_log: str = "cosmos_rl_logs"

    def __post_init__(self):
        if self.net_arch is None:
            self.net_arch = dict(pi=[256, 256], vf=[256, 256])


class WandbCallback(BaseCallback):
    """
    Callback for logging to Weights & Biases.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        # Log episode rewards
        if len(self.model.ep_info_buffer) > 0:
            for ep_info in self.model.ep_info_buffer:
                self.episode_rewards.append(ep_info["r"])
                self.episode_lengths.append(ep_info["l"])

        # Log to tensorboard every 100 steps
        if self.n_calls % 100 == 0:
            if len(self.episode_rewards) > 0:
                self.logger.record("rollout/mean_ep_reward", np.mean(self.episode_rewards[-100:]))
                self.logger.record("rollout/mean_ep_length", np.mean(self.episode_lengths[-100:]))

        return True


class CosmosRLTrainer:
    """
    Train RL policies in Cosmos-simulated OpenScope environment.

    This trainer:
    1. Creates vectorized Cosmos environments for fast training
    2. Trains PPO (or other RL algorithms) on the simulated environment
    3. Saves checkpoints and evaluation results
    4. Provides utilities for transferring to real OpenScope

    Example:
        >>> config = CosmosRLConfig(
        ...     cosmos_model_path="cosmos_finetuned/best_model.pt",
        ...     total_timesteps=10_000_000,
        ...     n_envs=8
        ... )
        >>> trainer = CosmosRLTrainer(config)
        >>> trainer.train()
        >>> trainer.save("cosmos_trained_policy")
    """

    def __init__(self, config: CosmosRLConfig):
        """
        Initialize Cosmos RL trainer.

        Args:
            config: Training configuration
        """
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Initializing CosmosRLTrainer")
        logger.info(f"Output directory: {self.output_dir}")

        # Create environments
        self.env = self._create_env()
        self.eval_env = self._create_eval_env()

        # Create model
        self.model = self._create_model()

        logger.info("CosmosRLTrainer initialized")
        logger.info(f"Training for {config.total_timesteps:,} timesteps")
        logger.info(f"Using {config.n_envs} parallel environments")

    def _create_env(self):
        """Create training environment."""

        def make_env():
            """Create a single environment."""
            env = CosmosOpenScopeEnv(
                cosmos_model_path=self.config.cosmos_model_path,
                reward_model_path=self.config.reward_model_path,
                state_extractor_path=self.config.state_extractor_path,
                airport=self.config.airport,
                max_aircraft=self.config.max_aircraft,
                episode_length=self.config.episode_length,
            )
            env = Monitor(env)
            return env

        # Create vectorized environment
        if self.config.use_subproc and self.config.n_envs > 1:
            env = SubprocVecEnv([make_env for _ in range(self.config.n_envs)])
            logger.info(f"Created SubprocVecEnv with {self.config.n_envs} environments")
        else:
            env = DummyVecEnv([make_env for _ in range(self.config.n_envs)])
            logger.info(f"Created DummyVecEnv with {self.config.n_envs} environments")

        return env

    def _create_eval_env(self):
        """Create evaluation environment."""

        def make_eval_env():
            env = CosmosOpenScopeEnv(
                cosmos_model_path=self.config.cosmos_model_path,
                reward_model_path=self.config.reward_model_path,
                state_extractor_path=self.config.state_extractor_path,
                airport=self.config.airport,
                max_aircraft=self.config.max_aircraft,
                episode_length=self.config.episode_length,
            )
            env = Monitor(env)
            return env

        # Single environment for evaluation
        eval_env = DummyVecEnv([make_eval_env])
        logger.info("Created evaluation environment")

        return eval_env

    def _create_model(self):
        """Create PPO model."""
        # Configure logger
        logger_config = configure(str(self.output_dir / self.config.tensorboard_log), ["tensorboard", "stdout"])

        # Create model
        model = PPO(
            policy=self.config.policy_type,
            env=self.env,
            learning_rate=self.config.learning_rate,
            n_steps=self.config.n_steps,
            batch_size=self.config.batch_size,
            n_epochs=self.config.n_epochs,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            clip_range=self.config.clip_range,
            ent_coef=self.config.ent_coef,
            vf_coef=self.config.vf_coef,
            max_grad_norm=self.config.max_grad_norm,
            verbose=1,
            tensorboard_log=str(self.output_dir / self.config.tensorboard_log),
            policy_kwargs=dict(net_arch=self.config.net_arch),
        )

        model.set_logger(logger_config)

        logger.info("Created PPO model")
        logger.info(f"Policy: {self.config.policy_type}")
        logger.info(f"Network architecture: {self.config.net_arch}")

        return model

    def train(self):
        """
        Run the training loop.
        """
        logger.info("Starting training")
        start_time = time.time()

        # Create callbacks
        callbacks = self._create_callbacks()

        # Train
        self.model.learn(
            total_timesteps=self.config.total_timesteps,
            callback=callbacks,
            log_interval=1,
            tb_log_name="PPO",
            reset_num_timesteps=True,
        )

        # Training complete
        elapsed_time = time.time() - start_time
        logger.info(f"Training completed in {elapsed_time:.2f} seconds")
        logger.info(f"Training speed: {self.config.total_timesteps / elapsed_time:.2f} steps/sec")

        # Save final model
        self.save("final_model")

    def _create_callbacks(self):
        """Create training callbacks."""
        callbacks = []

        # Checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=self.config.checkpoint_freq // self.config.n_envs,  # Adjust for vectorized env
            save_path=str(self.output_dir / "checkpoints"),
            name_prefix="cosmos_ppo",
            save_replay_buffer=False,
            save_vecnormalize=True,
        )
        callbacks.append(checkpoint_callback)

        # Evaluation callback
        eval_callback = EvalCallback(
            self.eval_env,
            best_model_save_path=str(self.output_dir / "best_model"),
            log_path=str(self.output_dir / "eval_logs"),
            eval_freq=self.config.eval_freq // self.config.n_envs,  # Adjust for vectorized env
            n_eval_episodes=self.config.n_eval_episodes,
            deterministic=True,
            render=False,
        )
        callbacks.append(eval_callback)

        # WandB callback (optional)
        try:
            import wandb

            if wandb.run is not None:
                wandb_callback = WandbCallback()
                callbacks.append(wandb_callback)
                logger.info("WandB logging enabled")
        except ImportError:
            logger.debug("WandB not available")

        return callbacks

    def save(self, name: str = "model"):
        """
        Save trained model.

        Args:
            name: Model name
        """
        save_path = self.output_dir / f"{name}.zip"
        self.model.save(save_path)
        logger.info(f"Model saved to: {save_path}")

        # Also save config
        config_path = self.output_dir / f"{name}_config.json"
        import json

        with open(config_path, "w") as f:
            json.dump(asdict(self.config), f, indent=2)
        logger.info(f"Config saved to: {config_path}")

    def load(self, path: str):
        """
        Load trained model.

        Args:
            path: Path to model file
        """
        self.model = PPO.load(path, env=self.env)
        logger.info(f"Model loaded from: {path}")

    def evaluate(self, n_episodes: int = 10) -> Dict[str, float]:
        """
        Evaluate trained policy.

        Args:
            n_episodes: Number of episodes to evaluate

        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"Evaluating policy for {n_episodes} episodes")

        episode_rewards = []
        episode_lengths = []

        for i in range(n_episodes):
            obs = self.eval_env.reset()
            done = False
            episode_reward = 0.0
            episode_length = 0

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.eval_env.step(action)

                episode_reward += reward[0]
                episode_length += 1

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

            logger.debug(f"Episode {i + 1}: reward={episode_reward:.2f}, length={episode_length}")

        # Compute statistics
        results = {
            "mean_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "mean_length": np.mean(episode_lengths),
            "std_length": np.std(episode_lengths),
        }

        logger.info(f"Evaluation results: mean_reward={results['mean_reward']:.2f} "
                   f"± {results['std_reward']:.2f}")

        return results

    def transfer_to_real_openscope(
        self,
        real_env_config: Optional[Dict[str, Any]] = None,
        n_finetune_steps: int = 100_000,
    ):
        """
        Transfer policy to real OpenScope environment and optionally fine-tune.

        Args:
            real_env_config: Configuration for real OpenScope environment
            n_finetune_steps: Number of fine-tuning steps (0 = no fine-tuning)
        """
        from environment import PlaywrightEnv, create_default_config

        logger.info("Transferring policy to real OpenScope environment")

        # Create real environment
        if real_env_config is None:
            real_env_config = {
                "airport": self.config.airport,
                "max_aircraft": self.config.max_aircraft,
                "episode_length": self.config.episode_length,
                "headless": True,
            }

        config = create_default_config(**real_env_config)
        real_env = PlaywrightEnv(**config.__dict__)
        real_env = Monitor(real_env)
        real_vec_env = DummyVecEnv([lambda: real_env])

        # Set environment for model
        self.model.set_env(real_vec_env)

        logger.info("Policy transferred to real OpenScope environment")

        # Fine-tune if requested
        if n_finetune_steps > 0:
            logger.info(f"Fine-tuning for {n_finetune_steps} steps in real environment")

            # Create fine-tuning callbacks
            finetune_callback = CheckpointCallback(
                save_freq=10_000,
                save_path=str(self.output_dir / "finetune_checkpoints"),
                name_prefix="finetune_ppo",
            )

            # Fine-tune
            self.model.learn(
                total_timesteps=n_finetune_steps,
                callback=finetune_callback,
                log_interval=10,
                tb_log_name="PPO_finetune",
                reset_num_timesteps=False,
            )

            logger.info("Fine-tuning completed")

            # Save fine-tuned model
            self.save("finetuned_model")

        # Clean up
        real_env.close()

    def close(self):
        """Clean up resources."""
        if hasattr(self, "env"):
            self.env.close()
        if hasattr(self, "eval_env"):
            self.eval_env.close()
        logger.info("CosmosRLTrainer closed")


def main():
    """Example training script."""
    import argparse

    parser = argparse.ArgumentParser(description="Train RL policy in Cosmos environment")
    parser.add_argument("--cosmos-model", type=str, required=True, help="Path to fine-tuned Cosmos model")
    parser.add_argument("--reward-model", type=str, default=None, help="Path to reward model")
    parser.add_argument("--output-dir", type=str, default="cosmos_rl_trained", help="Output directory")
    parser.add_argument("--timesteps", type=int, default=10_000_000, help="Total training timesteps")
    parser.add_argument("--n-envs", type=int, default=8, help="Number of parallel environments")
    parser.add_argument("--airport", type=str, default="KLAS", help="Airport code")
    parser.add_argument("--max-aircraft", type=int, default=10, help="Maximum aircraft")

    args = parser.parse_args()

    # Create config
    config = CosmosRLConfig(
        cosmos_model_path=args.cosmos_model,
        reward_model_path=args.reward_model,
        output_dir=args.output_dir,
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        airport=args.airport,
        max_aircraft=args.max_aircraft,
    )

    # Create trainer
    trainer = CosmosRLTrainer(config)

    try:
        # Train
        trainer.train()

        # Evaluate
        results = trainer.evaluate(n_episodes=20)
        logger.info(f"Final evaluation: {results}")

    finally:
        trainer.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    main()
