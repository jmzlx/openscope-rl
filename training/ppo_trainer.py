"""
PPO Trainer with Action Masking for OpenScope RL.

This script implements the baseline PPO approach with:
- Action masking for improved sample efficiency
- Vectorized parallel environments
- Curriculum learning (progressive difficulty)
- WandB logging and experiment tracking
- Checkpoint saving with VecNormalize stats
"""

import argparse
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, Callable

import gymnasium as gym
import numpy as np
import torch
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from sb3_contrib.common.wrappers import ActionMasker

import sys
sys.path.append(str(Path(__file__).parent.parent))

from environment import PlaywrightEnv, create_default_config
from environment.action_masking import create_action_mask_fn
# Note: We use stable-baselines3's built-in policy, not custom models
# from models import ATCActorCritic, NetworkConfig


class WandbCallback:
    """Custom callback for WandB logging."""

    def __init__(self, log_freq: int = 1000):
        self.log_freq = log_freq
        self.n_calls = 0

    def __call__(self, locals_dict: Dict[str, Any], globals_dict: Dict[str, Any]) -> bool:
        self.n_calls += 1

        if self.n_calls % self.log_freq == 0:
            # Log training metrics
            if "infos" in locals_dict:
                infos = locals_dict["infos"]
                for info in infos:
                    if "episode" in info:
                        wandb.log({
                            "train/episode_reward": info["episode"]["r"],
                            "train/episode_length": info["episode"]["l"],
                        })

        return True


def make_env(
    rank: int,
    seed: int,
    config: Dict[str, Any],
    use_action_masking: bool = True,
) -> Callable:
    """
    Create a single environment instance.

    Args:
        rank: Unique ID for this environment
        seed: Random seed
        config: Environment configuration
        use_action_masking: Whether to use action masking wrapper

    Returns:
        Function that creates the environment
    """
    def _init() -> gym.Env:
        env = PlaywrightEnv(**config)

        # Apply action masking wrapper
        if use_action_masking:
            mask_fn = create_action_mask_fn(env)
            env = ActionMasker(env, mask_fn)

        # Monitor wrapper for episode statistics
        env = Monitor(env)

        # Set seed
        env.reset(seed=seed + rank)

        return env

    return _init


def create_vec_env(
    n_envs: int,
    config: Dict[str, Any],
    seed: int = 0,
    use_action_masking: bool = True,
    normalize: bool = True,
) -> VecNormalize:
    """
    Create vectorized parallel environments.

    Args:
        n_envs: Number of parallel environments
        config: Environment configuration
        seed: Random seed
        use_action_masking: Whether to use action masking
        normalize: Whether to use VecNormalize

    Returns:
        Vectorized environment with normalization
    """
    # Create environment factories
    env_fns = [make_env(i, seed, config, use_action_masking) for i in range(n_envs)]

    # Create vectorized environment
    if n_envs > 1:
        vec_env = SubprocVecEnv(env_fns)
    else:
        vec_env = DummyVecEnv(env_fns)

    # Apply VecNormalize for observation/reward normalization
    if normalize:
        vec_env = VecNormalize(
            vec_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=0.99,
        )

    return vec_env


def create_ppo_model(
    env: gym.Env,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    device: str = "auto",
    policy_kwargs: Optional[Dict[str, Any]] = None,
) -> PPO:
    """
    Create PPO model with optimized hyperparameters for OpenScope.

    Hyperparameters are tuned for:
    - Sample efficiency with action masking
    - Stable learning in complex ATC environment
    - Good exploration-exploitation balance

    Args:
        env: Vectorized environment
        learning_rate: Learning rate
        n_steps: Number of steps per environment per update
        batch_size: Minibatch size
        n_epochs: Number of epochs per update
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        clip_range: PPO clip range
        ent_coef: Entropy coefficient
        vf_coef: Value function coefficient
        max_grad_norm: Max gradient norm
        device: Device to use (cpu/cuda/auto)
        policy_kwargs: Additional policy kwargs

    Returns:
        PPO model
    """
    if policy_kwargs is None:
        policy_kwargs = {}

    # Create PPO model
    model = PPO(
        "MultiInputPolicy",  # For Dict observation space
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        verbose=1,
        device=device,
        policy_kwargs=policy_kwargs,
    )

    return model


class CurriculumLearningCallback:
    """
    Callback for curriculum learning - progressively increase difficulty.

    Starts with few aircraft and gradually increases to full capacity.
    """

    def __init__(
        self,
        env: VecNormalize,
        stages: list = None,
    ):
        """
        Initialize curriculum learning callback.

        Args:
            env: Vectorized environment
            stages: List of (timestep, max_aircraft) tuples
        """
        self.env = env
        self.stages = stages or [
            (0, 2),        # Start: 2 aircraft
            (50000, 4),    # 50k steps: 4 aircraft
            (100000, 6),   # 100k steps: 6 aircraft
            (200000, 10),  # 200k steps: 10 aircraft
        ]
        self.current_stage = 0
        self.n_calls = 0

    def __call__(self, locals_dict: Dict[str, Any], globals_dict: Dict[str, Any]) -> bool:
        self.n_calls += 1

        # Check if we should advance to next stage
        if self.current_stage < len(self.stages) - 1:
            next_stage_timestep = self.stages[self.current_stage + 1][0]
            if self.n_calls >= next_stage_timestep:
                self.current_stage += 1
                max_aircraft = self.stages[self.current_stage][1]

                print(f"\n{'='*80}")
                print(f"CURRICULUM STAGE {self.current_stage + 1}: max_aircraft = {max_aircraft}")
                print(f"{'='*80}\n")

                # Note: Actually changing max_aircraft requires recreating environments
                # For now, we just log the transition
                # In practice, you'd need to recreate the vec_env with new config

                wandb.log({
                    "curriculum/stage": self.current_stage,
                    "curriculum/max_aircraft": max_aircraft,
                })

        return True


def train(
    total_timesteps: int = 500000,
    n_envs: int = 8,
    max_aircraft: int = 10,
    airport: str = "KLAS",
    save_dir: str = "./checkpoints",
    wandb_project: str = "openscope-rl-baseline",
    wandb_entity: str = "jmzlx.ai",
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    use_action_masking: bool = True,
    use_curriculum: bool = True,
    eval_freq: int = 10000,
    checkpoint_freq: int = 25000,
    seed: int = 0,
    device: str = "auto",
) -> PPO:
    """
    Train PPO agent with action masking.

    Args:
        total_timesteps: Total training timesteps
        n_envs: Number of parallel environments
        max_aircraft: Maximum number of aircraft
        airport: Airport ICAO code
        save_dir: Directory to save checkpoints
        wandb_project: WandB project name
        wandb_entity: WandB entity name
        learning_rate: Learning rate
        n_steps: Steps per environment per update
        batch_size: Minibatch size
        use_action_masking: Whether to use action masking
        use_curriculum: Whether to use curriculum learning
        eval_freq: Evaluation frequency
        checkpoint_freq: Checkpoint save frequency
        seed: Random seed
        device: Device (cpu/cuda/auto)

    Returns:
        Trained PPO model
    """
    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Initialize WandB
    run = wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        config={
            "algorithm": "PPO",
            "total_timesteps": total_timesteps,
            "n_envs": n_envs,
            "max_aircraft": max_aircraft,
            "airport": airport,
            "learning_rate": learning_rate,
            "n_steps": n_steps,
            "batch_size": batch_size,
            "use_action_masking": use_action_masking,
            "use_curriculum": use_curriculum,
            "seed": seed,
        },
        sync_tensorboard=True,
    )

    # Create environment config
    env_config = {
        "airport": airport,
        "max_aircraft": max_aircraft,
        "headless": True,
        "timewarp": 5,
        "episode_length": 3600,
        "action_interval": 5.0,
    }

    print(f"\n{'='*80}")
    print("BASELINE PPO TRAINING")
    print(f"{'='*80}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Parallel envs: {n_envs}")
    print(f"Max aircraft: {max_aircraft}")
    print(f"Airport: {airport}")
    print(f"Action masking: {use_action_masking}")
    print(f"Curriculum learning: {use_curriculum}")
    print(f"Device: {device}")
    print(f"{'='*80}\n")

    # Create vectorized environments
    print("Creating training environments...")
    env = create_vec_env(
        n_envs=n_envs,
        config=env_config,
        seed=seed,
        use_action_masking=use_action_masking,
        normalize=True,
    )

    print("Creating evaluation environment...")
    eval_env = create_vec_env(
        n_envs=1,
        config=env_config,
        seed=seed + 1000,
        use_action_masking=use_action_masking,
        normalize=True,
    )

    # Create PPO model
    print("Creating PPO model...")
    model = create_ppo_model(
        env=env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        device=device,
    )

    # Create callbacks
    callbacks = []

    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=str(save_path),
        name_prefix="ppo_openscope",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )
    callbacks.append(checkpoint_callback)

    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(save_path / "best_model"),
        log_path=str(save_path / "eval_logs"),
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
    )
    callbacks.append(eval_callback)

    # Curriculum learning callback (if enabled)
    if use_curriculum:
        curriculum_callback = CurriculumLearningCallback(env)
        # Note: curriculum callback needs to be integrated differently
        # For now, we start with fixed difficulty

    callback = CallbackList(callbacks)

    # Train the model
    print("\nStarting training...")
    start_time = time.time()

    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True,
    )

    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.1f}s ({training_time/60:.1f}min)")

    # Save final model
    final_model_path = save_path / "final_model"
    model.save(str(final_model_path))
    env.save(str(save_path / "vec_normalize.pkl"))

    print(f"\nFinal model saved to: {final_model_path}")
    print(f"VecNormalize stats saved to: {save_path / 'vec_normalize.pkl'}")

    # Close environments
    env.close()
    eval_env.close()

    # Finish WandB run
    wandb.finish()

    return model


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description="Train PPO agent for OpenScope")

    # Training parameters
    parser.add_argument("--total-timesteps", type=int, default=500000,
                       help="Total training timesteps")
    parser.add_argument("--n-envs", type=int, default=8,
                       help="Number of parallel environments")
    parser.add_argument("--seed", type=int, default=0,
                       help="Random seed")

    # Environment parameters
    parser.add_argument("--max-aircraft", type=int, default=10,
                       help="Maximum number of aircraft")
    parser.add_argument("--airport", type=str, default="KLAS",
                       help="Airport ICAO code")

    # Model parameters
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                       help="Learning rate")
    parser.add_argument("--n-steps", type=int, default=2048,
                       help="Steps per environment per update")
    parser.add_argument("--batch-size", type=int, default=64,
                       help="Minibatch size")

    # Features
    parser.add_argument("--no-action-masking", action="store_true",
                       help="Disable action masking")
    parser.add_argument("--use-curriculum", action="store_true",
                       help="Use curriculum learning")

    # Logging and checkpoints
    parser.add_argument("--save-dir", type=str, default="./checkpoints",
                       help="Directory to save checkpoints")
    parser.add_argument("--wandb-project", type=str, default="openscope-rl-baseline",
                       help="WandB project name")
    parser.add_argument("--wandb-entity", type=str, default="jmzlx.ai",
                       help="WandB entity name")
    parser.add_argument("--eval-freq", type=int, default=10000,
                       help="Evaluation frequency")
    parser.add_argument("--checkpoint-freq", type=int, default=25000,
                       help="Checkpoint save frequency")

    # Device
    parser.add_argument("--device", type=str, default="auto",
                       help="Device (cpu/cuda/auto)")

    args = parser.parse_args()

    # Train model
    model = train(
        total_timesteps=args.total_timesteps,
        n_envs=args.n_envs,
        max_aircraft=args.max_aircraft,
        airport=args.airport,
        save_dir=args.save_dir,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        use_action_masking=not args.no_action_masking,
        use_curriculum=args.use_curriculum,
        eval_freq=args.eval_freq,
        checkpoint_freq=args.checkpoint_freq,
        seed=args.seed,
        device=args.device,
    )

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
