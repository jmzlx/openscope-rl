"""
Training script using Stable-Baselines3
Much simpler than custom implementation - SB3 handles the training loop!
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import wandb
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from environment.wrappers import make_openscope_env
from models.sb3_policy import ATCTransformerPolicy
from utils.curriculum_callback import CurriculumCallback


def parse_args():
    parser = argparse.ArgumentParser(description="Train OpenScope RL Agent with SB3")
    parser.add_argument(
        "--config", type=str, default="config/training_config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Path to checkpoint to resume from"
    )
    parser.add_argument("--device", type=str, default="auto", help="Device to use (auto/cuda/cpu)")
    parser.add_argument(
        "--n-envs",
        type=int,
        default=1,
        help="Number of parallel environments (1=no vectorization, 4-8=recommended)",
    )
    parser.add_argument("--wandb", action="store_true", help="Use Weights & Biases logging")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load training configuration"""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def make_env(config: dict, rank: int):
    """
    Create a single environment (for vectorization)

    Args:
        config: Configuration dictionary
        rank: Environment index (for seeding)

    Returns:
        Callable that creates the environment
    """

    def _init():
        env_config = config["env"]
        env = make_openscope_env(
            game_url=env_config["game_url"],
            airport=env_config["airport"],
            timewarp=env_config["timewarp"],
            max_aircraft=env_config["max_aircraft"],
            episode_length=env_config["episode_length"],
            action_interval=env_config["action_interval"],
            headless=env_config["headless"],
            config=config,
            normalize_obs=True,
            normalize_reward=True,
            gamma=config["ppo"]["gamma"],
            record_stats=True,
        )
        # Monitor for logging
        env = Monitor(env)
        return env

    return _init


def main():
    args = parse_args()
    config = load_config(args.config)

    # Set random seeds
    seed = config["training"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create directories
    checkpoint_dir = Path(config["training"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    log_dir = Path(config["training"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)

    # Initialize wandb if requested
    if args.wandb and config["training"].get("use_wandb", False):
        wandb.init(
            project=config["training"]["wandb_project"],
            config=config,
            sync_tensorboard=True,  # Auto-sync SB3's tensorboard logs
            monitor_gym=True,
            save_code=True,
        )

    print(f"Training with {args.n_envs} parallel environment(s)")
    print(f"Device: {args.device}")

    # Create vectorized environment
    if args.n_envs > 1:
        print(f"Creating {args.n_envs} parallel environments (SubprocVecEnv)...")
        env = SubprocVecEnv([make_env(config, i) for i in range(args.n_envs)])
    else:
        print("Creating single environment (no vectorization)...")
        env = DummyVecEnv([make_env(config, 0)])

    # Create evaluation environment (single, non-vectorized)
    eval_env = DummyVecEnv([make_env(config, 999)])  # Different seed for eval

    # Create or load model
    network_config = config["network"]
    ppo_config = config["ppo"]

    policy_kwargs = dict(
        aircraft_feature_dim=network_config["aircraft_feature_dim"],
        global_feature_dim=network_config["global_feature_dim"],
        hidden_dim=network_config["hidden_dim"],
        num_heads=network_config["num_attention_heads"],
        num_layers=network_config["num_transformer_layers"],
        max_aircraft=network_config["max_aircraft_slots"],
    )

    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        model = PPO.load(
            args.checkpoint,
            env=env,
            device=args.device,
            custom_objects={"policy_class": ATCTransformerPolicy},
        )
    else:
        print("Creating new model...")
        model = PPO(
            ATCTransformerPolicy,
            env,
            learning_rate=ppo_config["learning_rate"],
            n_steps=ppo_config["n_steps"] // args.n_envs,  # Divide by n_envs for same total steps
            batch_size=ppo_config["batch_size"],
            n_epochs=ppo_config["n_epochs"],
            gamma=ppo_config["gamma"],
            gae_lambda=ppo_config["gae_lambda"],
            clip_range=ppo_config["clip_epsilon"],
            clip_range_vf=ppo_config.get("clip_range_vf", None),
            ent_coef=ppo_config["entropy_coef"],
            vf_coef=ppo_config["value_coef"],
            max_grad_norm=ppo_config["max_grad_norm"],
            use_sde=False,
            sde_sample_freq=-1,
            target_kl=None,
            tensorboard_log=str(log_dir),
            policy_kwargs=policy_kwargs,
            verbose=1,
            seed=seed,
            device=args.device,
        )

    # Print model info
    print("\nModel Architecture:")
    print(f"Total parameters: {sum(p.numel() for p in model.policy.parameters()):,}")

    # Setup callbacks
    callbacks = []

    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=config["training"]["save_interval"] // args.n_envs,
        save_path=str(checkpoint_dir),
        name_prefix="checkpoint",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )
    callbacks.append(checkpoint_callback)

    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(checkpoint_dir / "best_model"),
        log_path=str(log_dir),
        eval_freq=config["training"]["eval_interval"] // args.n_envs,
        n_eval_episodes=config["training"]["eval_episodes"],
        deterministic=True,
        render=False,
    )
    callbacks.append(eval_callback)

    # Curriculum callback
    if config["curriculum"].get("enabled", False):
        curriculum_callback = CurriculumCallback(config["curriculum"])
        callbacks.append(curriculum_callback)

    # Wandb callback (if enabled)
    if args.wandb and config["training"].get("use_wandb", False):
        from wandb.integration.sb3 import WandbCallback

        wandb_callback = WandbCallback(
            model_save_path=f"{log_dir}/wandb_models",
            gradient_save_freq=100,
            verbose=2,
        )
        callbacks.append(wandb_callback)

    # Combine all callbacks
    callback = CallbackList(callbacks)

    # Train!
    print(f"\nStarting training for {config['training']['total_timesteps']:,} timesteps...")
    print("=" * 60)

    try:
        model.learn(
            total_timesteps=config["training"]["total_timesteps"],
            callback=callback,
            log_interval=1,
            tb_log_name="ppo_openscope",
            reset_num_timesteps=True if not args.checkpoint else False,
            progress_bar=True,
        )

        # Save final model
        final_path = checkpoint_dir / "checkpoint_final.zip"
        model.save(final_path)
        print(f"\nFinal model saved to: {final_path}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        final_path = checkpoint_dir / "checkpoint_interrupted.zip"
        model.save(final_path)
        print(f"Model saved to: {final_path}")

    finally:
        env.close()
        eval_env.close()
        if args.wandb and config["training"].get("use_wandb", False):
            wandb.finish()

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
