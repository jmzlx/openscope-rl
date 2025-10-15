"""
Evaluation script for trained SB3 models
"""

import argparse

import numpy as np
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from environment.wrappers import make_openscope_env
from models.sb3_policy import ATCTransformerPolicy


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained OpenScope RL Agent")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to trained model (.zip file)"
    )
    parser.add_argument(
        "--config", type=str, default="config/training_config.yaml", help="Path to config file"
    )
    parser.add_argument("--n-episodes", type=int, default=20, help="Number of evaluation episodes")
    parser.add_argument(
        "--render", action="store_true", help="Render the environment (non-headless)"
    )
    parser.add_argument(
        "--deterministic", action="store_true", help="Use deterministic actions (no sampling)"
    )
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load training configuration"""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def evaluate(model: PPO, env, n_episodes: int, deterministic: bool = True):
    """
    Evaluate the model

    Args:
        model: Trained PPO model
        env: Environment
        n_episodes: Number of episodes to run
        deterministic: Use deterministic actions

    Returns:
        dict: Evaluation statistics
    """
    episode_rewards = []
    episode_lengths = []
    episode_scores = []

    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, info = env.step(action)

            episode_reward += reward[0]  # Unwrap from vec env
            episode_length += 1

            if done[0]:
                # Extract final score from info
                if "episode" in info[0]:
                    episode_scores.append(info[0].get("score", 0))
                break

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        print(
            f"Episode {episode + 1}/{n_episodes}: "
            f"Reward={episode_reward:.2f}, "
            f"Length={episode_length}, "
            f"Score={episode_scores[-1] if episode_scores else 'N/A'}"
        )

    return {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "mean_score": np.mean(episode_scores) if episode_scores else 0,
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "episode_scores": episode_scores,
    }


def main():
    args = parse_args()
    config = load_config(args.config)

    print(f"Loading model from: {args.checkpoint}")

    # Create environment
    env_config = config["env"]
    if args.render:
        env_config["headless"] = False

    def make_eval_env():
        env = make_openscope_env(
            **env_config,
            config=config,
            normalize_obs=True,
            normalize_reward=False,  # Don't normalize rewards during eval
            record_stats=True,
        )
        return env

    env = DummyVecEnv([make_eval_env])

    # Load model
    model = PPO.load(
        args.checkpoint,
        env=env,
        device=args.device,
        custom_objects={"policy_class": ATCTransformerPolicy},
    )

    print(f"\nEvaluating for {args.n_episodes} episodes...")
    print("=" * 60)

    # Evaluate
    stats = evaluate(model, env, args.n_episodes, args.deterministic)

    # Print results
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Mean Reward: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
    print(f"Mean Length: {stats['mean_length']:.1f}")
    print(f"Mean Score: {stats['mean_score']:.1f}")
    print()
    print(f"Episode Rewards: {[f'{r:.1f}' for r in stats['episode_rewards']]}")
    print("=" * 60)

    # Close
    env.close()


if __name__ == "__main__":
    main()
