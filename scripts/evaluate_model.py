#!/usr/bin/env python3
"""
Model evaluation script for OpenScope RL.

This script loads a trained model and evaluates it on the environment,
saving results to JSON.
"""

import argparse
import json
import sys
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from environment import PlaywrightEnv, create_default_config, DictToMultiDiscreteWrapper
from environment.action_masking import create_action_mask_fn
from sb3_contrib.common.wrappers import ActionMasker


def evaluate_model(
    model_path: str,
    n_episodes: int = 10,
    airport: str = "KLAS",
    max_aircraft: int = 10,
    headless: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate a trained model.
    
    Args:
        model_path: Path to saved model
        n_episodes: Number of evaluation episodes
        airport: Airport ICAO code
        max_aircraft: Maximum aircraft
        headless: Run in headless mode
        
    Returns:
        Dictionary with evaluation results
    """
    print(f"Loading model from: {model_path}")
    model = PPO.load(model_path)
    
    print(f"Creating evaluation environment: {airport}, max_aircraft={max_aircraft}")
    
    # Create environment
    config = create_default_config(
        airport=airport,
        max_aircraft=max_aircraft,
        headless=headless,
    )
    
    def make_env():
        env = PlaywrightEnv(
            airport=config.airport,
            max_aircraft=config.max_aircraft,
            headless=config.browser_config.headless,
            timewarp=config.timewarp,
        )
        env = DictToMultiDiscreteWrapper(env)
        mask_fn = create_action_mask_fn(env)
        env = ActionMasker(env, mask_fn)
        env = Monitor(env)
        return env
    
    eval_env = DummyVecEnv([make_env])
    
    print(f"\nEvaluating model for {n_episodes} episodes...")
    
    episode_rewards: List[float] = []
    episode_lengths: List[int] = []
    episode_stats: List[Dict[str, Any]] = []
    
    for episode in range(n_episodes):
        obs = eval_env.reset()
        episode_reward = 0.0
        episode_length = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            
            # Handle vectorized environments
            if isinstance(done, (list, tuple, np.ndarray)):
                done = done[0] if len(done) > 0 else False
            if isinstance(reward, (list, tuple, np.ndarray)):
                reward = reward[0] if len(reward) > 0 else 0.0
            
            episode_reward += float(reward)
            episode_length += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Extract stats from info
        stats = {}
        if isinstance(info, (list, tuple)) and len(info) > 0:
            info_dict = info[0] if isinstance(info[0], dict) else {}
            stats = {
                'successful_exits': info_dict.get('successful_exits', 0),
                'violations': info_dict.get('violations', 0),
                'total_aircraft': info_dict.get('total_aircraft_spawned', 0),
            }
        
        episode_stats.append(stats)
        
        print(f"Episode {episode + 1}/{n_episodes}: "
              f"Reward={episode_reward:.2f}, "
              f"Length={episode_length}, "
              f"Exits={stats.get('successful_exits', 0)}")
    
    # Calculate summary statistics
    results = {
        'model_path': model_path,
        'n_episodes': n_episodes,
        'airport': airport,
        'max_aircraft': max_aircraft,
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'episode_stats': episode_stats,
        'summary': {
            'mean_reward': sum(episode_rewards) / len(episode_rewards),
            'std_reward': (sum((r - sum(episode_rewards)/len(episode_rewards))**2 
                              for r in episode_rewards) / len(episode_rewards))**0.5,
            'min_reward': min(episode_rewards),
            'max_reward': max(episode_rewards),
            'mean_length': sum(episode_lengths) / len(episode_lengths),
            'total_successful_exits': sum(s.get('successful_exits', 0) for s in episode_stats),
            'total_violations': sum(s.get('violations', 0) for s in episode_stats),
        }
    }
    
    eval_env.close()
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained OpenScope RL model")
    parser.add_argument("model_path", type=str, help="Path to saved model")
    parser.add_argument("--n-episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--airport", type=str, default="KLAS", help="Airport ICAO code")
    parser.add_argument("--max-aircraft", type=int, default=10, help="Max aircraft")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    parser.add_argument("--output", type=str, help="Output JSON file path")
    
    args = parser.parse_args()
    
    # Evaluate model
    results = evaluate_model(
        model_path=args.model_path,
        n_episodes=args.n_episodes,
        airport=args.airport,
        max_aircraft=args.max_aircraft,
        headless=args.headless,
    )
    
    # Print summary
    print("\n" + "=" * 80)
    print("Evaluation Summary")
    print("=" * 80)
    summary = results['summary']
    print(f"Mean Reward: {summary['mean_reward']:.2f} ± {summary['std_reward']:.2f}")
    print(f"Reward Range: [{summary['min_reward']:.2f}, {summary['max_reward']:.2f}]")
    print(f"Mean Episode Length: {summary['mean_length']:.1f}")
    print(f"Total Successful Exits: {summary['total_successful_exits']}")
    print(f"Total Violations: {summary['total_violations']}")
    print("=" * 80)
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")
    else:
        # Default output path
        output_path = Path(args.model_path).parent / "evaluation_results.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()

