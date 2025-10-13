"""
Evaluation script for trained OpenScope RL agents
"""

import torch
import numpy as np
import argparse
import yaml
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

from environment.openscope_env import OpenScopeEnv
from models.networks import ATCActorCritic
from algorithms.ppo import PPO


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate OpenScope RL Agent")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained checkpoint")
    parser.add_argument("--config", type=str, default="config/training_config.yaml",
                        help="Path to config file")
    parser.add_argument("--n-episodes", type=int, default=20,
                        help="Number of evaluation episodes")
    parser.add_argument("--render", action="store_true",
                        help="Render the environment (non-headless)")
    parser.add_argument("--output", type=str, default="evaluation_results.json",
                        help="Output file for results")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    return parser.parse_args()


def evaluate_agent(
    env: OpenScopeEnv,
    ppo: PPO,
    n_episodes: int,
    render: bool = False
) -> dict:
    """
    Comprehensive evaluation of the agent
    """
    ppo.policy.eval()
    
    results = {
        'episodes': [],
        'statistics': {}
    }
    
    for episode_idx in tqdm(range(n_episodes), desc="Evaluating"):
        obs, info = env.reset()
        done = False
        
        episode_data = {
            'episode': episode_idx,
            'total_reward': 0,
            'length': 0,
            'final_score': 0,
            'actions_taken': [],
            'events': {
                'collisions': 0,
                'successful_landings': 0,
                'successful_departures': 0,
                'separation_losses': 0,
                'go_arounds': 0
            }
        }
        
        step = 0
        while not done and step < 1000:  # Max steps safety
            # Get action
            obs_tensor = ppo._obs_to_tensor(obs)
            
            with torch.no_grad():
                action, log_prob, entropy, value = ppo.policy.get_action_and_value(obs_tensor)
            
            action_np = ppo._action_to_numpy(action)
            
            # Record action
            episode_data['actions_taken'].append({
                'step': step,
                'aircraft_id': int(action_np['aircraft_id']),
                'command_type': int(action_np['command_type']),
                'value_estimate': float(value.item())
            })
            
            # Step
            obs, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated
            
            episode_data['total_reward'] += reward
            episode_data['length'] += 1
            
            step += 1
        
        # Get final info
        episode_data['final_score'] = info.get('score', 0)
        episode_data['aircraft_count'] = info.get('aircraft_count', 0)
        episode_data['conflict_count'] = info.get('conflict_count', 0)
        
        results['episodes'].append(episode_data)
    
    # Compute statistics
    rewards = [ep['total_reward'] for ep in results['episodes']]
    lengths = [ep['length'] for ep in results['episodes']]
    scores = [ep['final_score'] for ep in results['episodes']]
    
    results['statistics'] = {
        'mean_reward': float(np.mean(rewards)),
        'std_reward': float(np.std(rewards)),
        'min_reward': float(np.min(rewards)),
        'max_reward': float(np.max(rewards)),
        'mean_length': float(np.mean(lengths)),
        'mean_score': float(np.mean(scores)),
        'std_score': float(np.std(scores)),
        'success_rate': float(np.mean([s > 0 for s in scores]))
    }
    
    return results


def plot_results(results: dict, output_dir: Path):
    """Create visualization plots"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    episodes = results['episodes']
    
    # Reward distribution
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Reward over episodes
    rewards = [ep['total_reward'] for ep in episodes]
    axes[0, 0].plot(rewards, marker='o')
    axes[0, 0].axhline(y=np.mean(rewards), color='r', linestyle='--', label='Mean')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].set_title('Reward per Episode')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Score distribution
    scores = [ep['final_score'] for ep in episodes]
    axes[0, 1].hist(scores, bins=20, edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(x=np.mean(scores), color='r', linestyle='--', label=f'Mean: {np.mean(scores):.1f}')
    axes[0, 1].set_xlabel('Final Score')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Score Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Episode length
    lengths = [ep['length'] for ep in episodes]
    axes[1, 0].bar(range(len(lengths)), lengths, alpha=0.7)
    axes[1, 0].axhline(y=np.mean(lengths), color='r', linestyle='--', label='Mean')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Length (steps)')
    axes[1, 0].set_title('Episode Length')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Summary statistics
    stats = results['statistics']
    stats_text = (
        f"Mean Reward: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}\n"
        f"Mean Score: {stats['mean_score']:.1f} ± {stats['std_score']:.1f}\n"
        f"Mean Length: {stats['mean_length']:.1f}\n"
        f"Success Rate: {stats['success_rate']:.1%}\n"
        f"Min/Max Reward: {stats['min_reward']:.1f} / {stats['max_reward']:.1f}"
    )
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
                    family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[1, 1].axis('off')
    axes[1, 1].set_title('Summary Statistics')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'evaluation_summary.png', dpi=150, bbox_inches='tight')
    print(f"Saved plot: {output_dir / 'evaluation_summary.png'}")
    plt.close()


def main():
    args = parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup device
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create environment
    print("Creating environment...")
    env_config = config['env']
    env = OpenScopeEnv(
        game_url=env_config['game_url'],
        airport=env_config['airport'],
        timewarp=env_config['timewarp'],
        max_aircraft=env_config['max_aircraft'],
        episode_length=env_config['episode_length'],
        action_interval=env_config['action_interval'],
        headless=not args.render,
        config=config
    )
    
    # Create agent
    print("Creating agent...")
    network_config = config['network']
    action_config = config['actions']
    
    action_space_sizes = {
        'aircraft_id': network_config['max_aircraft_slots'] + 1,
        'command_type': 5,  # altitude, heading, speed, ils, direct (fixed)
        'altitude': len(action_config['altitudes']),
        'heading': len(action_config['heading_changes']),
        'speed': len(action_config['speeds'])
    }
    
    policy = ATCActorCritic(
        aircraft_feature_dim=network_config['aircraft_feature_dim'],
        global_feature_dim=network_config['global_feature_dim'],
        hidden_dim=network_config['hidden_dim'],
        num_heads=network_config['num_attention_heads'],
        num_layers=network_config['num_transformer_layers'],
        max_aircraft=network_config['max_aircraft_slots'],
        action_space_sizes=action_space_sizes
    )
    
    ppo_config = config['ppo']
    ppo = PPO(
        policy=policy,
        device=device,
        **ppo_config
    )
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ppo.load(args.checkpoint)
    
    # Evaluate
    print(f"\nEvaluating for {args.n_episodes} episodes...")
    results = evaluate_agent(env, ppo, args.n_episodes, args.render)
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    stats = results['statistics']
    print(f"Mean Reward:   {stats['mean_reward']:8.2f} ± {stats['std_reward']:.2f}")
    print(f"Min/Max:       {stats['min_reward']:8.2f} / {stats['max_reward']:.2f}")
    print(f"Mean Score:    {stats['mean_score']:8.1f} ± {stats['std_score']:.1f}")
    print(f"Mean Length:   {stats['mean_length']:8.1f} steps")
    print(f"Success Rate:  {stats['success_rate']:8.1%}")
    print("="*60)
    
    # Save results
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")
    
    # Create plots
    plot_dir = output_path.parent / 'plots'
    plot_results(results, plot_dir)
    
    # Cleanup
    env.close()


if __name__ == "__main__":
    main()

