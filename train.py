"""
Main training script for OpenScope RL agent
"""

import torch
import numpy as np
import yaml
import argparse
from pathlib import Path
import wandb
from tqdm import tqdm
import json
from datetime import datetime

from environment.openscope_env import OpenScopeEnv
from models.networks import ATCActorCritic
from algorithms.ppo import PPO
from utils.logger import Logger
from utils.curriculum import CurriculumManager


def parse_args():
    parser = argparse.ArgumentParser(description="Train OpenScope RL Agent")
    parser.add_argument("--config", type=str, default="config/training_config.yaml",
                        help="Path to config file")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--eval-only", action="store_true",
                        help="Only run evaluation, no training")
    parser.add_argument("--wandb", action="store_true",
                        help="Use Weights & Biases logging")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda/cpu)")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load training configuration"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_environment(config: dict) -> OpenScopeEnv:
    """Create the OpenScope environment"""
    env_config = config['env']
    return OpenScopeEnv(
        game_url=env_config['game_url'],
        airport=env_config['airport'],
        timewarp=env_config['timewarp'],
        max_aircraft=env_config['max_aircraft'],
        episode_length=env_config['episode_length'],
        action_interval=env_config['action_interval'],
        headless=env_config['headless'],
        config=config
    )


def create_agent(config: dict, device: str) -> tuple:
    """Create the PPO agent with actor-critic network"""
    network_config = config['network']
    ppo_config = config['ppo']
    action_config = config['actions']
    
    # Calculate action space sizes
    action_space_sizes = {
        'aircraft_id': network_config['max_aircraft_slots'] + 1,
        'command_type': 5,  # altitude, heading, speed, ils, direct (fixed)
        'altitude': len(action_config['altitudes']),
        'heading': len(action_config['heading_changes']),
        'speed': len(action_config['speeds'])
    }
    
    # Create actor-critic network
    policy = ATCActorCritic(
        aircraft_feature_dim=network_config['aircraft_feature_dim'],
        global_feature_dim=network_config['global_feature_dim'],
        hidden_dim=network_config['hidden_dim'],
        num_heads=network_config['num_attention_heads'],
        num_layers=network_config['num_transformer_layers'],
        max_aircraft=network_config['max_aircraft_slots'],
        action_space_sizes=action_space_sizes
    )
    
    # Create PPO agent
    ppo = PPO(
        policy=policy,
        learning_rate=ppo_config['learning_rate'],
        gamma=ppo_config['gamma'],
        gae_lambda=ppo_config['gae_lambda'],
        clip_epsilon=ppo_config['clip_epsilon'],
        value_coef=ppo_config['value_coef'],
        entropy_coef=ppo_config['entropy_coef'],
        max_grad_norm=ppo_config['max_grad_norm'],
        n_steps=ppo_config['n_steps'],
        n_epochs=ppo_config['n_epochs'],
        batch_size=ppo_config['batch_size'],
        device=device
    )
    
    return policy, ppo


def evaluate(env: OpenScopeEnv, ppo: PPO, n_episodes: int = 10) -> dict:
    """Evaluate the agent"""
    ppo.policy.eval()
    
    episode_rewards = []
    episode_lengths = []
    episode_scores = []
    collisions = 0
    successful_landings = 0
    successful_departures = 0
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        while not done:
            # Get observation tensor
            obs_tensor = ppo._obs_to_tensor(obs)
            
            # Get action (deterministic evaluation)
            with torch.no_grad():
                action, _, _, _ = ppo.policy.get_action_and_value(obs_tensor)
            
            # Convert to numpy
            action_np = ppo._action_to_numpy(action)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_scores.append(info.get('score', 0))
    
    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'mean_score': np.mean(episode_scores),
        'episode_rewards': episode_rewards
    }


def train(config: dict, args):
    """Main training loop"""
    
    # Setup
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create directories
    checkpoint_dir = Path(config['training']['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    log_dir = Path(config['training']['log_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize wandb
    if args.wandb and config['training']['use_wandb']:
        wandb.init(
            project=config['training']['wandb_project'],
            config=config,
            name=f"openscope_ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    
    # Create environment
    print("Creating environment...")
    env = create_environment(config)
    
    # Create agent
    print("Creating agent...")
    policy, ppo = create_agent(config, device)
    
    # Load checkpoint if provided
    start_step = 0
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        ppo.load(args.checkpoint)
        # Extract step number from checkpoint name if possible
        try:
            start_step = int(Path(args.checkpoint).stem.split('_')[-1])
        except:
            pass
    
    # Curriculum manager
    curriculum = CurriculumManager(config['curriculum']) if config['curriculum']['enabled'] else None
    
    # Logger
    logger = Logger(log_dir)
    
    # Training loop
    total_timesteps = config['training']['total_timesteps']
    save_interval = config['training']['save_interval']
    eval_interval = config['training']['eval_interval']
    
    print(f"\nStarting training for {total_timesteps} timesteps...")
    print(f"Network parameters: {sum(p.numel() for p in policy.parameters()):,}")
    
    global_step = start_step
    episode_count = 0
    
    pbar = tqdm(total=total_timesteps, initial=start_step)
    
    try:
        while global_step < total_timesteps:
            # Update curriculum if enabled
            if curriculum:
                curriculum.update(episode_count)
                if curriculum.should_update_env():
                    env_updates = curriculum.get_env_updates()
                    # Update environment settings
                    env.max_aircraft = env_updates.get('max_aircraft', env.max_aircraft)
            
            # Collect rollouts
            rollout_stats = ppo.collect_rollouts(env, ppo.n_steps)
            episode_count += len(rollout_stats['episode_rewards'])
            
            # Train
            train_stats = ppo.train()
            
            # Update global step
            global_step += ppo.n_steps
            pbar.update(ppo.n_steps)
            
            # Logging
            log_data = {
                'global_step': global_step,
                'episode_count': episode_count,
                **rollout_stats,
                **train_stats
            }
            
            if curriculum:
                log_data['curriculum_stage'] = curriculum.current_stage
            
            logger.log(log_data)
            
            if args.wandb and config['training']['use_wandb']:
                wandb.log(log_data, step=global_step)
            
            # Console output
            if len(rollout_stats['episode_rewards']) > 0:
                pbar.set_postfix({
                    'reward': f"{rollout_stats['mean_reward']:.2f}",
                    'length': f"{rollout_stats['mean_length']:.1f}",
                    'policy_loss': f"{train_stats['policy_loss']:.4f}",
                    'value_loss': f"{train_stats['value_loss']:.4f}"
                })
            
            # Save checkpoint
            if global_step % save_interval == 0:
                checkpoint_path = checkpoint_dir / f"checkpoint_{global_step}.pt"
                ppo.save(str(checkpoint_path))
                print(f"\nCheckpoint saved: {checkpoint_path}")
            
            # Evaluation
            if global_step % eval_interval == 0:
                print("\nRunning evaluation...")
                eval_stats = evaluate(env, ppo, config['training']['eval_episodes'])
                
                eval_log = {
                    'eval/mean_reward': eval_stats['mean_reward'],
                    'eval/std_reward': eval_stats['std_reward'],
                    'eval/mean_length': eval_stats['mean_length'],
                    'eval/mean_score': eval_stats['mean_score']
                }
                
                logger.log(eval_log)
                
                if args.wandb and config['training']['use_wandb']:
                    wandb.log(eval_log, step=global_step)
                
                print(f"Eval - Mean Reward: {eval_stats['mean_reward']:.2f} "
                      f"(±{eval_stats['std_reward']:.2f}), "
                      f"Mean Score: {eval_stats['mean_score']:.1f}")
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    finally:
        # Save final checkpoint
        final_checkpoint = checkpoint_dir / f"checkpoint_final_{global_step}.pt"
        ppo.save(str(final_checkpoint))
        print(f"\nFinal checkpoint saved: {final_checkpoint}")
        
        # Close environment
        env.close()
        
        pbar.close()
        
        if args.wandb and config['training']['use_wandb']:
            wandb.finish()
    
    print("\nTraining complete!")


def main():
    args = parse_args()
    config = load_config(args.config)
    
    # Set random seeds
    seed = config['training']['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    if args.eval_only:
        # Run evaluation only
        print("Evaluation mode")
        if not args.checkpoint:
            raise ValueError("Must provide checkpoint for evaluation")
        
        device = args.device if torch.cuda.is_available() else "cpu"
        env = create_environment(config)
        policy, ppo = create_agent(config, device)
        ppo.load(args.checkpoint)
        
        eval_stats = evaluate(env, ppo, n_episodes=20)
        
        print("\n=== Evaluation Results ===")
        print(f"Mean Reward: {eval_stats['mean_reward']:.2f} (±{eval_stats['std_reward']:.2f})")
        print(f"Mean Length: {eval_stats['mean_length']:.1f}")
        print(f"Mean Score: {eval_stats['mean_score']:.1f}")
        print(f"\nEpisode Rewards: {eval_stats['episode_rewards']}")
        
        env.close()
    else:
        # Run training
        train(config, args)


if __name__ == "__main__":
    main()

