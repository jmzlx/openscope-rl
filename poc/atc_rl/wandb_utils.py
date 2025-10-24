"""
Weights & Biases utilities for ATC RL project.

This module provides utility functions for integrating wandb with the ATC RL notebooks.
"""

import wandb
import numpy as np
from typing import Dict, Any, Optional, List
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv


class WandbATCCallback(BaseCallback):
    """
    Custom wandb callback for ATC RL training that tracks ATC-specific metrics.
    """
    
    def __init__(self, 
                 project_name: str = "atc-rl-poc",
                 run_name: Optional[str] = None,
                 config: Optional[Dict[str, Any]] = None,
                 verbose: int = 0):
        super().__init__(verbose)
        
        self.project_name = project_name
        self.run_name = run_name
        self.config = config or {}
        
        # Episode tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.separation_violations = []
        self.successful_exits = []
        self.num_aircraft_per_episode = []
        
        # Initialize wandb
        self._init_wandb()
        
    def _init_wandb(self):
        """Initialize wandb run."""
        wandb.init(
            project=self.project_name,
            name=self.run_name,
            config=self.config,
            reinit=True
        )
        
        # Define custom metrics
        wandb.define_metric("success_rate", summary="max")
        wandb.define_metric("safety_score", summary="max")
        wandb.define_metric("efficiency_score", summary="max")
        wandb.define_metric("avg_reward_10_episodes", summary="max")
        wandb.define_metric("avg_exits_10_episodes", summary="max")
        wandb.define_metric("avg_violations_10_episodes", summary="min")
        
    def _on_step(self) -> bool:
        """Called at each step during training."""
        # Check if episode ended
        if self.locals.get('dones')[0]:
            info = self.locals.get('infos')[0]
            
            # Store episode data
            self.episode_rewards.append(info['total_reward'])
            self.episode_lengths.append(info['step'])
            self.separation_violations.append(info['separations_lost'])
            self.successful_exits.append(info['successful_exits'])
            self.num_aircraft_per_episode.append(info['num_aircraft'])
            
            # Calculate ATC-specific metrics
            success_rate = info['successful_exits'] / max(info['num_aircraft'], 1)
            safety_score = 1.0 / (1.0 + info['separations_lost'])
            efficiency_score = info['successful_exits'] / max(info['step'], 1)
            
            # Log episode metrics
            wandb.log({
                "episode_reward": info['total_reward'],
                "episode_length": info['step'],
                "separation_violations": info['separations_lost'],
                "successful_exits": info['successful_exits'],
                "num_aircraft": info['num_aircraft'],
                "success_rate": success_rate,
                "safety_score": safety_score,
                "efficiency_score": efficiency_score,
            })
            
            # Log rolling averages every 10 episodes
            if len(self.episode_rewards) % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                avg_exits = np.mean(self.successful_exits[-10:])
                avg_violations = np.mean(self.separation_violations[-10:])
                avg_length = np.mean(self.episode_lengths[-10:])
                
                wandb.log({
                    "avg_reward_10_episodes": avg_reward,
                    "avg_exits_10_episodes": avg_exits,
                    "avg_violations_10_episodes": avg_violations,
                    "avg_length_10_episodes": avg_length,
                })
                
                # Log training progress to wandb instead of printing
                wandb.log({
                    "training_progress": {
                        "episode": len(self.episode_rewards),
                        "avg_reward": avg_reward,
                        "avg_exits": avg_exits,
                        "avg_violations": avg_violations,
                        "avg_length": avg_length
                    }
                })
                
                # Only print summary every 50 episodes to keep notebook clean
                if len(self.episode_rewards) % 50 == 0:
                    print(f"📊 Episode {len(self.episode_rewards)}: "
                          f"Avg Reward = {avg_reward:.1f}, "
                          f"Avg Exits = {avg_exits:.1f}, "
                          f"Avg Violations = {avg_violations:.1f}")
        
        return True
    
    def on_training_end(self) -> None:
        """Called when training ends."""
        # Log final summary statistics
        if self.episode_rewards:
            wandb.log({
                "final_avg_reward": np.mean(self.episode_rewards),
                "final_avg_exits": np.mean(self.successful_exits),
                "final_avg_violations": np.mean(self.separation_violations),
                "total_episodes": len(self.episode_rewards),
            })
        
        wandb.finish()


def setup_wandb_experiment(project_name: str = "atc-rl-poc",
                          run_name: Optional[str] = None,
                          config: Optional[Dict[str, Any]] = None,
                          tags: Optional[List[str]] = None) -> wandb.run:
    """
    Set up a wandb experiment with ATC-specific configuration.
    
    Args:
        project_name: Name of the wandb project
        run_name: Name of the specific run
        config: Configuration dictionary
        tags: List of tags for the run
        
    Returns:
        wandb run object
    """
    default_config = {
        "environment": "Simple2DATC",
        "algorithm": "PPO",
        "max_aircraft": 5,
        "max_steps": 100,
        "learning_rate": 3e-4,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
    }
    
    if config:
        default_config.update(config)
    
    run = wandb.init(
        project=project_name,
        name=run_name,
        config=default_config,
        tags=tags or [],
        reinit=True
    )
    
    return run


def log_model_evaluation(model_path: str, 
                        env: Any, 
                        num_episodes: int = 10,
                        deterministic: bool = True) -> Dict[str, float]:
    """
    Evaluate a trained model and log results to wandb.
    
    Args:
        model_path: Path to the saved model
        env: Environment to evaluate on
        num_episodes: Number of episodes to run
        deterministic: Whether to use deterministic actions
        
    Returns:
        Dictionary with evaluation metrics
    """
    from stable_baselines3 import PPO
    
    # Load model
    model = PPO.load(model_path)
    
    total_rewards = []
    total_exits = []
    total_violations = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        
        for step in range(200):  # Max steps per episode
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            if terminated or truncated:
                break
        
        total_rewards.append(episode_reward)
        total_exits.append(info['successful_exits'])
        total_violations.append(info['separations_lost'])
        episode_lengths.append(info['step'])
    
    # Calculate metrics
    metrics = {
        "eval_avg_reward": np.mean(total_rewards),
        "eval_std_reward": np.std(total_rewards),
        "eval_avg_exits": np.mean(total_exits),
        "eval_avg_violations": np.mean(total_violations),
        "eval_avg_length": np.mean(episode_lengths),
        "eval_success_rate": np.mean(total_exits) / np.mean(episode_lengths),
        "eval_safety_score": 1.0 / (1.0 + np.mean(total_violations)),
    }
    
    # Log to wandb
    wandb.log(metrics)
    
    return metrics


def create_atc_wandb_table(episode_data: List[Dict[str, Any]]) -> wandb.Table:
    """
    Create a wandb table for visualizing episode data.
    
    Args:
        episode_data: List of dictionaries containing episode information
        
    Returns:
        wandb Table object
    """
    table = wandb.Table(columns=[
        "episode", "reward", "length", "exits", "violations", 
        "aircraft_count", "success_rate", "safety_score"
    ])
    
    for i, data in enumerate(episode_data):
        success_rate = data['exits'] / max(data['aircraft_count'], 1)
        safety_score = 1.0 / (1.0 + data['violations'])
        
        table.add_data(
            i,
            data['reward'],
            data['length'],
            data['exits'],
            data['violations'],
            data['aircraft_count'],
            success_rate,
            safety_score
        )
    
    return table


def log_training_comparison(results: Dict[str, Dict[str, float]]):
    """
    Log comparison between different training runs.
    
    Args:
        results: Dictionary mapping run names to their metrics
    """
    comparison_table = wandb.Table(columns=[
        "run_name", "avg_reward", "avg_exits", "avg_violations", 
        "success_rate", "safety_score", "efficiency_score"
    ])
    
    for run_name, metrics in results.items():
        comparison_table.add_data(
            run_name,
            metrics.get('avg_reward', 0),
            metrics.get('avg_exits', 0),
            metrics.get('avg_violations', 0),
            metrics.get('success_rate', 0),
            metrics.get('safety_score', 0),
            metrics.get('efficiency_score', 0),
        )
    
    wandb.log({"training_comparison": comparison_table})


def save_model_with_wandb(model, model_name: str, metadata: Optional[Dict[str, Any]] = None):
    """
    Save model and log to wandb with metadata.
    
    Args:
        model: Trained model to save
        model_name: Name for the saved model
        metadata: Additional metadata to log
    """
    model_path = f"models/{model_name}"
    model.save(model_path)
    
    # Log model artifact
    artifact = wandb.Artifact(f"model-{model_name}", type="model")
    artifact.add_file(f"{model_path}.zip")
    
    if metadata:
        artifact.metadata.update(metadata)
    
    wandb.log_artifact(artifact)
    
    return model_path
