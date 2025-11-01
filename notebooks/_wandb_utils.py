"""
Weights & Biases utilities for OpenScope RL notebooks.

This module provides utility functions for integrating wandb with the OpenScope RL notebooks.
Independent of the POC implementation to keep main project and POC separate.
"""

import os
from typing import Dict, Any, Optional, List
import wandb
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class WandbATCCallback(BaseCallback):
    """
    Custom wandb callback for OpenScope RL training that tracks ATC-specific metrics.
    """
    
    def __init__(self, 
                 project_name: str = "openscope-rl",
                 run_name: Optional[str] = None,
                 config: Optional[Dict[str, Any]] = None,
                 entity: Optional[str] = None,
                 sync_tensorboard: bool = True,
                 offline: bool = False,
                 verbose: int = 0):
        """
        Initialize the callback.
        
        Args:
            project_name: Name of the wandb project
            run_name: Name of the specific run
            config: Configuration dictionary
            sync_tensorboard: Whether to sync tensorboard logs
            verbose: Verbosity level
        """
        super().__init__(verbose)
        
        self.project_name = project_name
        self.run_name = run_name
        self.config = config or {}
        self.entity = entity
        self.sync_tensorboard = sync_tensorboard
        self.offline = offline
        
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
        # Check if wandb is already initialized
        if wandb.run is not None:
            print(f"✅ Using existing WandB run: {wandb.run.name}")
            return
        
        # Ensure API key is set if available in environment
        if os.environ.get("WANDB_API_KEY"):
            try:
                wandb.login(key=os.environ.get("WANDB_API_KEY"), relogin=True)
            except Exception as e:
                print(f"⚠️  WandB login warning: {e}")
        
        # Set offline mode if requested
        mode = "offline" if self.offline else "online"
        
        init_kwargs = {
            "project": self.project_name,
            "name": self.run_name,
            "config": self.config,
            "sync_tensorboard": self.sync_tensorboard,
            "reinit": False,  # Don't reinit if run exists
            "mode": mode,
            "resume": "allow",  # Allow resuming runs
        }
        
        if self.entity:
            init_kwargs["entity"] = self.entity
        
        try:
            wandb.init(**init_kwargs)
            print(f"✅ WandB initialized in {mode} mode")
        except Exception as e:
            # If permission error, try offline mode
            if "permission" in str(e).lower() or "403" in str(e) or "unauthorized" in str(e).lower():
                print("⚠️  Permission error in callback. Switching to offline mode...")
                print("   Possible causes:")
                print("   1. Project doesn't exist - create it at https://wandb.ai")
                print("   2. Incorrect entity name - check your username")
                print("   3. Not logged in - run 'wandb login' in terminal")
                print("   You can sync offline runs later with: wandb sync wandb/offline-run-*")
                init_kwargs["mode"] = "offline"
                self.offline = True
                wandb.init(**init_kwargs)
            else:
                print(f"⚠️  WandB initialization error: {e}")
                print("   Falling back to offline mode...")
                init_kwargs["mode"] = "offline"
                self.offline = True
                wandb.init(**init_kwargs)
        
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
        dones = self.locals.get('dones', [False])
        if dones and dones[0]:
            infos = self.locals.get('infos', [{}])
            info = infos[0] if infos else {}
            
            # Extract episode metrics from info dict
            # Handle different info formats from SB3 Monitor wrapper
            episode_metrics = info.get('episode', {})
            if episode_metrics:
                # Standard SB3 episode metrics
                episode_reward = episode_metrics.get('r', 0)
                episode_length = episode_metrics.get('l', 0)
            else:
                # Fallback to direct info keys or default values
                episode_reward = info.get('total_reward', info.get('episode_reward', 0))
                episode_length = info.get('step', info.get('episode_length', 0))
            
            # Extract ATC-specific metrics
            separation_violations = info.get('separations_lost', info.get('separation_violations', 0))
            successful_exits = info.get('successful_exits', info.get('exits', 0))
            num_aircraft = info.get('num_aircraft', info.get('aircraft_count', 0))
            
            # Store episode data
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.separation_violations.append(separation_violations)
            self.successful_exits.append(successful_exits)
            self.num_aircraft_per_episode.append(num_aircraft)
            
            # Calculate ATC-specific metrics
            success_rate = successful_exits / max(num_aircraft, 1)
            safety_score = 1.0 / (1.0 + separation_violations)
            efficiency_score = successful_exits / max(episode_length, 1)
            
            # Log episode metrics
            wandb.log({
                "episode_reward": episode_reward,
                "episode_length": episode_length,
                "separation_violations": separation_violations,
                "successful_exits": successful_exits,
                "num_aircraft": num_aircraft,
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


def setup_wandb_experiment(project_name: str = "openscope-rl",
                          run_name: Optional[str] = None,
                          config: Optional[Dict[str, Any]] = None,
                          tags: Optional[List[str]] = None,
                          entity: Optional[str] = None,
                          sync_tensorboard: bool = True,
                          offline: bool = False) -> Optional[Any]:
    """
    Set up a wandb experiment with OpenScope-specific configuration.
    
    Args:
        project_name: Name of the wandb project
        run_name: Name of the specific run
        config: Configuration dictionary
        tags: List of tags for the run
        entity: WandB entity (username/team)
        sync_tensorboard: Whether to sync tensorboard logs
        offline: Whether to use offline mode from start
        
    Returns:
        wandb run object or None if initialization failed
    """
    # Check if wandb is already initialized
    if wandb.run is not None:
        print(f"✅ Using existing WandB run: {wandb.run.name}")
        return wandb.run
    
    default_config = {
        "environment": "PlaywrightEnv",
        "algorithm": "PPO",
        "max_aircraft": 50,
        "episode_length": 600,
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
    
    # Ensure API key is set if available in environment
    if os.environ.get("WANDB_API_KEY"):
        try:
            wandb.login(key=os.environ.get("WANDB_API_KEY"), relogin=True)
        except Exception as e:
            print(f"⚠️  WandB login warning: {e}")
    
    # Set offline mode if requested or if permission errors occur
    mode = "offline" if offline else "online"
    
    init_kwargs = {
        "project": project_name,
        "name": run_name,
        "config": default_config,
        "tags": tags or [],
        "sync_tensorboard": sync_tensorboard,
        "reinit": False,
        "mode": mode,
        "resume": "allow",
    }
    
    if entity:
        init_kwargs["entity"] = entity
    
    try:
        run = wandb.init(**init_kwargs)
        print(f"✅ WandB initialized in {mode} mode")
        if mode == "online" and run.url:
            print(f"   Dashboard: {run.url}")
        return run
    except Exception as e:
        # If permission error, try offline mode
        if "permission" in str(e).lower() or "403" in str(e) or "unauthorized" in str(e).lower():
            print("⚠️  Permission error detected. Trying offline mode...")
            print("   Possible causes:")
            print("   1. Project doesn't exist - create it at https://wandb.ai")
            print("   2. Incorrect entity name - check your username")
            print("   3. Not logged in - run 'wandb login' in terminal")
            print("   You can sync later with: wandb sync wandb/offline-run-*")
            init_kwargs["mode"] = "offline"
            run = wandb.init(**init_kwargs)
            return run
        else:
            print(f"⚠️  WandB initialization error: {e}")
            print("   Falling back to offline mode...")
            init_kwargs["mode"] = "offline"
            run = wandb.init(**init_kwargs)
            return run

