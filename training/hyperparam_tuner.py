"""
Optuna-based hyperparameter optimization for OpenScope RL.

This module provides systematic hyperparameter tuning using Optuna with:
- Algorithm-specific parameter samplers
- Pruning support (SuccessiveHalving, Median)
- Periodic evaluation during training
- CSV/DataFrame export of results
"""

import logging
from copy import deepcopy
from typing import Dict, Any, Optional, Callable
import numpy as np
import pandas as pd
import optuna
from optuna.pruners import SuccessiveHalvingPruner, MedianPruner
from optuna.samplers import TPESampler, RandomSampler

logger = logging.getLogger(__name__)


def sample_ppo_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sample PPO hyperparameters for Optuna trial.
    
    Args:
        trial: Optuna trial object
        
    Returns:
        Dictionary of hyperparameters
    """
    return {
        'learning_rate': trial.suggest_float('lr', 1e-5, 1e-3, log=True),
        'n_steps': trial.suggest_categorical('n_steps', [1024, 2048, 4096]),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
        'gamma': trial.suggest_categorical('gamma', [0.95, 0.98, 0.99, 0.995]),
        'gae_lambda': trial.suggest_categorical('gae_lambda', [0.9, 0.95, 0.98]),
        'ent_coef': trial.suggest_float('ent_coef', 1e-4, 0.1, log=True),
        'clip_range': trial.suggest_categorical('clip_range', [0.1, 0.2, 0.3]),
        'n_epochs': trial.suggest_categorical('n_epochs', [5, 10, 20]),
        'vf_coef': trial.suggest_float('vf_coef', 0.3, 0.7),
        'max_grad_norm': trial.suggest_float('max_grad_norm', 0.3, 1.0),
    }


def sample_dt_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sample Decision Transformer hyperparameters for Optuna trial.
    
    Args:
        trial: Optuna trial object
        
    Returns:
        Dictionary of hyperparameters
    """
    return {
        'hidden_size': trial.suggest_categorical('hidden_size', [128, 256, 512]),
        'n_layer': trial.suggest_int('n_layer', 4, 12),
        'n_head': trial.suggest_categorical('n_head', [4, 8]),
        'context_len': trial.suggest_categorical('context_len', [10, 20, 30]),
        'learning_rate': trial.suggest_float('lr', 1e-5, 1e-3, log=True),
        'dropout': trial.suggest_float('dropout', 0.0, 0.3),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
    }


HYPERPARAMS_SAMPLER = {
    'ppo': sample_ppo_params,
    'dt': sample_dt_params,
}


class HyperparamTuner:
    """
    Optuna-based hyperparameter optimization for OpenScope RL.
    
    This class provides systematic hyperparameter tuning with pruning
    and periodic evaluation during training.
    
    Example:
        >>> from training.ppo_trainer import create_vec_env, create_ppo_model
        >>> from environment import create_default_config
        >>> 
        >>> config = create_default_config(airport="KLAS", max_aircraft=10)
        >>> env_config = {
        ...     "airport": config.airport,
        ...     "max_aircraft": config.max_aircraft,
        ...     "headless": True,
        ... }
        >>> 
        >>> def env_factory(n_envs=1):
        ...     return create_vec_env(n_envs, env_config, seed=0)
        >>> 
        >>> def model_factory(**kwargs):
        ...     env = env_factory(n_envs=8)
        ...     return create_ppo_model(env, **kwargs)
        >>> 
        >>> tuner = HyperparamTuner(
        ...     algo='ppo',
        ...     env_factory=env_factory,
        ...     model_factory=model_factory,
        ...     n_trials=20,
        ...     n_timesteps=50000
        ... )
        >>> results = tuner.optimize()
    """
    
    def __init__(
        self,
        algo: str,
        env_factory: Callable,
        model_factory: Callable,
        n_trials: int = 50,
        n_timesteps: int = 100000,
        sampler: str = "tpe",
        pruner: str = "halving",
        n_jobs: int = 1,
        n_eval_episodes: int = 5,
        eval_interval: Optional[int] = None,
    ):
        # Validate inputs
        if not callable(env_factory):
            raise TypeError("env_factory must be callable")
        if not callable(model_factory):
            raise TypeError("model_factory must be callable")
        
        # Quick validation that env_factory works
        try:
            test_env = env_factory(n_envs=1)
            if not hasattr(test_env, 'reset') or not hasattr(test_env, 'step'):
                raise ValueError("env_factory must return an object with reset() and step() methods")
            test_env.close()
        except Exception as e:
            raise ValueError(f"env_factory validation failed: {e}") from e
        """
        Initialize hyperparameter tuner.
        
        Args:
            algo: Algorithm name ('ppo', 'dt')
            env_factory: Function that creates environment (n_envs) -> env
            model_factory: Function that creates model (**kwargs) -> model
            n_trials: Number of optimization trials
            n_timesteps: Number of timesteps per trial
            sampler: Sampler method ('tpe', 'random')
            pruner: Pruner method ('halving', 'median', 'none')
            n_jobs: Number of parallel jobs (1 for sequential)
            n_eval_episodes: Number of episodes for evaluation
            eval_interval: Evaluation interval in timesteps (auto if None)
        """
        if algo not in HYPERPARAMS_SAMPLER:
            raise ValueError(f"Unknown algorithm: {algo}. Available: {list(HYPERPARAMS_SAMPLER.keys())}")
        
        self.algo = algo
        self.env_factory = env_factory
        self.model_factory = model_factory
        self.n_trials = n_trials
        self.n_timesteps = n_timesteps
        self.n_eval_episodes = n_eval_episodes
        self.eval_interval = eval_interval or (n_timesteps // 20)  # 20 evaluations per trial
        self.n_jobs = n_jobs
        
        # Create sampler
        self.sampler = self._create_sampler(sampler)
        
        # Create pruner
        self.pruner = self._create_pruner(pruner)
        
        logger.info(f"HyperparamTuner initialized: algo={algo}, n_trials={n_trials}, "
                   f"n_timesteps={n_timesteps}, sampler={sampler}, pruner={pruner}")
    
    def _create_sampler(self, sampler_method: str):
        """Create Optuna sampler."""
        if sampler_method == 'random':
            return RandomSampler()
        elif sampler_method == 'tpe':
            return TPESampler(n_startup_trials=5)
        else:
            raise ValueError(f"Unknown sampler: {sampler_method}")
    
    def _create_pruner(self, pruner_method: str):
        """Create Optuna pruner."""
        if pruner_method == 'halving':
            return SuccessiveHalvingPruner(min_resource=1, reduction_factor=4, min_early_stopping_rate=0)
        elif pruner_method == 'median':
            return MedianPruner(n_startup_trials=5, n_warmup_steps=3)
        elif pruner_method == 'none':
            return MedianPruner(n_startup_trials=self.n_trials, n_warmup_steps=self.n_trials)
        else:
            raise ValueError(f"Unknown pruner: {pruner_method}")
    
    def _evaluate_model(self, model, test_env, n_episodes: int) -> float:
        """
        Evaluate model on test environment.
        
        Args:
            model: Trained model
            test_env: Test environment
            n_episodes: Number of evaluation episodes
            
        Returns:
            Mean reward across episodes
        """
        rewards = []
        
        for _ in range(n_episodes):
            obs = test_env.reset()
            episode_reward = 0.0
            done = False
            
            while not done:
                try:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, done, info = test_env.step(action)
                    
                    # Handle vectorized environments
                    if isinstance(done, (list, tuple, np.ndarray)):
                        done = done[0] if len(done) > 0 else False
                    if isinstance(reward, (list, tuple, np.ndarray)):
                        reward = reward[0] if len(reward) > 0 else 0.0
                    
                    episode_reward += float(reward)
                except Exception as e:
                    logger.error(f"Step failed during evaluation: {e}")
                    break
                
                if done:
                    break
            
            rewards.append(episode_reward)
        
        return np.mean(rewards)
    
    def _objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for Optuna optimization.
        
        Args:
            trial: Optuna trial
            
        Returns:
            Negative mean reward (for minimization, but we maximize)
        """
        # Sample hyperparameters
        algo_sampler = HYPERPARAMS_SAMPLER[self.algo]
        hyperparams = algo_sampler(trial)
        
        # Create model with sampled hyperparameters
        try:
            model = self.model_factory(**hyperparams)
        except Exception as e:
            logger.warning(f"Failed to create model with hyperparams {hyperparams}: {e}")
            raise optuna.TrialPruned()
        
        # Create test environment
        test_env = self.env_factory(n_envs=1)
        
        # Sync VecNormalize stats if applicable
        if hasattr(model, 'env') and hasattr(model.env, 'obs_rms'):
            if hasattr(test_env, 'obs_rms'):
                test_env.obs_rms = deepcopy(model.env.obs_rms)
                test_env.norm_reward = False
        
        # Track evaluation
        eval_idx = 0
        last_eval_timestep = 0
        best_reward = -np.inf
        
        def eval_callback(locals_dict: Dict[str, Any], globals_dict: Dict[str, Any]) -> bool:
            """Callback for periodic evaluation during training."""
            nonlocal eval_idx, last_eval_timestep, best_reward
            
            self_ = locals_dict['self']
            current_timesteps = self_.num_timesteps
            
            # Check if it's time to evaluate
            if (current_timesteps - last_eval_timestep) < self.eval_interval:
                return True
            
            last_eval_timestep = current_timesteps
            eval_idx += 1
            
            # Evaluate model
            try:
                mean_reward = self._evaluate_model(model, test_env, self.n_eval_episodes)
                best_reward = max(best_reward, mean_reward)
                
                # Report to Optuna
                trial.report(mean_reward, eval_idx)
                
                # Check if should prune
                if trial.should_prune():
                    logger.info(f"Trial {trial.number} pruned at step {current_timesteps}")
                    return False
                
            except Exception as e:
                logger.error(f"Evaluation failed: {e}")
                # If evaluation consistently fails, stop training
                return False
            
            return True
        
        # Train model
        try:
            model.learn(
                total_timesteps=self.n_timesteps,
                callback=eval_callback,
                progress_bar=False,
            )
        except Exception as e:
            logger.error(f"Training failed: {e}")
            # Clean up before raising
            try:
                if hasattr(model, 'env'):
                    model.env.close()
            except Exception:
                pass
            try:
                test_env.close()
            except Exception:
                pass
            raise optuna.TrialPruned()
        
        # Final evaluation
        try:
            final_reward = self._evaluate_model(model, test_env, self.n_eval_episodes)
            best_reward = max(best_reward, final_reward)
        except Exception as e:
            logger.warning(f"Final evaluation failed: {e}")
            # Use best reward from intermediate evaluations if available
            if best_reward == -np.inf:
                logger.error("No valid evaluations completed, pruning trial")
                try:
                    if hasattr(model, 'env'):
                        model.env.close()
                except Exception:
                    pass
                try:
                    test_env.close()
                except Exception:
                    pass
                raise optuna.TrialPruned()
        
        # Clean up
        try:
            if hasattr(model, 'env'):
                model.env.close()
        except Exception:
            pass
        try:
            test_env.close()
        except Exception:
            pass
        del model
        
        # Return negative reward (Optuna minimizes, but we want to maximize)
        return -best_reward
    
    def optimize(self, timeout: Optional[int] = None) -> pd.DataFrame:
        """
        Run hyperparameter optimization.
        
        Args:
            timeout: Maximum time in seconds (None for no limit)
            
        Returns:
            DataFrame with trial results
        """
        study = optuna.create_study(
            sampler=self.sampler,
            pruner=self.pruner,
            direction="minimize",  # We minimize negative reward = maximize reward
        )
        
        try:
            study.optimize(
                self._objective,
                n_trials=self.n_trials,
                n_jobs=self.n_jobs,
                timeout=timeout,
                catch=(ValueError, AssertionError, RuntimeError),
            )
        except KeyboardInterrupt:
            logger.info("Optimization interrupted by user")
        
        logger.info(f"Optimization complete: {len(study.trials)} trials finished")
        
        if study.best_trial:
            logger.info(f"Best trial: {study.best_trial.number}")
            logger.info(f"Best value: {study.best_trial.value}")
            logger.info(f"Best params: {study.best_trial.params}")
        
        return study.trials_dataframe()

