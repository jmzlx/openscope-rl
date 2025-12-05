"""
Configuration logging utilities for OpenScope RL.

This module provides tools for saving and loading training configurations
to YAML files for reproducibility, including git hash, timestamp, and dependencies.
"""

import logging
import os
import yaml
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import sys

logger = logging.getLogger(__name__)


def get_git_hash() -> Optional[str]:
    """
    Get current git commit hash.
    
    Returns:
        Git commit hash or None if not in git repo
    """
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).parent.parent.parent,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_git_branch() -> Optional[str]:
    """
    Get current git branch name.
    
    Returns:
        Git branch name or None if not in git repo
    """
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).parent.parent.parent,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_python_version() -> str:
    """Get Python version."""
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def get_dependencies() -> Dict[str, Optional[str]]:
    """
    Get installed package versions.
    
    Returns:
        Dictionary mapping package names to versions
    """
    dependencies = {}
    
    # Key packages to track
    key_packages = [
        'torch', 'numpy', 'gymnasium', 'stable-baselines3',
        'optuna', 'wandb', 'playwright',
    ]
    
    for package in key_packages:
        try:
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'show', package],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if line.startswith('Version:'):
                        dependencies[package] = line.split(':', 1)[1].strip()
                        break
                else:
                    # Package installed but no version found
                    dependencies[package] = None
            else:
                # Package not installed
                dependencies[package] = None
        except Exception as e:
            logger.debug(f"Could not get version for {package}: {e}")
            dependencies[package] = None
    
    return dependencies


def save_training_config(
    save_dir: str,
    model_config: Dict[str, Any],
    env_config: Dict[str, Any],
    training_config: Dict[str, Any],
    additional_info: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Save training configuration to YAML file.
    
    Args:
        save_dir: Directory to save config file
        model_config: Model hyperparameters
        env_config: Environment configuration
        training_config: Training configuration
        additional_info: Additional information to include
        
    Returns:
        Path to saved config file
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    config_dict = {
        'timestamp': datetime.utcnow().isoformat(),
        'git': {
            'commit_hash': get_git_hash(),
            'branch': get_git_branch(),
        },
        'python_version': get_python_version(),
        'dependencies': get_dependencies(),
        'model_config': model_config,
        'env_config': env_config,
        'training_config': training_config,
    }
    
    if additional_info:
        config_dict['additional_info'] = additional_info
    
    config_file = save_path / 'training_config.yml'
    
    with open(config_file, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"Training config saved to: {config_file}")
    
    return str(config_file)


def load_training_config(config_path: str) -> Dict[str, Any]:
    """
    Load training configuration from YAML file.
    
    Args:
        config_path: Path to config YAML file
        
    Returns:
        Dictionary with configuration
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    logger.info(f"Training config loaded from: {config_path}")
    
    return config_dict


def print_config_summary(config: Dict[str, Any]) -> None:
    """
    Print a human-readable summary of configuration.
    
    Args:
        config: Configuration dictionary
    """
    print("=" * 80)
    print("TRAINING CONFIGURATION SUMMARY")
    print("=" * 80)
    
    if 'timestamp' in config:
        print(f"Timestamp: {config['timestamp']}")
    
    if 'git' in config:
        git_info = config['git']
        if git_info.get('commit_hash'):
            print(f"Git Commit: {git_info['commit_hash']}")
        if git_info.get('branch'):
            print(f"Git Branch: {git_info['branch']}")
    
    if 'python_version' in config:
        print(f"Python Version: {config['python_version']}")
    
    print("\nModel Configuration:")
    if 'model_config' in config:
        for key, value in config['model_config'].items():
            print(f"  {key}: {value}")
    
    print("\nEnvironment Configuration:")
    if 'env_config' in config:
        for key, value in config['env_config'].items():
            print(f"  {key}: {value}")
    
    print("\nTraining Configuration:")
    if 'training_config' in config:
        for key, value in config['training_config'].items():
            print(f"  {key}: {value}")
    
    print("=" * 80)

