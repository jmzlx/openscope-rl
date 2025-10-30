"""
Training modules for OpenScope RL.

This package contains training scripts and utilities for various RL approaches:
- PPO: Baseline PPO with action masking
- Hierarchical RL: Two-level hierarchical policies
- Behavioral Cloning: Expert demonstrations and BC training
- Multi-Agent: MAPPO for multi-agent coordination
- Decision Transformer: Offline RL with transformer architecture
- Trajectory Transformer: Model-based planning with transformers
- Cosmos: Fine-tune NVIDIA Cosmos on OpenScope data
"""

from .cosmos_finetuner import CosmosFineTuner, CosmosTrainingConfig
from .cosmos_rl_trainer import CosmosRLTrainer

# PPO trainer
try:
    from .ppo_trainer import train as train_ppo, make_env, create_vec_env, create_ppo_model
except ImportError:
    pass

# Hierarchical RL trainer
try:
    from .hierarchical_trainer import HierarchicalPPOTrainer, HierarchicalPPOConfig
except ImportError:
    pass

# Behavioral cloning
try:
    from .rule_based_expert import (
        RuleBasedExpert,
        Demonstration,
        generate_demonstrations,
        load_demonstrations,
        print_demonstration_stats,
    )
    from .behavioral_cloning import (
        BehavioralCloningTrainer,
        DemonstrationDataset,
        train_bc_model,
    )
    from .bc_rl_hybrid import (
        BCRLHybridTrainer,
        BCRLBuffer,
    )
except ImportError:
    pass

# Multi-agent
try:
    from .mappo_trainer import MAPPOTrainer, MAPPOConfig
except ImportError:
    pass

# Decision Transformer
try:
    from .dt_trainer import DecisionTransformerTrainer, TrainingConfig as DTTrainingConfig
except ImportError:
    pass

# Trajectory Transformer
try:
    from .tt_trainer import TrajectoryTransformerTrainer, TrainingConfig as TTTrainingConfig, create_trainer
    from .tt_planner import BeamSearchPlanner, PlannerConfig
except ImportError:
    pass

__all__ = [
    # Cosmos
    "CosmosFineTuner",
    "CosmosTrainingConfig",
    "CosmosRLTrainer",
    
    # PPO
    "train_ppo",
    "make_env",
    "create_vec_env",
    "create_ppo_model",
    
    # Hierarchical RL
    "HierarchicalPPOTrainer",
    "HierarchicalPPOConfig",
    
    # Behavioral cloning
    "RuleBasedExpert",
    "Demonstration",
    "generate_demonstrations",
    "load_demonstrations",
    "print_demonstration_stats",
    "BehavioralCloningTrainer",
    "DemonstrationDataset",
    "train_bc_model",
    "BCRLHybridTrainer",
    "BCRLBuffer",
    
    # Multi-agent
    "MAPPOTrainer",
    "MAPPOConfig",
    
    # Decision Transformer
    "DecisionTransformerTrainer",
    "DTTrainingConfig",
    
    # Trajectory Transformer
    "TrajectoryTransformerTrainer",
    "TTTrainingConfig",
    "create_trainer",
    "BeamSearchPlanner",
    "PlannerConfig",
]
