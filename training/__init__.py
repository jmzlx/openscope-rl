"""
Training module for behavioral cloning and hybrid BC+RL.

This module provides tools for:
1. Generating expert demonstrations using rule-based controllers
2. Pre-training policies using behavioral cloning
3. Fine-tuning with hybrid BC+RL approach
"""

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


__all__ = [
    # Expert demonstrations
    'RuleBasedExpert',
    'Demonstration',
    'generate_demonstrations',
    'load_demonstrations',
    'print_demonstration_stats',

    # Behavioral cloning
    'BehavioralCloningTrainer',
    'DemonstrationDataset',
    'train_bc_model',

    # Hybrid BC+RL
    'BCRLHybridTrainer',
    'BCRLBuffer',
]
