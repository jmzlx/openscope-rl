# Baseline PPO with Action Masking - Implementation Guide

**Worktree**: `.trees/01-baseline-ppo/`
**Branch**: `experiment/01-baseline-ppo`
**Priority**: ⭐⭐⭐ HIGH (Start here!)

## Objective

Establish baseline performance using PPO with action masking. This serves as the benchmark for all other approaches.

## Why Action Masking?

Action masking prevents the agent from selecting invalid actions, dramatically improving sample efficiency:
- Don't waste time learning "don't select non-existent aircraft"
- Don't waste time learning "ILS requires a target runway"
- Focus learning on meaningful ATC decisions

**Expected improvement**: 2-3x sample efficiency vs vanilla PPO

## Implementation Steps

### Step 1: Implement Action Masking (2-3 hours)

Create `environment/action_masking.py`:

```python
"""Action masking for OpenScope RL environment."""

import numpy as np
from typing import Dict, List, Any

def get_action_mask(obs: Dict[str, np.ndarray],
                   aircraft_data: List[Dict[str, Any]],
                   max_aircraft: int = 10) -> Dict[str, np.ndarray]:
    """
    Create action masks for current observation.

    Args:
        obs: Environment observation
        aircraft_data: List of aircraft state dicts
        max_aircraft: Maximum number of aircraft

    Returns:
        Dictionary of boolean masks for each action component
    """
    num_aircraft = len(aircraft_data)

    # Aircraft ID mask: Only valid aircraft + "no action" option
    aircraft_mask = np.zeros(max_aircraft + 1, dtype=bool)
    aircraft_mask[:num_aircraft + 1] = True  # Valid aircraft + no-action

    # Command type mask (initially all valid)
    command_mask = np.ones(5, dtype=bool)

    # Check if any aircraft lacks target runway (disable ILS globally if so)
    has_runway = all(ac.get('targetRunway') for ac in aircraft_data)
    if not has_runway:
        command_mask[3] = False  # ILS command index

    # Altitude, heading, speed masks (all valid by default)
    altitude_mask = np.ones(18, dtype=bool)
    heading_mask = np.ones(13, dtype=bool)
    speed_mask = np.ones(8, dtype=bool)

    return {
        'aircraft_id': aircraft_mask,
        'command_type': command_mask,
        'altitude': altitude_mask,
        'heading': heading_mask,
        'speed': speed_mask,
    }


class ActionMaskWrapper(gym.Wrapper):
    """Wrapper that adds action masks to observations."""

    def __init__(self, env):
        super().__init__(env)
        # Extend observation space to include masks
        self.observation_space = spaces.Dict({
            **env.observation_space.spaces,
            'action_mask': spaces.Box(0, 1, shape=(self._compute_mask_size(),), dtype=bool)
        })

    def _compute_mask_size(self):
        # Flatten all mask components
        return (self.max_aircraft + 1) + 5 + 18 + 13 + 8

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self._add_mask(obs, info)
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = self._add_mask(obs, info)
        return obs, info

    def _add_mask(self, obs, info):
        """Add action mask to observation."""
        aircraft_data = info.get('raw_state', {}).get('aircraft', [])
        masks = get_action_mask(obs, aircraft_data, self.max_aircraft)

        # Flatten masks into single array
        flat_mask = np.concatenate([
            masks['aircraft_id'],
            masks['command_type'],
            masks['altitude'],
            masks['heading'],
            masks['speed'],
        ])

        obs['action_mask'] = flat_mask
        return obs
```

### Step 2: Create Training Script (3-4 hours)

Create `training/ppo_trainer.py`:

```python
"""PPO training with action masking for OpenScope."""

import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import wandb
from wandb.integration.sb3 import WandbCallback

from environment import PlaywrightEnv, create_default_config
from environment.action_masking import ActionMaskWrapper


def make_env(rank, config):
    """Factory function for creating environments."""
    def _init():
        env = PlaywrightEnv(**config.__dict__)
        env = ActionMaskWrapper(env)  # Add action masking
        return env
    return _init


def train_ppo_baseline(
    total_timesteps=500_000,
    num_envs=8,
    wandb_project="openscope-rl-baseline",
    save_dir="./models",
):
    """
    Train PPO with action masking on OpenScope.

    Args:
        total_timesteps: Total training steps
        num_envs: Number of parallel environments
        wandb_project: Weights & Biases project name
        save_dir: Directory to save checkpoints
    """
    # Initialize WandB
    wandb.init(
        project=wandb_project,
        entity="jmzlx.ai",
        name="baseline-ppo-action-masking",
        config={
            "algorithm": "PPO",
            "action_masking": True,
            "num_envs": num_envs,
            "total_timesteps": total_timesteps,
        },
        sync_tensorboard=True,
    )

    # Create environment config
    config = create_default_config(
        max_aircraft=10,
        episode_length=600,
        headless=True,
        airport="KLAS",
        timewarp=5
    )

    # Create vectorized environment
    env = SubprocVecEnv([make_env(i, config) for i in range(num_envs)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    # Create evaluation environment
    eval_env = SubprocVecEnv([make_env(0, config)])
    eval_env = VecNormalize(eval_env, training=False, norm_reward=False)

    # Create PPO model
    model = PPO(
        "MultiInputPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=f"runs/{wandb.run.id}",
    )

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=save_dir,
        name_prefix="ppo_baseline",
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=save_dir,
        eval_freq=5_000,
        deterministic=True,
        render=False,
    )

    wandb_callback = WandbCallback(
        gradient_save_freq=100,
        model_save_path=f"{save_dir}/wandb",
        verbose=2,
    )

    # Train
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback, wandb_callback],
    )

    # Save final model
    model.save(f"{save_dir}/ppo_baseline_final")
    env.save(f"{save_dir}/vecnorm_stats.pkl")

    wandb.finish()

    return model


if __name__ == "__main__":
    train_ppo_baseline()
```

### Step 3: Create Demo Notebook (2-3 hours)

Create `notebooks/01_baseline_ppo_demo.ipynb` with:

1. **Setup & Imports**
2. **Demonstrate Action Masking** (small example)
3. **Quick Training** (10k steps for demo)
4. **Evaluation** (visualize learned behavior)
5. **Comparison** (vs random policy)

### Step 4: Curriculum Learning (Optional, 2-3 hours)

Add progressive difficulty to training:

```python
class CurriculumCallback:
    """Progressively increase difficulty during training."""

    def __init__(self):
        self.current_stage = 0
        self.stages = [
            {"max_aircraft": 3, "threshold": 0.8},  # 80% success → advance
            {"max_aircraft": 6, "threshold": 0.75},
            {"max_aircraft": 10, "threshold": 0.70},
        ]
```

## Expected Results

After 500k timesteps (~6-8 hours training on your hardware):
- **Success rate**: 60-75%
- **Violations**: <5 per episode
- **Sample efficiency**: 2-3x better than vanilla PPO (due to action masking)

## Testing Checklist

- [ ] Action masking prevents selecting non-existent aircraft
- [ ] Action masking prevents ILS without runway
- [ ] Training converges (loss decreases, reward increases)
- [ ] WandB logs successfully
- [ ] Checkpoints save correctly
- [ ] Evaluation shows improvement over random policy
- [ ] Notebook runs end-to-end in <30 minutes

## Commit Your Work

```bash
cd .trees/01-baseline-ppo
git add .
git commit -m "Implement baseline PPO with action masking"
git push origin experiment/01-baseline-ppo
```

## Next Steps

Once this baseline is working:
1. Compare to Decision Transformer (worktree 05)
2. Compare to Cosmos approach (worktree 07)
3. Consider merging successful approaches to main branch

---

**Questions?** Check the main repo's `experiments/` folder for shared utilities, or review the POC implementations in `poc/atc_rl/` for working examples.
