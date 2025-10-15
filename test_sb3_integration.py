"""
Test script to verify SB3 integration works correctly
Run this before starting full training to catch any issues
"""

import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from environment.wrappers import make_openscope_env
from models.sb3_policy import ATCTransformerPolicy

print("=" * 60)
print("SB3 Integration Test")
print("=" * 60)

# Load config
print("\n[1/6] Loading configuration...")
with open("config/training_config.yaml") as f:
    config = yaml.safe_load(f)
print("✓ Config loaded")

# Create environment
print("\n[2/6] Creating environment with wrappers...")


def make_test_env():
    env = make_openscope_env(
        game_url=config["env"]["game_url"],
        airport=config["env"]["airport"],
        timewarp=10,  # Fast for testing
        max_aircraft=5,  # Fewer for testing
        episode_length=600,
        action_interval=3,
        headless=True,
        config=config,
        normalize_obs=True,
        normalize_reward=True,
        gamma=config["ppo"]["gamma"],
    )
    return env


env = DummyVecEnv([make_test_env])
print("✓ Environment created with wrappers")

# Create model
print("\n[3/6] Creating PPO model with custom policy...")
policy_kwargs = dict(
    aircraft_feature_dim=config["network"]["aircraft_feature_dim"],
    global_feature_dim=config["network"]["global_feature_dim"],
    hidden_dim=config["network"]["hidden_dim"],
    num_heads=config["network"]["num_attention_heads"],
    num_layers=config["network"]["num_transformer_layers"],
    max_aircraft=config["network"]["max_aircraft_slots"],
)

model = PPO(
    ATCTransformerPolicy,
    env,
    learning_rate=3e-4,
    n_steps=128,  # Small for testing
    batch_size=64,
    gamma=0.99,
    policy_kwargs=policy_kwargs,
    verbose=0,
    device="cuda" if torch.cuda.is_available() else "cpu",
)

param_count = sum(p.numel() for p in model.policy.parameters())
print(f"✓ Model created with {param_count:,} parameters")

# Test forward pass
print("\n[4/6] Testing model forward pass...")
obs = env.reset()
action, _ = model.predict(obs, deterministic=True)
print("✓ Forward pass successful")
print(f"  Observation keys: {list(obs[0].keys())}")
print(f"  Action keys: {list(action[0].keys())}")

# Test training step
print("\n[5/6] Testing training for 10 steps...")
import time

start = time.time()
model.learn(total_timesteps=10, log_interval=None, progress_bar=False)
elapsed = time.time() - start
print(f"✓ Training successful ({elapsed:.1f}s)")

# Test save/load
print("\n[6/6] Testing save/load...")
import tempfile

with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as f:
    model.save(f.name)
    loaded_model = PPO.load(f.name, env=env, custom_objects={"policy_class": ATCTransformerPolicy})
    print("✓ Save/load successful")

# Cleanup
env.close()

print("\n" + "=" * 60)
print("✓ All tests passed! SB3 integration is working correctly.")
print("=" * 60)
print("\nNext steps:")
print("1. Install dependencies: pip install -r requirements.txt")
print("2. Install playwright: playwright install chromium")
print("3. Start game server: cd .. && npm run start")
print("4. Run training: python train_sb3.py --n-envs 4")
print()
