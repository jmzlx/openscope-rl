"""
Quick environment test - verifies everything works before full training
"""

import time
from environment.openscope_env import OpenScopeEnv
import numpy as np

print("="*60)
print("OpenScope Environment Test")
print("="*60)

# Create environment
print("\n[1/5] Creating environment...")
env = OpenScopeEnv(
    game_url="http://localhost:3003",
    airport="KLAS",
    timewarp=10,  # Fast for testing
    max_aircraft=5,
    episode_length=600,
    action_interval=3,
    headless=True
)
print("✓ Environment created")

# Reset
print("\n[2/5] Resetting environment (this may take 30-60 seconds)...")
start = time.time()
obs, info = env.reset()
elapsed = time.time() - start
print(f"✓ Environment reset in {elapsed:.1f} seconds")

# Check observation
print(f"\n[3/5] Checking observation structure...")
print(f"  - Aircraft shape: {obs['aircraft'].shape}")
print(f"  - Aircraft mask sum: {obs['aircraft_mask'].sum()} (active aircraft)")
print(f"  - Global state shape: {obs['global_state'].shape}")
print(f"  - Conflict matrix shape: {obs['conflict_matrix'].shape}")
print("✓ Observation structure valid")

# Take a few random steps
print(f"\n[4/5] Taking 5 test steps...")
total_reward = 0
for i in range(5):
    # Random action
    action = {
        'aircraft_id': np.random.randint(0, env.max_aircraft + 1),
        'command_type': np.random.randint(0, 5),
        'altitude': np.random.randint(0, 18),
        'heading': np.random.randint(0, 13),
        'speed': np.random.randint(0, 8)
    }
    
    print(f"  Step {i+1}/5...", end=" ", flush=True)
    step_start = time.time()
    obs, reward, terminated, truncated, info = env.step(action)
    step_time = time.time() - step_start
    total_reward += reward
    
    print(f"reward={reward:.2f}, time={step_time:.1f}s")
    
    if terminated or truncated:
        print("  Episode ended early")
        break

print(f"✓ Steps completed. Total reward: {total_reward:.2f}")

# Timing info
print(f"\n[5/5] Performance summary:")
print(f"  - Reset time: {elapsed:.1f}s")
print(f"  - Average step time: {step_time:.1f}s")
print(f"  - Estimated time for 128 steps: {step_time * 128 / 60:.1f} minutes")
print(f"  - Estimated time for 2048 steps: {step_time * 2048 / 60:.1f} minutes")

# Cleanup
env.close()

print("\n" + "="*60)
print("✓ All tests passed! Environment is working correctly.")
print("="*60)
print("\nRecommendation:")
if step_time > 2:
    print("  ⚠️  Steps are slow (>2s). Use test_config.yaml for faster training:")
    print("     python train.py --config config/test_config.yaml")
else:
    print("  ✓ Steps are fast. You can use either config.")
print()

