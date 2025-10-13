"""
Diagnose why training is crashing
"""

import sys
import time
from environment.openscope_env import OpenScopeEnv

print("="*60)
print("Environment Diagnostic Test")
print("="*60)

configs_to_test = [
    ("Conservative", {"timewarp": 5, "action_interval": 5, "episode_length": 600}),
    ("Moderate", {"timewarp": 10, "action_interval": 3, "episode_length": 600}),
    ("Aggressive", {"timewarp": 20, "action_interval": 2, "episode_length": 300}),
    ("Ultra", {"timewarp": 30, "action_interval": 2, "episode_length": 300}),
]

for name, config in configs_to_test:
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"  Timewarp: {config['timewarp']}x")
    print(f"  Action interval: {config['action_interval']}s")
    print(f"  Episode length: {config['episode_length']}s")
    print("="*60)
    
    try:
        # Create environment
        print("  [1/4] Creating environment...", end=" ", flush=True)
        env = OpenScopeEnv(
            timewarp=config['timewarp'],
            action_interval=config['action_interval'],
            episode_length=config['episode_length'],
            max_aircraft=3,
            headless=True
        )
        print("✓")
        
        # Reset
        print("  [2/4] Resetting environment...", end=" ", flush=True)
        start = time.time()
        obs, info = env.reset()
        reset_time = time.time() - start
        print(f"✓ ({reset_time:.1f}s)")
        
        # Take steps
        print("  [3/4] Taking 10 steps...", end=" ", flush=True)
        start = time.time()
        total_reward = 0
        for i in range(10):
            action = {
                'aircraft_id': 0,
                'command_type': 0,
                'altitude': 5,
                'heading': 6,
                'speed': 4
            }
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                print(f"\n    Episode ended at step {i+1}")
                break
        
        elapsed = time.time() - start
        steps_per_sec = 10 / elapsed
        print(f"✓ ({elapsed:.1f}s, {steps_per_sec:.2f} steps/sec)")
        
        # Test episode completion
        print("  [4/4] Testing episode completion...", end=" ", flush=True)
        obs, info = env.reset()
        
        # Try to complete a short episode
        done = False
        step_count = 0
        max_steps = 100  # Should complete in less than this
        start = time.time()
        
        while not done and step_count < max_steps:
            action = {
                'aircraft_id': 0,
                'command_type': 0,
                'altitude': 0,
                'heading': 0,
                'speed': 0
            }
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step_count += 1
        
        elapsed = time.time() - start
        
        if done:
            print(f"✓ Episode completed in {step_count} steps ({elapsed:.1f}s)")
        else:
            print(f"✗ Episode did NOT complete after {step_count} steps ({elapsed:.1f}s)")
            print(f"    WARNING: Episodes may be too long or not terminating!")
        
        env.close()
        
        print(f"\n  RESULT: {'✓ STABLE' if done else '✗ PROBLEMATIC'}")
        
        if not done:
            print(f"  → Config is too aggressive, episodes don't complete")
            break
            
    except KeyboardInterrupt:
        print("\n\n  Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"✗ FAILED")
        print(f"  Error: {e}")
        print(f"  → Config is TOO aggressive, causes crashes")
        break

print("\n" + "="*60)
print("Diagnostic Complete")
print("="*60)

