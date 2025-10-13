"""
Benchmark different timewarp settings to find optimal speed
"""

from environment.openscope_env import OpenScopeEnv
import time
import sys

def test_speed(timewarp, action_interval, num_steps=20):
    """Test environment speed with given settings"""
    print(f"\nTesting: timewarp={timewarp}, action_interval={action_interval}")
    print("-" * 60)
    
    try:
        env = OpenScopeEnv(
            timewarp=timewarp,
            action_interval=action_interval,
            headless=True,
            max_aircraft=3
        )
        
        print("  Resetting environment...", end=" ", flush=True)
        reset_start = time.time()
        env.reset()
        reset_time = time.time() - reset_start
        print(f"done ({reset_time:.1f}s)")
        
        print(f"  Running {num_steps} steps...", end=" ", flush=True)
        start = time.time()
        
        for i in range(num_steps):
            action = {
                'aircraft_id': 0,
                'command_type': 0,
                'altitude': 0,
                'heading': 0,
                'speed': 0
            }
            env.step(action)
        
        elapsed = time.time() - start
        steps_per_sec = num_steps / elapsed
        
        print(f"done ({elapsed:.1f}s)")
        print(f"\n  ✓ Speed: {steps_per_sec:.2f} steps/sec")
        print(f"  ✓ Time per step: {elapsed/num_steps:.3f}s")
        print(f"  ✓ Est. 100K steps: {100000/steps_per_sec/3600:.1f} hours")
        
        env.close()
        return steps_per_sec, True
        
    except Exception as e:
        print(f"\n  ✗ Failed: {e}")
        return 0, False

print("="*60)
print("OpenScope Speed Benchmark")
print("="*60)
print("\nThis will test different timewarp settings.")
print("Testing 20 steps each (takes ~1-2 min total)")
print()

configs = [
    ("Current (test)", 10, 3),
    ("Aggressive", 30, 2),
    ("Extreme", 50, 1),
]

results = []

for name, timewarp, action_interval in configs:
    print(f"\n{'='*60}")
    print(f"{name.upper()}")
    speed, success = test_speed(timewarp, action_interval)
    results.append((name, timewarp, action_interval, speed, success))
    
    if not success:
        print(f"\n⚠️  {name} config failed - too aggressive!")
        break

print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)

print("\n{:<20} {:>10} {:>10} {:>12} {:>10}".format(
    "Config", "Timewarp", "Interval", "Steps/sec", "Status"
))
print("-" * 60)

for name, tw, interval, speed, success in results:
    status = "✓ OK" if success else "✗ FAIL"
    if success:
        print(f"{name:<20} {tw:>10}x {interval:>9}s {speed:>11.2f} {status:>10}")
    else:
        print(f"{name:<20} {tw:>10}x {interval:>9}s {'N/A':>11} {status:>10}")

if results[-1][4]:  # Last test succeeded
    best = max(results, key=lambda x: x[3] if x[4] else 0)
    print("\n" + "="*60)
    print(f"RECOMMENDATION: Use {best[0]} config")
    print(f"  Speed: {best[3]:.2f} steps/sec")
    print(f"  Est. 100K steps: {100000/best[3]/3600:.1f} hours")
    print("="*60)

