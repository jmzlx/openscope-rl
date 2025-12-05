#!/usr/bin/env python3
"""
Simple demo script for OpenScope environment visualization.

This script creates an environment and runs it with random actions,
displaying basic statistics and FPS.
"""

import argparse
import time
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from environment import PlaywrightEnv, create_default_config


def main():
    parser = argparse.ArgumentParser(description="Demo OpenScope environment")
    parser.add_argument("--airport", type=str, default="KLAS", help="Airport ICAO code")
    parser.add_argument("--max-aircraft", type=int, default=10, help="Max aircraft")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    parser.add_argument("--n-steps", type=int, default=1000, help="Number of steps to run")
    parser.add_argument("--timewarp", type=int, default=5, help="Timewarp speed")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("OpenScope Environment Demo")
    print("=" * 80)
    print(f"Airport: {args.airport}")
    print(f"Max Aircraft: {args.max_aircraft}")
    print(f"Headless: {args.headless}")
    print(f"Timewarp: {args.timewarp}")
    print(f"Steps: {args.n_steps}")
    print("=" * 80)
    
    # Create environment
    config = create_default_config(
        airport=args.airport,
        max_aircraft=args.max_aircraft,
        headless=args.headless,
        timewarp=args.timewarp,
    )
    
    print("\nCreating environment...")
    env = PlaywrightEnv(
        airport=config.airport,
        max_aircraft=config.max_aircraft,
        headless=config.browser_config.headless,
        timewarp=config.timewarp,
    )
    
    print("Resetting environment...")
    obs, info = env.reset()
    
    print("\nRunning demo...")
    print("Press Ctrl+C to stop early\n")
    
    start_time = time.time()
    total_reward = 0.0
    episode_count = 0
    
    try:
        for step in range(args.n_steps):
            # Sample random action
            action = env.action_space.sample()
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            
            # Print progress every 100 steps
            if (step + 1) % 100 == 0:
                elapsed = time.time() - start_time
                fps = (step + 1) / elapsed
                avg_reward = total_reward / (step + 1)
                print(f"Step {step + 1}/{args.n_steps} | "
                      f"FPS: {fps:.2f} | "
                      f"Avg Reward: {avg_reward:.2f} | "
                      f"Aircraft: {info.get('num_aircraft', 'N/A')}")
            
            # Handle episode end
            if terminated or truncated:
                episode_count += 1
                print(f"\nEpisode {episode_count} ended (reward: {total_reward:.2f})")
                obs, info = env.reset()
                total_reward = 0.0
        
        elapsed = time.time() - start_time
        fps = args.n_steps / elapsed
        
        print("\n" + "=" * 80)
        print("Demo Complete")
        print("=" * 80)
        print(f"Total Steps: {args.n_steps}")
        print(f"Total Time: {elapsed:.2f}s")
        print(f"FPS: {fps:.2f}")
        print(f"Episodes: {episode_count}")
        print("=" * 80)
        
    except KeyboardInterrupt:
        elapsed = time.time() - start_time
        fps = step / elapsed if step > 0 else 0
        
        print("\n\nDemo interrupted by user")
        print(f"Steps completed: {step}")
        print(f"FPS: {fps:.2f}")
    
    finally:
        print("\nClosing environment...")
        env.close()
        print("Done!")


if __name__ == "__main__":
    main()

