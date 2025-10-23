"""
Test that the environment works inside Jupyter (with asyncio event loop running)
"""

import sys
import asyncio
sys.path.insert(0, '/home/jmzlx/Projects/atc/openscope-rl/poc')

print("Testing Jupyter Compatibility (with asyncio event loop)")
print("=" * 70)

# Simulate Jupyter environment by running in asyncio
async def test_in_jupyter():
    """Simulates Jupyter's asyncio environment"""
    print("\n✅ Running inside asyncio event loop (like Jupyter)")

    from environment.openscope_env import OpenScopeEnv

    config = {
        'env': {'airport': 'KLAS', 'timewarp': 15},
        'rewards': {
            'separation_loss': -200,
            'conflict_warning': -2.0,
            'timestep_penalty': -0.01,
            'safe_separation_bonus': 0.05,
        }
    }

    print("\n1. Creating environment inside event loop...")
    env = OpenScopeEnv(
        game_url="http://localhost:3003",
        airport="KLAS",
        timewarp=15,
        max_aircraft=20,
        headless=True,
        config=config,
    )
    print("✅ Environment created (nest_asyncio working!)")

    print("\n2. Testing reset...")
    obs, info = env.reset()
    print(f"✅ Reset successful: {len(info['raw_state']['aircraft'])} aircraft")

    print("\n3. Testing step...")
    action = {'aircraft_id': 20, 'command_type': 0, 'heading': 0, 'altitude': 0, 'speed': 0}
    obs, reward, done, trunc, info = env.step(action)
    print(f"✅ Step successful: Time={info['raw_state']['time']:.1f}s")

    print("\n4. Closing...")
    env.close()
    print("✅ Environment closed")

# Run in asyncio event loop (like Jupyter does)
asyncio.run(test_in_jupyter())

print("\n" + "=" * 70)
print("✅ Jupyter compatibility verified!")
print("=" * 70)
print("\nThe environment works correctly inside Jupyter notebooks:")
print("  ✅ nest_asyncio enables sync Playwright in asyncio loops")
print("  ✅ No 'already running event loop' errors")
print("  ✅ Ready for use in Jupyter notebooks!")
