# OpenScope RL Environment - Setup Guide

## Critical Requirements

### 1. Run on WSL with Xvfb

OpenScope requires a **non-headless browser** for the game loop to work (requestAnimationFrame). On WSL, use xvfb:

```bash
# Install xvfb (one time)
sudo apt-get update && sudo apt-get install -y xvfb

# Start xvfb (every session)
Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &

# Set DISPLAY environment variable
export DISPLAY=:99
```

Or use the helper script:
```bash
./start_xvfb.sh
```

### 2. Configuration

In `config/training_config.yaml`:

```yaml
env:
  headless: false  # MUST be false!
  timewarp: 15     # Speeds up training
  action_interval: 10  # Seconds between decisions
  episode_length: 1800  # 30 min episodes
```

## How It Works

### Game Loop
- OpenScope's JavaScript bundle auto-executes when page loads
- Game loop runs via `requestAnimationFrame` (requires non-headless browser)
- Aircraft positions update in real-time

### Time Tracking
- `window.TimeKeeper` is not exposed as a global variable
- **Solution**: Python tracks time by counting steps
- `simulated_time += action_interval` on each `env.step()`

### What Works
✅ Aircraft spawn and move  
✅ Commands execute (altitude, heading, speed)  
✅ Conflicts detected  
✅ Score tracking  
✅ Full game simulation  

### What Doesn't Work
❌ Direct access to `window.TimeKeeper`  
❌ Direct access to `window.App`  
✅ **Workaround**: Track time in Python

## Quick Start

```python
import os
os.environ['DISPLAY'] = ':99'  # For xvfb

from environment.openscope_env import OpenScopeEnv

env = OpenScopeEnv(
    game_url="http://localhost:3003",
    airport="KLAS",
    timewarp=15,
    max_aircraft=20,
    episode_length=1800,
    action_interval=10,
    headless=False,
    config=config
)

obs, info = env.reset()

for _ in range(100):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    if done or truncated:
        obs, info = env.reset()

env.close()
```

## Performance

With optimizations:
- **~12x faster** than original config
- Timewarp 15: 3x speedup
- Action interval 10: 2x fewer steps
- Episode length 1800: 2x faster episodes

## Troubleshooting

### Time stuck at 0
- ✅ **Expected!** Time is tracked in Python, not JavaScript
- Check: Are aircraft positions changing? If yes, it's working!

### TargetClosedError: No XServer
- Solution: Run with xvfb (see above)

### JavaScript bundle doesn't load
- Check: Is game server running? `http://localhost:3003`
- The bundle auto-loads, no manual intervention needed

### Planes flickering
- Don't manually `eval()` the bundle - it auto-executes
- This was a debugging step, not needed in production

## Files

- `environment/openscope_env.py` - Main Gymnasium environment
- `config/training_config.yaml` - Configuration
- `start_xvfb.sh` - Helper to start virtual display
- `WSL_DISPLAY_FIX.md` - Detailed xvfb setup
- `HEADLESS_MODE_FIX.md` - Technical details on the issue

## Training

See `train_sb3.py` for full training script with:
- PPO algorithm
- Curriculum learning
- Checkpointing
- TensorBoard logging

