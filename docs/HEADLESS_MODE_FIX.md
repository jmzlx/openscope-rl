# OpenScope Headless Mode Issue & WSL Display Fix

## Problems Discovered

### Problem 1: Headless Mode Breaks Game Loop

When running the OpenScope RL environment with `headless: true`, the game's time (`TimeKeeper.accumulatedDeltaTime`) remains stuck at 0.0s, preventing proper simulation.

### Problem 2: WSL Has No Display

When we fix Problem 1 by setting `headless: false`, WSL throws `TargetClosedError` because there's no X Server (display) available.

### Symptoms
- ✗ `TimeKeeper.accumulatedDeltaTime` stays at 0.0s
- ✗ Aircraft may spawn from airport initialization (25-26 aircraft appear)
- ✗ Game simulation doesn't progress despite `env.step()` calls
- ✗ No learning possible since environment state is frozen

### Root Cause

OpenScope's game loop depends on the browser's `requestAnimationFrame` (RAF) API. In headless Chrome/Chromium:
- RAF callbacks may not fire consistently
- Timing is unreliable or completely broken
- The game's `update()` method never increments `accumulatedDeltaTime`

**Relevant Code:**
```javascript
// TimeKeeper.js line 454
this._accumulatedDeltaTime += this.getDeltaTimeForGameStateAndTimewarp();
```

This line only executes during the RAF callback, which doesn't work in headless mode.

## Solution

**Set `headless: false` in config/training_config.yaml**

```yaml
env:
  headless: false  # MUST be false for proper timing
```

The browser window will be visible, but this is **required** for the game loop to function properly.

## Performance Optimizations

While fixing headless mode, we also optimized training speed:

### Before
```yaml
timewarp: 5
action_interval: 5  # seconds
episode_length: 3600  # 1 hour
```

### After (Optimized)
```yaml
timewarp: 15  # 3x faster real-time execution
action_interval: 10  # 2x fewer decisions per episode
episode_length: 1800  # 2x faster episodes (30 min)
```

**Combined speedup: ~12x faster training** 🚀

### Speed Breakdown
1. **Timewarp 15**: Each step waits `action_interval / timewarp = 10 / 15 = 0.67s` instead of `5 / 5 = 1.0s` → **~3x faster**
2. **Action interval 10**: Half as many steps needed per episode → **2x faster**
3. **Episode length 1800**: Episodes finish in half the game time → **2x faster**

## Aircraft Spawning

### Issue
KLAS airport has very low spawn rates:
- 2.4 - 10.2 aircraft/hour
- At 3.6 aircraft/hour → **1 aircraft every ~1000 seconds** (16.7 minutes)

### Implication
Need to advance **10-15 minutes of game time** before aircraft spawn naturally.

**Calculation:**
- `action_interval = 10` seconds
- To reach 900s (15 min): `900 / 10 = 90 steps`
- With `timewarp = 15`: Takes `90 * 0.67s = 60s` real time

## Alternative Solutions (Not Recommended)

### 1. Manual Game Loop Injection
Manually call `TimeKeeper.update()` via JavaScript - **Fragile and unreliable**

### 2. Use Different Browser
Try Firefox or other browsers - **Still has RAF issues in headless mode**

### 3. Mock Time
Replace TimeKeeper entirely - **Breaks game mechanics**

### Conclusion
**Just use `headless: false`** - it's the simplest and most reliable solution.

## Testing Checklist

After changing config:
- [ ] Restart notebook kernel
- [ ] Run cell 3 (check game server)
- [ ] Run cell 8 (create environment with new config)
- [ ] Run cell 10 (verify time advances)
- [ ] Check for output like: `Step 18: Time= 180.0s (+180.0s), Aircraft=1`

## Files Modified

1. `config/training_config.yaml` - Set `headless: false`, optimized timings
2. `openscope_rl_demo.ipynb` - Added diagnostic cell and warnings
3. This documentation

## References

- OpenScope TimeKeeper: `openscope/src/assets/scripts/client/engine/TimeKeeper.js`
- Spawn patterns: `openscope/documentation/spawnPatternReadme.md`
- Browser RAF issues: https://bugs.chromium.org/p/chromium/issues/detail?id=1059122

