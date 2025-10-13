# Speed Configuration Comparison

## Current Performance (test_config.yaml)

**Settings:**
- Timewarp: 10x
- Action interval: 3s
- Episode length: 600s
- n_steps: 128

**Performance:**
- Steps/sec: ~2.9
- Real time per step: ~0.35s
- Progress update: every ~44 seconds (128 steps)
- 100K steps: ~9.5 hours

---

## Speed Optimization Analysis

### Time Breakdown (Current):

```
Per Step = action_interval ÷ timewarp
         = 3 seconds ÷ 10
         = 0.3s game time

Plus overhead:
  - Browser latency: ~0.03s
  - State extraction: ~0.01s
  - Network inference: ~0.01s
  Total: ~0.35s real time per step
```

### Maximum Theoretical Speed:

```
Best case with timewarp=50, action_interval=1:
  Game time: 1s ÷ 50 = 0.02s
  + overhead: ~0.05s
  = ~0.07s per step
  = ~14 steps/sec (5x faster than current!)
```

---

## Configuration Presets

### 1. CONSERVATIVE (original training_config.yaml)
```yaml
timewarp: 5
action_interval: 5
episode_length: 3600
n_steps: 2048
```
- **Speed**: ~1 step/sec
- **Stability**: ★★★★★ (very stable)
- **Best for**: Final training runs
- **100K steps**: ~28 hours

### 2. BALANCED (test_config.yaml) ← YOU ARE HERE
```yaml
timewarp: 10
action_interval: 3
episode_length: 600
n_steps: 128
```
- **Speed**: ~2.9 steps/sec
- **Stability**: ★★★★☆ (stable)
- **Best for**: Regular development/testing
- **100K steps**: ~9.5 hours

### 3. AGGRESSIVE (ultra_fast_config.yaml) ← RECOMMENDED NEXT
```yaml
timewarp: 30
action_interval: 2
episode_length: 300
n_steps: 64
```
- **Speed**: ~6-8 steps/sec (estimated)
- **Stability**: ★★★☆☆ (occasional crashes expected)
- **Best for**: Rapid prototyping
- **100K steps**: ~3.5 hours

### 4. EXTREME (not recommended for production)
```yaml
timewarp: 50
action_interval: 1
episode_length: 180
n_steps: 32
```
- **Speed**: ~10-14 steps/sec (theoretical)
- **Stability**: ★★☆☆☆ (frequent crashes likely)
- **Best for**: Quick experiments only
- **100K steps**: ~2 hours (if stable)

---

## Recommended Speedup Strategy

### Phase 1: Test Ultra-Fast Config (NOW)
```bash
# Stop current training (Ctrl+C)
python train.py --config config/ultra_fast_config.yaml
```

**Expected Results:**
- ✅ 2-3x faster than current
- ⚠️ May crash occasionally (restart if needed)
- 📊 Progress updates every ~10-15 seconds

### Phase 2: If Stable, Push Further

Create `ludicrous_config.yaml` with:
```yaml
timewarp: 40-50
action_interval: 1
n_steps: 32
```

### Phase 3: Monitor for Issues

**Signs it's too fast:**
- 🔴 Browser crashes frequently (>1 per hour)
- 🔴 JavaScript errors in console
- 🔴 Game state extraction failures
- 🔴 Training loss = NaN

**Signs it's good:**
- ✅ Smooth progress
- ✅ No errors
- ✅ Metrics look reasonable

---

## Known Limits

### Browser/Game Limits:
- **timewarp > 50**: Game physics may break
- **action_interval < 1**: Commands may not register
- **Too fast**: Browser can't keep up with rendering

### Diminishing Returns:
Beyond ~10-15 steps/sec, overhead dominates:
- JavaScript execution time
- IPC between Python/Browser
- Network latency (even local)

### Sweet Spot:
For openScope, the practical maximum is probably:
- **~8-12 steps/sec** (with timewarp 30-40)
- Beyond that, instability outweighs speed gains

---

## Quick Test Command

Test how fast you can go:

```bash
cd /home/jmzlx/Projects/openscope/rl_training

# Test ultra-fast config
time python -c "
from environment.openscope_env import OpenScopeEnv
import time

env = OpenScopeEnv(timewarp=30, action_interval=2, headless=True)
env.reset()

start = time.time()
for i in range(20):
    action = {
        'aircraft_id': 0,
        'command_type': 0,
        'altitude': 0,
        'heading': 0,
        'speed': 0
    }
    env.step(action)

elapsed = time.time() - start
print(f'20 steps in {elapsed:.1f}s = {20/elapsed:.2f} steps/sec')
env.close()
"
```

This will show you the actual speed with different settings!

---

## Bottom Line

**Current**: 2.9 steps/sec → 9.5 hours for 100K steps

**With ultra_fast_config**: ~6-8 steps/sec → ~3-4 hours for 100K steps

**Theoretical max**: ~10-14 steps/sec → ~2 hours for 100K steps

**Recommendation**: Try `ultra_fast_config.yaml` now! 🚀

