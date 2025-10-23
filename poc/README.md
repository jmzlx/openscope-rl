# OpenScope RL POC - Optimally Simplified

This directory contains the **simplified proof-of-concept** implementation for training RL agents on OpenScope.

## 🎯 What's New

**Optimized environment**: Reduced from **591 lines → 410 lines** (30.6% reduction)

**Key improvements:**
- ✅ Removed 127 lines of custom threading
- ✅ Added full type hints
- ✅ Cleaner sync Playwright usage
- ✅ No over-abstraction (no builders, no async wrappers)
- ✅ Same API - drop-in replacement

See [SIMPLIFICATION_SUMMARY.md](SIMPLIFICATION_SUMMARY.md) for details.

---

## Files Included

### Core Files
- **[openscope_rl_demo.ipynb](openscope_rl_demo.ipynb)** - Interactive demo notebook
- **[pyproject.toml](pyproject.toml)** - Python dependencies

### Environment Module
- **[environment/openscope_env.py](environment/openscope_env.py)** - **Optimized** Gymnasium environment (410 lines)
- `environment/__init__.py` - Module initialization

### Models Module
- **[models/networks.py](models/networks.py)** - Transformer architecture (ATCTransformerEncoder, ATCActorCritic)
- **[models/sb3_policy.py](models/sb3_policy.py)** - SB3 policy wrapper (ATCTransformerPolicy)
- `models/__init__.py` - Module initialization

### Tests
- **[test_env.py](test_env.py)** - Basic environment tests
- **[test_notebook_compat.py](test_notebook_compat.py)** - Notebook compatibility tests

### Documentation
- **[SIMPLIFICATION_SUMMARY.md](SIMPLIFICATION_SUMMARY.md)** - Detailed before/after comparison
- **[README.md](README.md)** - This file

---

## Prerequisites

1. **Python Dependencies**: Install from `pyproject.toml`
   ```bash
   pip install -e .
   # or with uv (faster)
   uv sync
   ```

2. **OpenScope Game Server**: Must be running on `http://localhost:3003`
   ```bash
   cd ../openscope && npm start
   ```

3. **Playwright Browser**: Install browser dependencies
   ```bash
   playwright install chromium
   # or
   uv run playwright install chromium
   ```

---

## Quick Start

### 1. Run Tests
```bash
# Basic functionality
uv run python poc/test_env.py

# Notebook compatibility
uv run python poc/test_notebook_compat.py
```

### 2. Use in Code
```python
from environment.openscope_env import OpenScopeEnv

env = OpenScopeEnv(
    game_url="http://localhost:3003",
    airport="KLAS",
    timewarp=15,
    headless=True,
    config={'rewards': {...}}
)

obs, info = env.reset()
# ... your training loop
env.close()
```

### 3. Run Notebook
```bash
jupyter notebook openscope_rl_demo.ipynb
```

---

## What Makes It Optimal

### Before (Original)
- 591 lines
- Custom threading with queues (127 lines)
- No type hints
- Magic numbers everywhere

### After (Optimized)
- **410 lines** (-30.6%)
- **Sync Playwright** (no threading needed)
- **Full type hints** (better IDE support)
- **Inline operations** with comments

**Key principle**: Remove complexity, don't add abstraction.

---

## Architecture

```
environment/openscope_env.py (410 lines)
└── OpenScopeEnv
    ├── Initialization (spaces, config)
    ├── Browser management (sync Playwright)
    ├── Command execution (JS injection)
    ├── State extraction (JS evaluation)
    ├── Observation conversion (numpy arrays)
    ├── Reward calculation (score + shaping)
    └── Gym interface (reset/step/close)
```

Clean, flat structure - no complex inheritance.

---

## Usage with SB3

The environment works seamlessly with Stable-Baselines3:

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from models.sb3_policy import ATCTransformerPolicy

vec_env = DummyVecEnv([lambda: env])
model = PPO(ATCTransformerPolicy, vec_env, verbose=1)
model.learn(total_timesteps=100000)
```

---

## Testing

All tests pass:

```bash
$ uv run python poc/test_env.py
✅ Environment created
✅ Reset successful
✅ Step successful
✅ Game time advancing
✅ All tests passed!
```

```bash
$ uv run python poc/test_notebook_compat.py
✅ Notebook compatibility verified!
```

---

## Notes

- **Self-contained POC** - all code included
- **Configuration embedded in notebook** for easier testing
- **Headless mode supported** (works on servers without displays)
- **Drop-in replacement** - same API as original
- **No changes needed** to existing notebook code

---

## Learn More

- **[SIMPLIFICATION_SUMMARY.md](SIMPLIFICATION_SUMMARY.md)** - Detailed analysis of changes
- **[openscope_rl_demo.ipynb](openscope_rl_demo.ipynb)** - Interactive demo
- **[environment/openscope_env.py](environment/openscope_env.py)** - Optimized environment code

---

## Conclusion

**Simpler is better.** We reduced code by 30.6% while making it clearer and more maintainable.

Perfect for POC experimentation and learning! 🚀
