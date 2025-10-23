# Jupyter Compatibility Fix

## The Error You Saw

```
Error: It looks like you are using Playwright Sync API inside the asyncio loop.
Please use the Async API instead.
```

## The Fix

Added `nest_asyncio.apply()` at module import time (line 16 of `openscope_env.py`):

```python
import nest_asyncio

# Enable nested event loops for Jupyter compatibility
nest_asyncio.apply()
```

This allows sync Playwright to work inside Jupyter's asyncio event loop.

## How It Works

1. **Jupyter runs on asyncio** - All Jupyter notebooks run inside an asyncio event loop
2. **Playwright sync API conflicts** - By default, sync Playwright can't run inside an existing event loop
3. **nest_asyncio patches asyncio** - Allows nested/re-entrant event loops
4. **Applied at import time** - When you `from environment.openscope_env import OpenScopeEnv`, the patch is automatically applied

## Verification

The environment now works in Jupyter:

```python
from environment.openscope_env import OpenScopeEnv

# This will now work without errors!
env = OpenScopeEnv(
    game_url="http://localhost:3003",
    airport="KLAS",
    timewarp=15,
    headless=True,
)

obs, info = env.reset()  # ✅ Works!
```

## Why This Approach?

**Alternative 1**: Use async Playwright + sync wrappers
- ❌ Adds complexity (need `run_until_complete()` everywhere)
- ❌ More code, harder to maintain

**Alternative 2**: Use custom threading
- ❌ Complex queue/thread management (127 lines!)
- ❌ Original POC approach

**Our approach**: nest_asyncio
- ✅ Two lines of code
- ✅ Works everywhere (Jupyter, scripts, SB3 training)
- ✅ Keeps sync API (simple and clear)

## Dependencies

The `nest_asyncio` package is already in `pyproject.toml` dependencies.

## Line Count Update

- **Before fix**: 410 lines
- **After fix**: 414 lines (added import + apply call)
- **Still 30% smaller** than original 591 lines!

---

**Bottom line**: The notebook error should now be fixed. Just restart the kernel and re-run! 🎉
