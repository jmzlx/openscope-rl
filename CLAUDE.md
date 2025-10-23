# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a reinforcement learning training system for teaching AI agents to play the openScope air traffic control simulator. It uses **Stable-Baselines3** (SB3) with a custom Transformer-based neural network architecture to control air traffic.

**Key Technologies:**
- Stable-Baselines3 for PPO algorithm
- PyTorch with custom Transformer encoder
- Gymnasium with wrappers for normalization
- Playwright for browser automation
- wandb/Tensorboard for experiment tracking
- Jupyter notebook support for interactive experimentation

**Dependencies**: The parent directory must contain the openScope game server (Node.js). The game server must be running at http://localhost:3003 before training can begin.

**Project Structure:**
- **Main codebase**: Production-ready training system with SB3 integration
- **POC directory** (`poc/`): Simplified proof-of-concept (30% less code) for experimentation
- **Self-contained POC** (`poc-self-contained/`): Fully standalone version with embedded config
- **Documentation**: Extensive .md files covering setup, architecture, and troubleshooting

## Development Setup

This project uses [uv](https://github.com/astral-sh/uv) for fast Python package management.

```bash
# Install uv (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Python dependencies (automatically creates .venv)
uv sync

# Install Playwright browsers (required for game automation)
uv run playwright install chromium

# Start the openScope game server (from parent directory)
cd ..
npm install
npm run build
npm run start  # Must be running at localhost:3003
```

**Why uv?**
- ⚡ 10-100x faster than pip
- 🔒 Reliable with lock file (`uv.lock`)
- 🎯 Automatic virtual environment management
- 📦 Better dependency resolution

## Common Commands

### Testing
```bash
# Verify SB3 integration works
uv run test_sb3_integration.py

# Test environment cleanup (browser/process management)
uv run verify_cleanup.py

# POC tests (simplified environment)
uv run python poc/test_env.py
uv run python poc/test_notebook_compat.py
uv run python poc/test_jupyter_compat.py
```

### Training
```bash
# Basic training (single environment)
uv run train_sb3.py

# Training with 4 parallel environments (4x speedup!)
uv run train_sb3.py --n-envs 4

# Training with 8 parallel environments (8x speedup!)
uv run train_sb3.py --n-envs 8

# Fast test training (quick feedback, reduced parameters)
uv run train_sb3.py --config config/test_config.yaml --n-envs 4

# Resume from checkpoint
uv run train_sb3.py --checkpoint checkpoints/checkpoint_50000_steps.zip --n-envs 4

# With wandb logging
uv run train_sb3.py --wandb --n-envs 4

# Custom device
uv run train_sb3.py --device cuda --n-envs 4
```

### Evaluation
```bash
# Evaluate trained model
uv run evaluate_sb3.py --checkpoint checkpoints/checkpoint_final.zip --n-episodes 20

# Deterministic evaluation
uv run evaluate_sb3.py --checkpoint checkpoints/best_model.zip --deterministic

# With rendering (non-headless browser)
uv run evaluate_sb3.py --checkpoint checkpoints/best_model.zip --render
```

### Monitoring
```bash
# TensorBoard (SB3 auto-logs to this)
uv run tensorboard --logdir logs/

# wandb (if using --wandb flag)
# View at: https://wandb.ai/your-username/openscope-rl
```

## Architecture Overview

### Core Components

**Environment** (`environment/openscope_env.py`):
- Gymnasium wrapper around openScope browser game
- Uses Playwright for browser automation (keyboard commands, JavaScript injection)
- Observation space: per-aircraft features (14-dim), global state (4-dim), conflict matrix
- Action space: hierarchical Dict (aircraft selection → command type → parameters)
- Reward function: game score deltas + configurable shaped rewards
- **Wrapped** with Gymnasium wrappers for normalization (see `environment/wrappers.py`)

**Network** (`models/networks.py`):
- `ATCTransformerEncoder`: Self-attention over variable number of aircraft
- `ATCActorCritic`: Actor-critic with **shared** transformer encoder
  - Actor: Hierarchical action head (aircraft selection → command type → parameters)
  - Critic: Value function head for state evaluation
  - Uses attention pooling to handle variable aircraft count
- **Key optimization**: Single shared encoder for both policy and value (50% memory reduction, 2x faster)

**SB3 Policy Wrapper** (`models/sb3_policy.py`):
- Bridges our custom Transformer with SB3's training infrastructure
- `ATCTransformerPolicy`: SB3-compatible wrapper around `ATCActorCritic`
- Handles Dict action space for hierarchical actions

**PPO Algorithm** (Stable-Baselines3):
- Battle-tested PPO implementation with GAE
- Vectorized environments for parallel rollouts
- Automatic checkpointing, logging, evaluation
- Mixed precision training support

**Training Loop** (`train_sb3.py`):
- Uses SB3's `PPO.learn()` - handles everything!
- Vectorized environments (1-8 parallel browsers)
- Curriculum learning via SB3 callback
- Checkpoint and evaluation callbacks
- wandb integration via WandbCallback

**Utilities**:
- `environment/wrappers.py`: Creates environment with Gymnasium wrappers
- `utils/curriculum_callback.py`: SB3 callback for curriculum learning
- `utils/curriculum.py`: Curriculum stage management

**Scripts**:
- `train_sb3.py`: Main training script with vectorized environments and callbacks
- `evaluate_sb3.py`: Evaluation script for trained models
- `test_sb3_integration.py`: Integration test for SB3 setup
- `verify_cleanup.py`: Test browser/process cleanup

**Configuration**:
- `config/training_config.yaml`: Main training configuration (production)
- `config/test_config.yaml`: Fast test configuration (quick iteration)
- `pyproject.toml`: Python dependencies and project metadata

**Notebooks**:
- `openscope_rl_selfcontained.ipynb`: Self-contained demo notebook
- `poc/openscope_rl_demo.ipynb`: POC demo notebook with simplified environment

### Key Architectural Decisions

1. **Shared encoder architecture**: Single transformer encoder for policy and value (50% memory, 2x speed)
2. **Stable-Baselines3**: Leverages battle-tested RL library instead of custom implementation
3. **Vectorized environments**: 4-8 parallel browser instances for 4-8x speedup
4. **Gymnasium wrappers**: Automatic observation/reward normalization with running statistics
5. **Hierarchical Dict actions**: Natural representation for aircraft selection + command + parameters
6. **Transformer for aircraft**: Handles variable aircraft count naturally via self-attention and masking
7. **Browser automation**: Uses Playwright for flexible game interface (trade-off: slower than API)

## Configuration

Main config: `config/training_config.yaml`
- `env`: Airport, timewarp (1-10), max aircraft, episode length, action_interval
- `ppo`: Learning rate, gamma, GAE lambda, clip epsilon, n_steps, batch size, epochs
- `network`: Hidden dimensions (256), attention heads (8), transformer layers (4)
- `actions`: Discrete action sets (altitudes, headings, speeds)
- `rewards`: Reward coefficients for shaping (now actually used!)
- `curriculum`: Stage definitions (max_aircraft, difficulty, episodes)

Fast test config: `config/test_config.yaml` - reduced parameters for quick iteration

## Working with This Codebase

### Modifying the Network Architecture

The network is in `models/networks.py`. Key class:
- `ATCActorCritic`: Single class with shared encoder + policy/value heads
  - Modify `hidden_dim`, `num_heads`, `num_layers` for architecture changes
  - Both policy and value use same encoder (don't separate them!)

When changing action space dimensions:
1. Update `models/networks.py`: Action head output sizes in `ATCActorCritic.__init__()`
2. Update `environment/openscope_env.py`: `action_space` definition
3. Update `config/training_config.yaml`: `actions` section
4. Update `models/sb3_policy.py` if action structure changes

### Modifying the Environment

Browser automation is in `environment/openscope_env.py`:
- `_execute_command()`: Executes ATC commands via keyboard simulation
- `_get_game_state()`: Injects JavaScript to extract game state
- `_calculate_reward()`: Reward function (uses `config['rewards']` now!)
- `_state_to_observation()`: Converts raw game state to network input

**Important**: Environment is wrapped with Gymnasium wrappers (see `environment/wrappers.py`):
- `NormalizeObservation`: Running mean/std for observations
- `NormalizeReward`: Running mean/std for rewards
- `RecordEpisodeStatistics`: Automatic episode tracking

### Using Vectorized Environments

**Key advantage**: Train with 4-8 parallel browser instances for 4-8x speedup!

```bash
# Start with 4 parallel environments
uv run train_sb3.py --n-envs 4
```

**Considerations**:
- Each browser needs ~1-2GB RAM
- Each browser takes 1-3 seconds per step
- 4 parallel envs = 4x speedup (if you have 4+ CPU cores)
- Game server must handle multiple connections

**Troubleshooting parallel environments**:
- If browser crashes: Reduce `timewarp` or `n-envs`
- If RAM limited: Use fewer parallel environments
- If CPU limited: Parallel envs won't speed up much

### Debugging Training Issues

**Agent does nothing / low entropy**:
- Increase `entropy_coef` in config
- Check action masking in environment
- Verify observations are reasonable (not all zeros)

**Training unstable / loss explodes**:
- Reduce `learning_rate`
- Reduce `clip_epsilon` (more conservative updates)
- Check for NaN in observations
- Verify normalization wrappers are applied

**Slow training**:
- **Use vectorized environments**: `--n-envs 4` or `--n-envs 8`
- Use `config/test_config.yaml` for faster iteration
- Increase `timewarp` in config (but watch for browser crashes)
- Reduce `episode_length` for faster episodes

**Browser crashes**:
- Reduce `timewarp` (high values strain browser)
- Reduce `n-envs` (fewer parallel browsers)
- Increase `action_interval` (give browser more time)
- Check game server logs for errors

**Out of memory**:
- Reduce `n-envs` (fewer parallel environments)
- Reduce `batch_size` in config
- Reduce `n_steps` in config

### Custom Callbacks

Create SB3 callbacks for custom training logic:

```python
from stable_baselines3.common.callbacks import BaseCallback

class MyCallback(BaseCallback):
    def _on_step(self) -> bool:
        # Called after each environment step
        # Access: self.locals, self.globals, self.model
        return True  # Continue training
```

Add to training:
```python
callbacks.append(MyCallback())
```

See `utils/curriculum_callback.py` for example.

### Running Tests

```bash
# Integration test - verifies SB3 setup works
uv run test_sb3_integration.py
```

This tests:
- Environment creation with wrappers
- Model initialization with custom policy
- Forward pass through network
- Training step
- Save/load functionality

Run this before full training to catch issues early.

### Using Jupyter Notebooks

The project includes Jupyter notebook support for interactive experimentation:

```bash
# Install Jupyter dependencies (included in pyproject.toml)
uv sync

# Launch Jupyter
uv run jupyter notebook

# Open the self-contained demo notebook
# File: openscope_rl_selfcontained.ipynb
# Or POC version: poc/openscope_rl_demo.ipynb
```

**Notebook compatibility notes:**
- Uses `nest-asyncio` to handle event loop conflicts
- Supports both headless and non-headless modes
- Includes embedded configuration for easy experimentation
- POC directory has a simplified environment (30% less code)

### Code Style

- Uses Black formatter (run: `uv run black .`)
- Uses Ruff linter (run: `uv run ruff check .`)
- Type hints for function signatures
- Docstrings in Google style

## Performance Benchmarks

**With Vectorized Environments** (major improvement!):
- Single environment: ~0.5 steps/second
- 4 parallel environments: ~2.0 steps/second (4x speedup)
- 8 parallel environments: ~4.0 steps/second (8x speedup)

**Network Performance**:
- Forward pass: ~5ms (shared encoder = 2x faster than before)
- Memory: 50% less than separate encoders
- Bottleneck: Still browser automation, but vectorization helps!

**Expected Training Time**:
- 10M steps, single env: ~5-7 days
- 10M steps, 4 parallel envs: ~1-2 days
- 10M steps, 8 parallel envs: ~18-24 hours

## Checkpoints and Logging

**Checkpoints** (saved to `checkpoints/` directory):
- SB3 format: `checkpoint_<steps>_steps.zip`
- Contains: model, optimizer, normalization stats, config
- Best model: `checkpoints/best_model/best_model.zip` (from EvalCallback)
- Final model: `checkpoints/checkpoint_final.zip`

**Loading checkpoints**:
```python
from stable_baselines3 import PPO
from models.sb3_policy import ATCTransformerPolicy

model = PPO.load(
    "checkpoints/checkpoint_final.zip",
    custom_objects={"policy_class": ATCTransformerPolicy}
)
```

**Logging**:
- **TensorBoard**: Automatic logging to `logs/` directory
  - View with: `tensorboard --logdir logs/`
  - Includes: rewards, losses, episode length, policy stats
- **wandb**: Optional with `--wandb` flag
  - Syncs TensorBoard logs automatically
  - Adds system metrics, code saving, model artifacts

## OpenScope Game Integration

The training system interfaces with the openScope web game:
- Game runs in Chromium browser (headless or visible)
- Commands sent via keyboard simulation (`page.keyboard.press()`)
- Game state extracted via JavaScript injection (`page.evaluate()`)
- Game must be running at localhost:3003 before training

Command format examples:
- Altitude: `AA123 c 10000` (climb to 10,000 ft)
- Heading: `AA123 t l 270` (turn left to heading 270)
- Speed: `AA123 sp 250` (speed 250 knots)
- ILS: `AA123 i 25L` (ILS approach runway 25L)

## Common Gotchas

1. **Parent directory dependency**: openScope game server must be in `../` and running at localhost:3003
2. **Vectorized environments need RAM**: Each browser instance needs 1-2GB. Use `--n-envs 4` max if RAM limited
3. **Checkpoint format change**: Old `.pt` checkpoints incompatible with new `.zip` format (must retrain)
4. **Custom objects for loading**: Must pass `custom_objects={"policy_class": ATCTransformerPolicy}` when loading
5. **Transformer masking**: Aircraft mask is True for valid, False for padding (opposite of attention mask)
6. **Action space is Dict**: Not MultiDiscrete - each component is separate Discrete space
7. **Observation normalization**: Wrappers compute running stats - observations will differ from raw env
8. **Headless mode**: Set `headless: true` in config for WSL/remote servers without X display (see [docs/WSL_DISPLAY_FIX.md](docs/WSL_DISPLAY_FIX.md))
9. **Jupyter event loops**: Use `nest-asyncio` to avoid Playwright event loop conflicts in notebooks
10. **Browser threading**: Main environment uses dedicated thread for Playwright; POC uses simpler sync approach

## POC and Experimentation

The project includes two POC (proof-of-concept) directories for experimentation:

### `poc/` - Simplified POC
- **410 lines** (30% less than main environment)
- Removes custom threading in favor of sync Playwright
- Full type hints and better documentation
- Same API as main codebase (drop-in replacement)
- Includes Jupyter notebooks for interactive testing
- Run: `uv run python poc/test_env.py`

### `poc-self-contained/` - Fully Standalone
- Completely self-contained with embedded configuration
- No dependencies on parent directories
- Ideal for quick experimentation and learning
- Includes optimization guide

**When to use POC**:
- Learning the codebase
- Rapid prototyping of new features
- Testing environment changes without affecting production code
- Jupyter notebook experimentation

**When to use main codebase**:
- Production training runs
- Multi-environment vectorization
- Full curriculum learning
- wandb integration

## Documentation Files

The repository includes extensive documentation:

**Setup and Configuration:**
- [README.md](README.md) - Main project documentation
- [docs/OPENSCOPE_SETUP_FINAL.md](docs/OPENSCOPE_SETUP_FINAL.md) - OpenScope game server setup
- [docs/WSL_DISPLAY_FIX.md](docs/WSL_DISPLAY_FIX.md) - Fix for WSL display issues
- [docs/HEADLESS_MODE_FIX.md](docs/HEADLESS_MODE_FIX.md) - Headless browser configuration

**Architecture and Integration:**
- [docs/OPENSCOPE_SOURCE_ANALYSIS.md](docs/OPENSCOPE_SOURCE_ANALYSIS.md) - Source code analysis
- [docs/OPENSCOPE_GAP_ANALYSIS.md](docs/OPENSCOPE_GAP_ANALYSIS.md) - Analysis of missing features
- [docs/GAP_ANALYSIS_SUPPLEMENTARY.md](docs/GAP_ANALYSIS_SUPPLEMENTARY.md) - Additional gap analysis

Refer to these documents when:
- Setting up the environment for the first time
- Troubleshooting browser/display issues
- Understanding game internals for environment modifications
- Exploring advanced integration techniques

## Migration from Old System

If you have old code/checkpoints, see `SB3_MIGRATION_GUIDE.md` for:
- What changed and why
- Performance improvements
- Breaking changes
- How to migrate custom code
- Troubleshooting guide

**TLDR**: Old `.pt` checkpoints are incompatible. Retrain with `train_sb3.py` for 4-8x speedup!

## Development Workflow

**Typical development cycle:**

1. **Start game server** (in parent directory):
   ```bash
   cd ../openscope && npm start
   ```

2. **Test changes**:
   ```bash
   # Quick integration test
   uv run test_sb3_integration.py

   # Or test in POC environment
   uv run python poc/test_env.py
   ```

3. **Fast iteration** with test config:
   ```bash
   uv run train_sb3.py --config config/test_config.yaml --n-envs 1
   ```

4. **Full training** once validated:
   ```bash
   uv run train_sb3.py --n-envs 4 --wandb
   ```

5. **Monitor** with TensorBoard:
   ```bash
   uv run tensorboard --logdir logs/
   ```

**Code quality tools:**
```bash
# Format code
uv run black .

# Lint code
uv run ruff check .

# Type checking (if needed)
uv run mypy .
```

**Useful environment variables:**
- `DISPLAY=:99` - Set for Xvfb virtual display (WSL/remote)
- `PLAYWRIGHT_BROWSERS_PATH` - Custom browser installation path
