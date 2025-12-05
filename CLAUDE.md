# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a demonstration environment for exploring air traffic control using the OpenScope ATC simulator. It provides interactive notebooks and self-contained environments for experimentation.

**Key Technologies:**
- Playwright for browser automation (supports both sync and async APIs)
- Jupyter notebooks for interactive exploration
- Gymnasium-compatible environments for RL
- PyTorch and Stable-Baselines3 for training
- Self-contained POC environments for rapid prototyping

**Critical Dependency**: The parent directory must contain the OpenScope game server (Node.js). The game server must be running at http://localhost:3003 before the environment can be used. This project connects to a single existing server instance - it does not spawn multiple servers.

## Development Setup

This project uses [uv](https://github.com/astral-sh/uv) for fast Python package management.

```bash
# Install uv (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Python dependencies (automatically creates .venv)
uv sync

# Install Playwright browsers (required for game automation)
uv run playwright install chromium

# Start the OpenScope game server (from parent directory)
cd ../openscope
npm install
npm run build
npm run start  # Must be running at localhost:3003
```

## Common Commands

### Running Demos
```bash
# Start OpenScope server first (in parent directory)
cd ../openscope && npm start

# Run the main demo notebook
jupyter notebook openscope_simple_demo.ipynb

# Explore POC environments
jupyter notebook poc/atc_2d_demo.ipynb
jupyter notebook poc/atc_3d_demo.ipynb
```

### Code Quality
```bash
# Format code
uv run black .

# Lint code
uv run ruff check .

# Type check
uv run mypy environment models
```

## Architecture Overview

The codebase is organized into three main modules, each with clean separation of concerns:

### 1. Main Environment Module (`environment/`)

The main OpenScope integration using Playwright browser automation:

**Core Components:**
- `playwright_env.py` - Main `PlaywrightEnv` class (inherits from `gymnasium.Env`)
- `config.py` - Dataclass-based configuration system with validation
- `utils.py` - Browser management with **Jupyter/asyncio compatibility**
- `game_interface.py` - OpenScope game communication via JavaScript injection
- `state_processor.py` - Game state → RL observation conversion
- `reward_calculator.py` - Pluggable reward strategies (default, safety, efficiency)
- `spaces.py` - Gymnasium observation/action space definitions
- `metrics.py` - Episode metrics tracking
- `constants.py` - Environment constants and JavaScript injection scripts
- `exceptions.py` - Custom exception hierarchy

**Key Architecture Patterns:**
- **Strategy Pattern**: Multiple reward calculation strategies
- **Separation of Concerns**: Each module has a single, well-defined responsibility
- **Configuration Dataclasses**: Type-safe configuration with validation
- **Async/Sync Compatibility**: `PageWrapper` class enables Jupyter notebook support

### 2. Models Module (`models/`)

Neural network architectures for RL training:

**Components:**
- `networks.py` - Main `ATCActorCritic` model with shared encoder
- `config.py` - Network configuration dataclasses
- `encoders.py` - Transformer encoder and attention pooling for variable aircraft count
- `heads.py` - Separate policy and value heads

**Architecture:**
- Transformer-based aircraft sequence processing
- Attention pooling aggregates variable-length aircraft sequences
- Shared encoder reduces parameter count
- Separate policy/value heads for actor-critic methods

### 3. POC Module (`poc/atc_rl/`)

Self-contained, fast ATC environments for rapid prototyping:

**Environments:**
- `environment_2d.py` - Simple 2D ATC with vectorized conflict detection
- `environment_3d.py` - Realistic 3D ATC with altitude, runways, physics

**Supporting Modules:**
- `physics.py` - Vectorized physics calculations and conflict detection
- `constants.py` - All POC configuration in one place

**Key Feature**: POC environments have **zero dependencies** on the main environment module - they are completely self-contained for easy experimentation.

## Critical Implementation Details

### Jupyter Notebook Compatibility

The environment supports both standard Python scripts and Jupyter notebooks through intelligent async/sync API detection:

**How it works:**
1. `BrowserManager.initialize()` detects if running in an asyncio event loop
2. If yes (Jupyter): uses Playwright's async API with `async_playwright()`
3. If no (script): uses sync API with `sync_playwright()`
4. `PageWrapper` class provides unified sync interface for both cases

**Why this matters:**
- Jupyter notebooks run with an active asyncio event loop
- Playwright's sync API cannot run inside an event loop
- The wrapper allows seamless use in both environments

**Code location**: [environment/utils.py:24-103](environment/utils.py#L24-L103) (PageWrapper class)

### Game Communication Flow

The environment communicates with OpenScope through JavaScript injection:

1. **Browser Setup**: `BrowserManager` launches Chromium and navigates to game URL
2. **Script Injection**: JavaScript functions injected via `page.add_init_script()`
3. **State Extraction**: `page.evaluate()` executes JS to get game state from `window.gameController`
4. **Command Execution**: DOM manipulation to send commands to OpenScope's input system
5. **State Processing**: Raw game state → RL observation via `StateProcessor`

**Key Scripts** (in `constants.py`):
- `JS_GET_GAME_STATE_SCRIPT` - Extracts aircraft, conflicts, score, time
- `JS_EXECUTE_COMMAND_SCRIPT` - Simulates user input to game

### Configuration System

All components use dataclass-based configuration:

```python
from environment import PlaywrightEnv, create_default_config

# Create custom configuration
config = create_default_config(
    airport="KJFK",
    max_aircraft=15,
    headless=False,
    timewarp=10
)

# Create environment
env = PlaywrightEnv(**config.__dict__)
```

**Benefits:**
- Type safety with Python type hints
- Validation at configuration creation
- Easy to serialize/deserialize
- IDE autocomplete support

### Reward Strategies

Multiple reward calculation strategies available:

- `"default"` - Balanced safety and performance
- `"safety"` - Heavily prioritizes conflict avoidance
- `"efficiency"` - Emphasizes throughput and aircraft exits
- `"progress"` - Continuous progress rewards (distance, altitude compliance, waypoint advancement)

Change strategy: `env.set_reward_strategy("progress")`

### Hyperparameter Optimization

Use `training.hyperparam_tuner.HyperparamTuner` for Optuna-based systematic hyperparameter optimization:

```python
from training.hyperparam_tuner import HyperparamTuner

tuner = HyperparamTuner(
    algo="ppo",
    env_factory=env_factory,
    model_factory=model_factory,
    n_trials=50,
    sampler="tpe",
    pruner="halving",
)
results = tuner.optimize()
```

### Performance Benchmarking

Use `experiments.performance_benchmark` for measuring environment and model performance:

```python
from experiments.performance_benchmark import benchmark_env_performance, benchmark_model_inference

env_metrics = benchmark_env_performance(env, n_steps=10000)
model_metrics = benchmark_model_inference(model, env, n_steps=1000)
```

### Configuration Logging

Use `training.config_logger` for saving and loading training configurations with reproducibility:

```python
from training.config_logger import save_training_config, load_training_config

save_training_config(
    save_dir="experiments/run_001",
    model_config={"learning_rate": 3e-4, ...},
    env_config={"airport": "KLAS", ...},
    training_config={"total_timesteps": 1000000, ...},
)
```

## Modifying the Environment

### Adding New Commands

1. Add command type to `CommandType` enum in `config.py`
2. Update action space in `spaces.py`
3. Implement command execution in `game_interface.py`
4. Update `constants.py` with any new JavaScript injection needed

### Changing Observation Space

1. Modify state extraction in `state_processor.py`
2. Update observation space definition in `spaces.py`
3. Ensure model architecture supports new observation shape

### Adding Reward Components

1. Extend `RewardConfig` dataclass in `config.py`
2. Create new strategy in `reward_calculator.py`
3. Register strategy in `create_reward_calculator()` factory

## POC Environments

The POC environments are designed for rapid iteration without OpenScope dependencies:

```python
from poc.atc_rl import Simple2DATCEnv, Realistic3DATCEnv

# Simple 2D environment
env_2d = Simple2DATCEnv(max_aircraft=5)
obs, info = env_2d.reset()

# Realistic 3D environment
env_3d = Realistic3DATCEnv(max_aircraft=10)
obs, info = env_3d.reset()
```

**Use POC environments when:**
- Prototyping new reward functions
- Testing RL algorithms quickly
- Debugging observation/action spaces
- Generating training data without browser overhead

**Use main environment when:**
- Training with real OpenScope game dynamics
- Testing on actual airport configurations
- Evaluating on realistic ATC scenarios

## Documentation

**Main Documentation:**
- [README.md](README.md) - Project overview, quick start, and comprehensive guide
- [CLAUDE.md](CLAUDE.md) - This file - development guide for Claude Code

## Common Pitfalls

### Game Server Not Running
**Error**: "Navigation failed with HTTP 503" or "Game failed to become ready"
**Solution**: Ensure OpenScope server is running at http://localhost:3003

### Playwright Async API Error (FIXED)
**Error**: "using Playwright Sync API inside the asyncio loop"
**Solution**: This is now automatically handled - the environment detects Jupyter and uses async API

### State Extraction Returns Empty
**Error**: Aircraft array is empty but game shows aircraft
**Solution**: Check JavaScript console for errors - game controller may not be initialized

### Browser Hangs on Headless Mode
**Issue**: Environment freezes in headless mode
**Solution**: Set `headless=False` for debugging, or increase timeout in `game_interface.py`

## Code Style

This project follows:
- **Black** for code formatting (line length: 100)
- **Ruff** for linting (see `pyproject.toml` for rules)
- **Google-style docstrings** for documentation
- **Type hints** throughout (gradual typing, not strict)
