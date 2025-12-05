# OpenScope RL: Reinforcement Learning for Air Traffic Control

A comprehensive reinforcement learning framework for training AI agents to control air traffic using the [OpenScope ATC simulator](https://github.com/openscope/openscope). This project provides a production-ready Gymnasium environment, data collection pipelines, and multiple RL approaches for learning effective air traffic control policies.

## 🎯 Overview

**OpenScope** is a browser-based air traffic control (ATC) simulator featuring realistic aircraft physics, airport layouts, and ATC procedures. This project wraps OpenScope in a Gymnasium-compatible environment using Playwright for browser automation, enabling reinforcement learning research on realistic ATC scenarios.

### Key Features

- 🛫 **Realistic ATC Environment**: Full OpenScope simulator with 180+ airports and realistic aircraft physics
- 🤖 **Gymnasium Interface**: Standard RL environment API with structured observations and actions
- 📊 **Data Collection Pipeline**: Tools for collecting expert demonstrations and training datasets
- 🧠 **Multiple RL Approaches**: PPO, Decision Transformer, World Models, Hierarchical RL, and more
- 🔄 **Modular Architecture**: Clean separation of concerns for maintainability and extensibility
- 📈 **Comprehensive Metrics**: Episode tracking, conflict detection, and performance analysis

## 🚀 Quick Start

### 1. Start OpenScope Server

```bash
# Navigate to OpenScope directory (must be sibling to this repo)
cd ../openscope
npm install
npm run build
npm run start  # Runs at localhost:3003
```

### 2. Install Dependencies

```bash
# Install Python dependencies
uv sync

# Install Playwright browser
uv run playwright install chromium
```

### 3. Run Demo

```bash
# Option 1: Interactive environment demo
uv run python scripts/demo_env.py --airport KLAS --max-aircraft 10

# Option 2: Evaluate a trained model
uv run python scripts/evaluate_model.py --model-path checkpoints/ppo_model.zip --n-episodes 10

# Option 3: Open the exploration notebook
jupyter notebook notebooks/00_explore_openscope_api.ipynb

# Option 4: Train a PPO model
uv run python training/ppo_trainer.py --total-timesteps 100000 --n-envs 4

# Option 5: Run smoke tests (verify all entry points work)
uv run python scripts/smoke_test.py
```

## 📁 Project Structure

```
openscope-rl/
├── environment/               # OpenScope environment (refactored)
│   ├── playwright_env.py      # Main environment orchestrator
│   ├── config.py             # Configuration dataclasses
│   ├── utils.py              # Browser management utilities
│   ├── game_interface.py      # Game communication interface
│   ├── state_processor.py    # State processing and observations
│   ├── reward_calculator.py  # Reward calculation strategies
│   ├── spaces.py             # Observation/action space definitions
│   ├── metrics.py            # Episode metrics tracking
│   ├── constants.py          # Environment constants
│   └── exceptions.py         # Custom exceptions
├── models/                    # Neural network models (refactored)
│   ├── networks.py            # Main ATCActorCritic model
│   ├── config.py             # Network configuration
│   ├── encoders.py           # Transformer encoders
│   └── heads.py              # Policy and value heads
├── poc/                      # Proof of concept demos (self-contained)
│   ├── atc_rl/               # POC ATC environments
│   │   ├── environment_2d.py # Simple 2D ATC environment
│   │   ├── environment_3d.py # Realistic 3D ATC environment
│   │   ├── physics.py        # Physics calculations
│   │   ├── constants.py      # POC constants
│   │   └── constants.py      # POC constants
│   └── *.ipynb               # Demo notebooks
└── openscope_async_demo.ipynb # Main demo notebook
```

## 🎮 The OpenScope Environment

### What is OpenScope?

OpenScope is an open-source, web-based air traffic control simulator that provides:

- **180+ Real Airports**: Including KJFK, KLAS, KSEA, EGLL, and more
- **Realistic Physics**: Aircraft acceleration, turning rates, wind effects
- **Standard ATC Procedures**: ILS approaches, SIDs, STARs, holding patterns
- **Conflict Detection**: Real-time separation violation monitoring
- **Scoring System**: Penalties for conflicts, violations, and inefficiency

### Our Gymnasium Wrapper

The `PlaywrightEnv` class wraps OpenScope in a standard Gymnasium interface:

```python
from environment import PlaywrightEnv

# Create environment
env = PlaywrightEnv(
    airport="KLAS",           # Las Vegas McCarran
    max_aircraft=20,          # Track up to 20 aircraft
    timewarp=5,               # 5x speed
    headless=False,           # Show browser window
    reward_strategy="default" # Reward calculation method
)

# Standard RL loop
obs, info = env.reset()
for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

### Observation Space

The environment provides structured observations with the following components:

**1. Aircraft Features** (`shape: (max_aircraft, 14)`)
- Position (x, y) - Normalized coordinates
- Altitude, heading, speed, ground speed
- Assigned altitude, heading, speed (from previous commands)
- Boolean flags: is_on_ground, is_taxiing, is_established
- Category: is_arrival, is_departure

**2. Aircraft Mask** (`shape: (max_aircraft,)`)
- Boolean mask indicating which aircraft slots are active
- Handles variable number of aircraft (0 to max_aircraft)

**3. Global State** (`shape: (4,)`)
- Current simulation time (hours)
- Aircraft density (fraction of max_aircraft)
- Conflict density (conflicts per aircraft)
- Current score (normalized)

**4. Conflict Matrix** (`shape: (max_aircraft, max_aircraft)`)
- Pairwise conflict information
- Values: 0 (no conflict), 0.5 (predicted conflict), 1.0 (separation violation)
- Symmetric matrix for efficient attention mechanisms

### Action Space

Actions are represented as discrete dictionaries:

```python
{
    "aircraft_id": int,      # 0 to max_aircraft (max_aircraft = no-op)
    "command_type": int,     # 0=altitude, 1=heading, 2=speed, 3=ILS, 4=direct
    "altitude": int,         # Altitude level (0-40, representing 0-40,000 ft)
    "heading": int,          # Heading change (-180 to +180 degrees, quantized)
    "speed": int,            # Speed level (140-520 knots, quantized)
}
```

**Command Types:**
- **ALTITUDE**: Change target altitude
- **HEADING**: Change heading (turn left/right)
- **SPEED**: Change target speed
- **ILS**: Establish on ILS approach
- **DIRECT**: Direct to waypoint/fix

**No-Op Action**: Setting `aircraft_id = max_aircraft` performs no action, allowing the agent to "wait and observe."

### Reward Structure

The environment supports multiple reward strategies:

**Default Strategy:**
- +10 per aircraft successfully exited
- -50 per conflict detected
- -200 per separation violation
- -0.1 per time step (efficiency incentive)
- -500 episode termination penalty (if score too low)

**Safety-Focused Strategy:**
- Higher penalties for conflicts/violations
- Lower rewards for successful exits
- Emphasis on maintaining separation

**Efficiency-Focused Strategy:**
- Higher rewards for quick exits
- Lower penalties for minor conflicts
- Emphasis on throughput

**Progress Strategy:**
- Continuous rewards for distance progress (arrivals/departures)
- Altitude compliance with flight plans
- Approach establishment bonuses (on final, on glidepath)
- Waypoint advancement rewards
- Combined with safety rewards for balanced learning

Custom reward strategies can be implemented by subclassing `RewardCalculator`.

## 📊 Data Collection

The project includes a comprehensive data collection pipeline for offline RL and imitation learning.

### Training Data Collector

The `TrainingDataCollector` class collects complete episode trajectories:

```python
from data.training_data_collector import TrainingDataCollector

collector = TrainingDataCollector(
    output_dir="training_data",
    airport="KLAS",
    max_aircraft=10,
    save_raw_states=True,  # Include full game state
    save_wind_data=True,   # Include wind components
)

# Collect episodes with different policies
episodes = collector.collect_episodes(
    num_episodes=100,
    policy="random",        # Options: "random", "heuristic", "expert"
    max_steps=1000,
)
```

### What Data is Collected?

Each episode contains:

**1. Observations** (normalized for RL)
- Aircraft features (position, velocity, state)
- Conflict matrix
- Global environment state
- All aligned with Gymnasium observation space

**2. Actions** (discrete action dictionaries)
- Aircraft selection
- Command type and parameters
- Recorded in environment format

**3. Rewards** (float per step)
- Calculated using specified reward strategy
- Includes all components (conflicts, exits, efficiency)

**4. Episode Metadata**
- Episode length (steps)
- Total reward
- Final score
- Violation count
- Conflict count
- Successful landings/departures

**5. Returns-to-Go** (computed automatically)
- Cumulative future rewards from each step
- Used for Decision Transformer training

**6. Raw State Data** (optional, for analysis)
- Complete OpenScope game state
- Wind components (via `getWindComponents()` method)
- Runway assignments
- Flight plans
- All aircraft details

### Dataset Structure

Collected data is saved in pickle format with the following structure:

```
training_data/
├── ep_00000_random.pkl       # Individual episodes
├── ep_00001_random.pkl
├── ...
└── dataset_summary.json      # Dataset statistics
```

**Dataset Summary:**
```json
{
  "num_episodes": 100,
  "total_transitions": 45230,
  "avg_episode_length": 452.3,
  "avg_reward": -234.5,
  "total_violations": 23,
  "total_conflicts": 156,
  "airport": "KLAS",
  "max_aircraft": 10,
  "observation_info": { ... }
}
```

### Offline RL Dataset

For offline RL algorithms (Decision Transformer, CQL, etc.), use the `OfflineRLDataset`:

```python
from data.offline_dataset import OfflineDatasetCollector, OfflineRLDataset
from torch.utils.data import DataLoader

# Collect episodes
collector = OfflineDatasetCollector(env)
episodes = collector.collect_random_episodes(num_episodes=500)
episodes += collector.collect_heuristic_episodes(num_episodes=300)

# Create PyTorch dataset
dataset = OfflineRLDataset(
    episodes=episodes,
    context_len=20,      # Transformer context window
    max_aircraft=20,
    state_dim=14,        # Aircraft feature dimension
    scale_returns=True,  # Normalize returns
)

# Create dataloader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Training loop
for batch in dataloader:
    returns = batch["returns"]           # (B, T, 1)
    states = batch["states"]             # (B, T, N, 14)
    actions = batch["actions"]           # Dict of (B, T)
    timesteps = batch["timesteps"]       # (B, T)
    attention_mask = batch["attention_mask"]  # (B, T)
    # ... train model
```

The dataset automatically handles:
- Variable-length episodes (padding/truncating to context_len)
- Return scaling (for training stability)
- Attention masks (for valid timesteps)
- Batching with collation

### Data Collection Policies

**Random Policy:**
- Samples actions uniformly from action space
- Useful for exploration and baseline dataset

**Heuristic Policy:**
- Simple rule-based controller
- Resolves conflicts by altitude/heading changes
- 70% no-op rate (avoids over-controlling)

**Expert Policy:** (Coming soon)
- Advanced rule-based controller
- Implements ATC best practices
- For behavioral cloning

**Custom Policy:**
```python
def my_policy(obs):
    # Your policy logic
    return action

episodes = collector.collect_custom_policy_episodes(
    policy_fn=my_policy,
    num_episodes=100,
)
```

## 🔧 Setup

1. **Install dependencies**:
   ```bash
   uv sync
   uv run playwright install chromium
   ```

2. **Start OpenScope server** (in parent directory):
   ```bash
   cd ../openscope
   npm install
   npm run build
   npm run start  # Must be running at localhost:3003
   ```

3. **Run the demo**:
   ```bash
   jupyter notebook openscope_async_demo.ipynb
   ```

## 🔧 Hyperparameter Optimization

Use Optuna-based hyperparameter tuning for systematic optimization:

```python
from training.hyperparam_tuner import HyperparamTuner

def env_factory(n_envs=1):
    from stable_baselines3.common.vec_env import DummyVecEnv
    from environment import PlaywrightEnv
    return DummyVecEnv([lambda: PlaywrightEnv(airport="KLAS", max_aircraft=10)] * n_envs)

def model_factory(env, hyperparams):
    from stable_baselines3 import PPO
    return PPO("MultiInputPolicy", env, **hyperparams)

tuner = HyperparamTuner(
    algo="ppo",
    env_factory=env_factory,
    model_factory=model_factory,
    n_trials=50,
    n_timesteps=100000,
    sampler="tpe",  # Tree-structured Parzen Estimator
    pruner="halving",  # Successive Halving
)

results = tuner.optimize()
print(results.head())  # View best hyperparameters
```

## 🧠 Training Approaches

This project implements multiple RL approaches for comparison:

### 1. Baseline PPO ⭐
- Standard Proximal Policy Optimization
- Action masking for invalid actions
- Vectorized environments (8+ parallel)
- Curriculum learning (2→4→6→10 aircraft)
- **Expected**: 60-75% success rate after 500k steps
- **Notebook**: `notebooks/01_baseline_ppo_demo.ipynb`

### 2. Hierarchical RL
- Two-level policy: aircraft selection → command generation
- Reduced action space complexity
- Interpretable decision making
- Attention-based aircraft selection
- **Notebook**: `notebooks/02_hierarchical_rl_demo.ipynb`

### 3. Behavioral Cloning + RL
- Pre-train on expert demonstrations
- Fine-tune with PPO
- Faster convergence than pure RL
- **Notebook**: `notebooks/03_behavioral_cloning_demo.ipynb`

### 4. Multi-Agent RL
- Each aircraft as independent agent
- Shared policy with communication
- MAPPO implementation
- Emergent coordination behaviors
- **Notebook**: `notebooks/04_multi_agent_demo.ipynb`

### 5. Decision Transformer ⭐⭐
- Offline RL via sequence modeling
- Return-to-go conditioning
- Learn from fixed dataset (no online interaction)
- 5-10x sample efficiency vs PPO
- **Notebook**: `notebooks/05_decision_transformer_demo.ipynb`

### 6. Trajectory Transformer
- Unified world model + policy
- Predicts full trajectories
- Multi-step planning via beam search
- **Notebook**: `notebooks/06_trajectory_transformer_demo.ipynb`

### 7. Cosmos World Model 🚀
- Fine-tune NVIDIA Cosmos on OpenScope
- Train RL in learned world model
- 10-100x faster training potential
- Requires GPU cluster
- **Notebook**: `notebooks/07_cosmos_world_model_demo.ipynb`

### 8. Simple World Model
- Lightweight learned dynamics model
- Model-based RL (Dreamer-style)
- Faster than model-free on complex tasks
- **Notebook**: `notebooks/08_simple_world_model_demo.ipynb`

### 9. Conservative Q-Learning (CQL)
- Offline RL with conservative value estimation
- Prevents over-estimation on out-of-distribution actions
- Works with suboptimal demonstrations
- **Notebook**: `notebooks/09_cql_offline_rl_demo.ipynb`

### 10. DreamerV3
- State-of-the-art model-based RL
- Recurrent world model with discrete latents
- Sample-efficient learning
- **Notebook**: `notebooks/10_dreamerv3_demo.ipynb`

### 11. TD-MPC 2 ⭐⭐
- Transformer-based world model for dynamics prediction
- Model Predictive Control (MPC) with Cross-Entropy Method
- Q-learning for long-term value estimation
- Combines short-term planning with long-term value
- Sample-efficient model-based RL
- **Notebook**: `notebooks/11_tdmpc2_demo.ipynb`


## 📚 Documentation

### Main Documentation
- `README.md` (this file) - Overview, quick start, and comprehensive guide
- `CLAUDE.md` - Development guide for Claude Code

### Notebooks
- `notebooks/00_explore_openscope_api.ipynb` - Environment exploration
- `notebooks/01-11_*.ipynb` - Approach-specific demos (see above)


## 🏗️ Architecture

The environment is built with a modular architecture for maintainability and extensibility:

### Core Components

**`environment/playwright_env.py`** - Main environment orchestrator
- Manages browser lifecycle
- Coordinates all components
- Implements Gymnasium interface

**`environment/game_interface.py`** - Game communication layer
- JavaScript evaluation in browser
- Command execution
- State extraction

**`environment/state_processor.py`** - State processing
- Converts raw game state to observations
- Normalizes values for RL
- Handles variable aircraft count

**`environment/reward_calculator.py`** - Reward computation
- Multiple reward strategies
- Conflict detection
- Episode scoring

**`environment/spaces.py`** - Space definitions
- Observation space (Dict)
- Action space (Dict)
- Validation utilities

**`environment/metrics.py`** - Episode tracking
- Command statistics
- Conflict/violation counts
- Performance metrics

### Data Collection Components

**`data/training_data_collector.py`** - Training data collection
- Episode collection from policies
- Observation/action recording
- Returns-to-go computation
- Dataset export

**`data/offline_dataset.py`** - PyTorch datasets
- Context window sampling
- Batch collation
- Episode loading/saving

**`data/cosmos_collector.py`** - Cosmos-specific data
- Video frame collection
- Action annotation
- World model training data

### Model Components

**`models/networks.py`** - Neural network architectures
- ATCActorCritic (PPO)
- Transformer encoders
- Policy/value heads

**`models/decision_transformer.py`** - Decision Transformer
- GPT-style architecture
- Return-to-go conditioning
- Offline RL training

**`models/trajectory_transformer.py`** - Trajectory Transformer
- Full trajectory modeling
- Beam search planning

**`models/dynamics_model.py`** - World models
- State transition prediction
- Reward prediction
- Model-based RL support

### Training Components

**`training/ppo_trainer.py`** - PPO training
- Vectorized environments
- GAE computation
- Policy optimization

**`training/hyperparam_tuner.py`** - Hyperparameter optimization
- Optuna-based systematic tuning
- Algorithm-specific samplers (PPO, Decision Transformer)
- Pruning support (SuccessiveHalving, Median)
- Periodic evaluation during training
- Proper error handling and resource cleanup

**`training/config_logger.py`** - Configuration tracking
- YAML-based config saving/loading
- Git hash, branch, Python version tracking
- Dependency version tracking
- Reproducibility support

**`training/dt_trainer.py`** - Decision Transformer training
- Sequence modeling
- Return conditioning
- Offline learning

**`training/world_model_trainer.py`** - World model training
- Dynamics learning
- Reward modeling

**`training/cql_trainer.py`** - CQL training
- Conservative Q-learning
- Offline RL with Q-functions

## 🎯 Use Cases

### 1. Research
- Compare RL algorithms on realistic ATC task
- Study multi-agent coordination
- Explore world model learning
- Benchmark offline RL methods

### 2. Education
- Learn RL concepts with real-world domain
- Understand air traffic control
- Experiment with different approaches
- Visualize learning progress

### 3. Production
- Develop ATC assistance tools
- Pre-train policies for real systems
- Generate training data for human controllers
- Test ATC procedures

### 4. Data Collection
- Collect expert demonstrations
- Generate diverse trajectories
- Create offline RL datasets
- Build world model training data

## 📊 Benchmarking

Standard benchmark scenarios are defined in `experiments/benchmark.py`:

1. **Simple Arrivals**: 2-3 aircraft, wide separation
2. **Mixed Traffic**: 5-7 aircraft, arrivals + departures
3. **Dense Traffic**: 10+ aircraft, high conflict probability
4. **Conflict Resolution**: Pre-configured conflict scenarios
5. **Major Airport Peak**: 15+ aircraft at KJFK

Compare approaches using:
```python
from experiments.benchmark import run_benchmark

results = run_benchmark(
    agent=your_agent,
    scenarios=["simple_arrivals", "mixed_traffic", "dense_traffic"],
    num_episodes=100,
)
```

**Performance Benchmarking:**
```python
from experiments.performance_benchmark import (
    benchmark_env_performance,
    benchmark_model_inference,
    run_all_benchmarks
)

# Benchmark environment FPS and throughput
env_metrics = benchmark_env_performance(env, n_steps=10000)

# Benchmark model inference speed
model_metrics = benchmark_model_inference(model, env, n_steps=1000)

# Run all benchmarks
all_metrics = run_all_benchmarks(env, model)
```

Metrics tracked:
- Success rate (% aircraft exited safely)
- Violation count
- Average reward
- Commands per aircraft
- Episode length
- Environment FPS and throughput
- Model inference speed

## 🤝 Contributing

Contributions are welcome! Here's how to get started:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/my-feature`
3. **Make your changes**:
   - Follow existing code style
   - Add type hints
   - Include docstrings
4. **Add tests**: Update `tests/` if applicable
5. **Test your changes**: `pytest tests/` or `uv run python scripts/smoke_test.py`
6. **Submit a pull request**

### Areas for Contribution

- New RL algorithms
- Improved reward functions
- Additional benchmarks
- Performance optimizations
- Documentation improvements
- Bug fixes

## 📞 Support & Resources

### Getting Help
- **Issues**: [GitHub Issues](https://github.com/yourusername/openscope-rl/issues)
- **Discussions**: Use GitHub Discussions for questions
- **Documentation**: Check `docs/` and notebooks
- **Examples**: Explore `poc/` for working demos

### Related Projects
- [OpenScope](https://github.com/openscope/openscope) - The ATC simulator
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) - RL algorithms
- [Gymnasium](https://gymnasium.farama.org/) - RL environment API
- [Playwright](https://playwright.dev/) - Browser automation

### Papers & References

**Decision Transformer:**
- Chen et al., "Decision Transformer: Reinforcement Learning via Sequence Modeling" (NeurIPS 2021)
- [arXiv:2106.01345](https://arxiv.org/abs/2106.01345)

**Trajectory Transformer:**
- Janner et al., "Offline Reinforcement Learning as One Big Sequence Modeling Problem" (NeurIPS 2021)
- [arXiv:2106.02039](https://arxiv.org/abs/2106.02039)

**NVIDIA Cosmos:**
- [NVIDIA Cosmos Documentation](https://docs.nvidia.com/cosmos/)
- [Cosmos Blog Post](https://blogs.nvidia.com/blog/cosmos-world-foundation-models/)

**PPO:**
- Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
- [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)

## 🏆 Acknowledgments

- **OpenScope Team** - For the excellent open-source ATC simulator
- **NVIDIA** - For Cosmos World Foundation Models
- **Stability AI** - For transformer architectures
- **OpenAI** - For PPO and baseline implementations
- **PyTorch Team** - For the deep learning framework
- **Gymnasium/Farama** - For the RL environment API

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🧪 Testing

Run tests to verify everything works:

```bash
# Run all tests
pytest tests/

# Run smoke tests (quick verification of entry points)
uv run python scripts/smoke_test.py

# Run specific test
pytest tests/test_data_extraction.py -v
```

**Note**: Some integration tests require OpenScope server running at localhost:3003.

## 🚀 What's Next?

1. **Run the exploration notebook**: `notebooks/00_explore_openscope_api.ipynb`
2. **Try a training demo**: Start with `notebooks/01_baseline_ppo_demo.ipynb`
3. **Collect data**: Use `data/training_data_collector.py`
4. **Explore approaches**: See training notebooks in `notebooks/` directory
5. **Benchmark your agent**: Use `experiments/benchmark.py`

Happy training! ✈️🎮