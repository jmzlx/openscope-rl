# OpenScope RL Training System

A complete reinforcement learning training system for teaching AI agents to play the openScope air traffic control simulator using PyTorch and PPO (Proximal Policy Optimization).

## Overview

This system trains a Transformer-based neural network to control air traffic in the openScope simulator. The agent learns to:
- Manage multiple aircraft simultaneously
- Maintain safe separation standards
- Guide arrivals to successful landings
- Clear departures efficiently
- Avoid collisions and airspace violations

## Architecture

### Neural Network
- **Transformer Encoder**: Processes variable numbers of aircraft using self-attention
- **Attention Pooling**: Aggregates aircraft features into fixed-size representation
- **Actor-Critic**: Separate policy and value networks sharing the encoder
- **Hierarchical Actions**: Selects aircraft, command type, and parameters

### Training Algorithm
- **PPO (Proximal Policy Optimization)**: Stable policy gradient method
- **GAE (Generalized Advantage Estimation)**: For variance reduction
- **Curriculum Learning**: Gradually increases difficulty
- **Reward Shaping**: Dense rewards to guide learning

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for fast, reliable dependency management.

1. Install uv (if not already installed):
```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with pip
pip install uv
```

2. Install dependencies:
```bash
cd rl_training
uv sync
```

3. Install Playwright browsers (for game automation):
```bash
uv run playwright install chromium
```

**Note**: `uv sync` automatically creates a virtual environment at `.venv` and installs all dependencies from `pyproject.toml`.

3. Start the openScope game server:
```bash
cd ..
npm install
npm run build
npm run start
```

The game should be running at http://localhost:3003

## Quick Start

### Training

Basic training with default settings:
```bash
uv run train_sb3.py
```

Training with custom config and parallel environments:
```bash
uv run train_sb3.py --config config/training_config.yaml --n-envs 4 --wandb
```

Resume from checkpoint:
```bash
uv run train_sb3.py --checkpoint checkpoints/checkpoint_50000_steps.zip --n-envs 4
```

### Evaluation

Evaluate a trained model:
```bash
uv run evaluate_sb3.py --checkpoint checkpoints/checkpoint_final.zip --n-episodes 20
```

With visualization (non-headless):
```bash
uv run evaluate_sb3.py --checkpoint checkpoints/checkpoint_final.zip --render
```

### Visualize Training Progress

TensorBoard is automatically integrated. View training metrics:
```bash
uv run tensorboard --logdir logs/
```

## Configuration

Edit `config/training_config.yaml` to customize:

### Environment Settings
```yaml
env:
  airport: "KLAS"          # Airport to train on
  timewarp: 5              # Game speed multiplier (1-10)
  max_aircraft: 20         # Maximum aircraft in scenario
  episode_length: 3600     # Episode duration (game seconds)
  action_interval: 5       # Command interval (seconds)
```

### PPO Hyperparameters
```yaml
ppo:
  learning_rate: 3.0e-4    # Adam learning rate
  gamma: 0.99              # Discount factor
  gae_lambda: 0.95         # GAE lambda
  clip_epsilon: 0.2        # PPO clip parameter
  n_steps: 2048            # Steps per update
  n_epochs: 10             # Optimization epochs
  batch_size: 64           # Minibatch size
```

### Network Architecture
```yaml
network:
  hidden_dim: 256                  # Transformer hidden dimension
  num_attention_heads: 8           # Multi-head attention
  num_transformer_layers: 4        # Encoder depth
```

### Curriculum Learning
```yaml
curriculum:
  enabled: true
  stages:
    - name: "single_arrival"
      max_aircraft: 1
      episodes: 1000
    - name: "moderate_traffic"
      max_aircraft: 8
      episodes: 3000
    # ...
```

## Project Structure

```
rl_training/
├── config/
│   └── training_config.yaml       # Training configuration
├── environment/
│   └── openscope_env.py          # Gymnasium environment wrapper
├── models/
│   └── networks.py               # Neural network architectures
├── algorithms/
│   └── ppo.py                    # PPO implementation
├── utils/
│   ├── logger.py                 # Training logger
│   └── curriculum.py             # Curriculum manager
├── train.py                      # Main training script
├── evaluate.py                   # Evaluation script
├── visualize_training.py         # Visualization script
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## How It Works

### 1. Environment Interface

The `OpenScopeEnv` class wraps the openScope game using Playwright for browser automation:
- Executes commands by simulating keyboard input
- Extracts game state via JavaScript injection
- Converts state to structured observations
- Computes rewards from game events and shaped rewards

### 2. Observation Space

The agent observes:
- **Per-aircraft features** (32 dims): position, altitude, heading, speed, phase, assignments
- **Aircraft mask**: Indicates valid aircraft (handles variable count)
- **Global state** (16 dims): time, aircraft count, conflicts, score
- **Conflict matrix**: Pairwise separation status

### 3. Action Space

Hierarchical discrete actions:
1. **Aircraft selection**: Which aircraft to command (0-19 or "no action")
2. **Command type**: altitude, heading, speed, ILS, or direct
3. **Parameters**: Target altitude/heading/speed indices

### 4. Reward Function

```python
reward = game_score_delta  # From openScope scoring
         - 0.01            # Small timestep penalty
         - 5.0 * violations # Extra penalty for violations
         - 1.0 * conflicts  # Warning for close calls
         + 0.05            # Bonus for safe operations
```

### 5. Training Loop

```
For each update:
  1. Collect n_steps of experience using current policy
  2. Compute advantages with GAE
  3. Update policy with PPO for n_epochs
  4. Log metrics and save checkpoints
```

## Tips for Training

### Hyperparameter Tuning

- **Learning rate**: Start with 3e-4, decrease if unstable
- **Timewarp**: Higher values (5-10) train faster but may be less stable
- **n_steps**: Larger values (2048-4096) provide better gradient estimates
- **clip_epsilon**: 0.1-0.3 works well, lower is more conservative

### Curriculum Design

Start simple and gradually increase:
1. Single aircraft, straight approaches
2. Multiple aircraft, no conflicts
3. Moderate traffic with some conflicts
4. Full complexity with all scenarios

### Monitoring Training

Watch for:
- **Policy loss**: Should decrease initially, then stabilize
- **Value loss**: Should decrease steadily
- **Entropy**: Should start high, gradually decrease
- **Mean reward**: Should trend upward
- **Clip fraction**: Around 0.1-0.3 indicates good step size

### Common Issues

**Agent does nothing**: Increase entropy coefficient or reduce clip_epsilon

**Training unstable**: Reduce learning rate, increase batch size

**Slow improvement**: Increase reward shaping, adjust curriculum

**Browser crashes**: Reduce timewarp, add delays in environment

## Advanced Usage

### Multi-Airport Training

Train on multiple airports by modifying the curriculum:
```yaml
curriculum:
  stages:
    - name: "KLAS_easy"
      airport: "KLAS"
      max_aircraft: 5
    - name: "KSEA_medium"
      airport: "KSEA"
      max_aircraft: 10
```

### Distributed Training

Use Ray for parallel environment rollouts:
```python
# Coming soon
```

### Transfer Learning

Fine-tune a pre-trained model on a new airport:
```bash
uv run train_sb3.py \
  --checkpoint checkpoints/klas_pretrained.zip \
  --config config/ksea_finetune.yaml \
  --n-envs 4
```

## Performance Benchmarks

Expected performance after full training (~10M steps):

| Metric | Random Agent | Trained Agent |
|--------|--------------|---------------|
| Mean Score | -500 | +150 |
| Success Rate | 10% | 85% |
| Collision Rate | 15% | <1% |
| Separation Loss | 30% | 5% |

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{openscope_rl,
  title={OpenScope Reinforcement Learning Training System},
  author={Your Name},
  year={2024},
  url={https://github.com/openscope/openscope}
}
```

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## Acknowledgments

- openScope team for the excellent ATC simulator
- OpenAI for PPO and baseline implementations
- PyTorch team for the deep learning framework

