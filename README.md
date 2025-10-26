# Multi-Agent RL for Air Traffic Control

**Experiment 04**: Multi-agent reinforcement learning where each aircraft is an independent cooperative agent.

## Overview

This experiment implements a multi-agent formulation of the ATC problem using MAPPO (Multi-Agent PPO) with centralized training and decentralized execution.

### Key Concept

Instead of a single agent controlling all aircraft sequentially, we treat **each aircraft as an independent agent** with a shared policy. Agents communicate through attention mechanisms and learn to cooperate for the team objective.

## Multi-Agent Formulation

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                 Multi-Agent ATC System                  │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Aircraft 1 ──┐                                        │
│  Aircraft 2 ──┤                                        │
│  Aircraft 3 ──┼─→ Shared Encoder ─→ Communication ─┐  │
│  Aircraft 4 ──┤                    (Attention)      │  │
│  Aircraft 5 ──┘                                     │  │
│                                                      │  │
│  ┌───────────────────────────────────────────────────┘  │
│  │                                                      │
│  ├─→ Decentralized Actors (each agent's action)        │
│  │   [Command, Altitude, Heading, Speed]               │
│  │                                                      │
│  └─→ Centralized Critic (team value)                   │
│      [Global state → Value estimate]                   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Components

1. **Shared Encoder**: All agents share a transformer encoder
2. **Communication Module**: Self-attention allows agents to exchange information
3. **Decentralized Actors**: Each agent outputs its own action independently
4. **Centralized Critic**: Value function sees global state (all agents)

### MAPPO (Multi-Agent PPO)

- **Centralized Training**: Critic sees global state during training
- **Decentralized Execution**: Actors only use local observations + communication
- **Shared Policy**: All agents share the same policy network (parameter efficiency)
- **Variable Agents**: Handles dynamic spawning/exiting of aircraft

## Project Structure

```
.trees/04-multi-agent/
├── models/
│   ├── multi_agent_policy.py    # Multi-agent architecture
│   ├── networks.py               # Original single-agent model
│   ├── encoders.py              # Shared encoders
│   ├── heads.py                 # Policy/value heads
│   └── config.py                # Network configuration
├── training/
│   ├── mappo_trainer.py         # MAPPO training algorithm
│   └── __init__.py
├── notebooks/
│   └── 04_multi_agent_demo.ipynb # Demonstration notebook
├── environment/                  # OpenScope environment
└── README.md                     # This file
```

## Quick Start

### 1. Install Dependencies

```bash
cd .trees/04-multi-agent
uv sync
uv run playwright install chromium
```

### 2. Start OpenScope Server

```bash
cd ../../../openscope  # Parent directory
npm install
npm run build
npm start  # Must be running at localhost:3003
```

### 3. Explore the Demo

```bash
jupyter notebook notebooks/04_multi_agent_demo.ipynb
```

## Usage

### Creating a Multi-Agent Policy

```python
from models import MultiAgentPolicy, create_default_network_config

# Create configuration
config = create_default_network_config(
    max_aircraft=10,
    hidden_dim=256,
    num_encoder_layers=4,
    num_attention_heads=8
)

# Create policy
policy = MultiAgentPolicy(config)

# Forward pass
obs = {
    "aircraft": torch.randn(2, 10, 14),
    "aircraft_mask": torch.ones(2, 10, dtype=torch.bool),
    "global_state": torch.randn(2, 4)
}

action_logits, value, comm_attention = policy(obs, return_communication=True)
```

### Training with MAPPO

```python
from training import MAPPOTrainer, MAPPOConfig
from environment import PlaywrightEnv

# Create environment
env = PlaywrightEnv(...)

# Training configuration
config = MAPPOConfig(
    max_aircraft=10,
    learning_rate=3e-4,
    total_timesteps=1_000_000,
    steps_per_rollout=2048,
    log_dir="logs/mappo",
    save_dir="checkpoints/mappo"
)

# Train
trainer = MAPPOTrainer(env, config)
trainer.train()
```

### Analyzing Communication

```python
# Get communication attention weights
with torch.no_grad():
    action_logits, value, comm_attn = policy(obs, return_communication=True)

# comm_attn is a list of attention weight tensors
# Each has shape: (batch_size, num_heads, num_agents, num_agents)

# Visualize communication (see notebook for full visualization code)
import matplotlib.pyplot as plt
import seaborn as sns

attn = comm_attn[0][0].mean(dim=0).cpu().numpy()  # Average over heads
sns.heatmap(attn, annot=True, fmt='.3f')
plt.title('Agent Communication Attention')
plt.show()
```

## Key Features

### 1. Communication via Attention

Agents communicate through multi-head self-attention:

```python
# CommunicationModule in multi_agent_policy.py
- Multi-layer self-attention between agents
- Each layer: attention → layer norm → FFN → layer norm
- Returns attention weights for visualization
```

### 2. Variable Number of Agents

Handles dynamic aircraft spawning/exiting:

```python
# aircraft_mask indicates active agents
obs = {
    "aircraft": torch.randn(batch, max_aircraft, features),
    "aircraft_mask": torch.tensor([True, True, False, ...])  # Only 2 active
}

# Policy automatically masks out inactive agents
action_logits, value, _ = policy(obs)
# action_logits shape: (batch, max_aircraft, action_dim)
# Only first 2 agents produce valid actions
```

### 3. Centralized Critic, Decentralized Actors

**Critic** (centralized - sees everything):
- All agent features (pooled via attention)
- Global state (time, aircraft count, conflicts, score)
- Outputs: Single team value estimate

**Actors** (decentralized - local view + communication):
- Own aircraft state
- Communicated information from other agents
- Outputs: Per-agent action distributions

### 4. Parameter Sharing

All agents share the same policy network:
- **Benefit**: Number of parameters independent of aircraft count
- **Benefit**: Sample efficiency (each timestep provides N agent samples)
- **Benefit**: Generalization to different numbers of aircraft

## Emergent Behaviors

After training, we observe emergent coordination patterns:

### 1. Spatial Clustering
Agents attending more to nearby aircraft (learned, not hardcoded)

### 2. Conflict Resolution
When potential conflicts exist, agents show increased attention to conflicting aircraft

### 3. Responsibility Sharing
Agents learn to divide tasks (e.g., some handle arrivals, others departures)

### 4. Communication Efficiency
Attention becomes more sparse and focused over training

## Advantages vs Single-Agent

| Aspect | Single-Agent | Multi-Agent |
|--------|-------------|-------------|
| **Scalability** | Poor (action space grows with aircraft) | Excellent (shared policy) |
| **Realism** | Sequential decisions | Parallel decisions |
| **Communication** | Implicit | Explicit (attention) |
| **Coordination** | Learned implicitly | Learned explicitly |
| **Sample Efficiency** | 1 sample per step | N samples per step |
| **Emergent Behavior** | Limited | Rich patterns |

## Challenges

### 1. Non-Stationarity
Each agent's environment changes as other agents learn
- **Solution**: Centralized training helps stabilize learning

### 2. Credit Assignment
Hard to determine which agent deserves credit/blame
- **Solution**: Per-agent advantages, global value baseline

### 3. Training Complexity
MAPPO more complex than standard PPO
- **Solution**: Careful hyperparameter tuning, curriculum learning

### 4. Communication Overhead
Attention over all agents is O(N²)
- **Solution**: Sparse attention, local neighborhoods (future work)

## Future Improvements

### 1. Graph Neural Networks
Replace full attention with graph convolutions:
- Define graph based on spatial proximity
- More efficient for large numbers of aircraft

### 2. Hierarchical Communication
Multi-level coordination:
- Local clusters (nearby aircraft)
- Global broadcast (critical events)

### 3. Curriculum Learning
Progressive training:
1. Start with 2-3 aircraft
2. Gradually increase to 10+
3. Add complexity (weather, failures)

### 4. Self-Play
Agents learn against versions of themselves:
- Robust to diverse behaviors
- Discovers novel strategies

### 5. Meta-Learning
Learn to adapt to different airports/traffic patterns:
- Few-shot adaptation to new scenarios
- Transfer learning across environments

## Performance Metrics

Track these metrics during training:

### Agent-Level
- **Per-agent entropy**: Exploration level
- **Per-agent advantage**: Credit assignment
- **Action distribution**: Strategy diversity

### Communication
- **Attention sparsity**: Communication efficiency
- **Attention consistency**: Learned patterns
- **Information flow**: Which agents communicate most

### Team-Level
- **Team reward**: Overall performance
- **Conflict rate**: Safety metric
- **Throughput**: Efficiency metric

## Visualization Tools

The demo notebook includes:

1. **Attention Heatmaps**: See which agents communicate
2. **Communication Graphs**: Network visualization of attention
3. **Coordination Analysis**: Statistical analysis of patterns
4. **Comparison Tools**: Single-agent vs multi-agent

## References

### Papers
- **MAPPO**: Yu et al. "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games" (2021)
- **CommNet**: Sukhbaatar et al. "Learning Multiagent Communication with Backpropagation" (2016)
- **QMIX**: Rashid et al. "QMIX: Monotonic Value Function Factorisation for DDQN" (2018)

### Implementations
- PyTorch MARL library
- OpenAI Multi-Agent Particle Environments
- SMAC (StarCraft Multi-Agent Challenge)

## Citation

If you use this work, please cite:

```bibtex
@software{openscope_marl_2025,
  title = {Multi-Agent Reinforcement Learning for Air Traffic Control},
  author = {OpenScope RL Team},
  year = {2025},
  url = {https://github.com/yourusername/openscope-rl}
}
```

## License

MIT License - See LICENSE file for details

## Support

- **Issues**: Create an issue on GitHub
- **Documentation**: See `notebooks/04_multi_agent_demo.ipynb`
- **Examples**: Run the demo notebook for interactive examples

## Acknowledgments

- OpenScope team for the ATC simulator
- PyTorch team for the framework
- MAPPO authors for the algorithm
- Multi-agent RL community for inspiration
