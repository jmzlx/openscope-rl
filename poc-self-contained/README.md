# Self-Contained RL ATC Demonstrations

**Train AI for Air Traffic Control - No External Dependencies!**

This directory contains completely self-contained Jupyter notebooks that demonstrate reinforcement learning for air traffic control without requiring the OpenScope game server or browser automation.

## Notebooks

### 1. `simple_2d_atc.ipynb` - Quick Start Demo ⭐
**Best for**: Learning RL basics, quick demonstrations, algorithm testing

- **Training Time**: 5-10 minutes to see learning
- **Complexity**: Low - 2D continuous space (no altitude)
- **Aircraft**: 3-5 aircraft at a time
- **Goal**: Route aircraft to exit points without collisions
- **Visualization**: Top-down view with aircraft trajectories
- **Perfect for**: First-time RL learners, quick demos, testing ideas

### 2. `realistic_3d_atc.ipynb` - Full ATC Simulation
**Best for**: Realistic demonstrations, research, advanced RL

- **Training Time**: 20-30 minutes to see learning
- **Complexity**: High - Full 3D space with realistic physics
- **Aircraft**: Up to 10 aircraft
- **Features**: Altitude control, multiple runways, landing procedures
- **Visualization**: Multiple views (top-down + altitude profile)
- **Perfect for**: Realistic ATC scenarios, advanced RL techniques

## Features

✅ **Zero external dependencies** - Pure Python simulation
✅ **Fast training** - See results in minutes, not hours
✅ **Visual feedback** - Watch the agent learn in real-time
✅ **Educational** - Clear code with extensive comments
✅ **Extensible** - Easy to modify and experiment
✅ **Complete** - Environment + training + evaluation

## Requirements

```bash
pip install gymnasium numpy matplotlib stable-baselines3
```

Or with uv (recommended):
```bash
uv pip install gymnasium numpy matplotlib stable-baselines3
```

## Quick Start

1. **Start with the simple version**:
   ```bash
   jupyter notebook simple_2d_atc.ipynb
   ```
   Run all cells and watch the agent learn in ~5 minutes!

2. **Try the realistic version**:
   ```bash
   jupyter notebook realistic_3d_atc.ipynb
   ```
   More complex but more impressive results!

## What You'll Learn

### RL Concepts Demonstrated
- **Observation spaces**: How to represent ATC state
- **Action spaces**: Discrete vs continuous control
- **Reward shaping**: Guiding agent behavior
- **PPO algorithm**: Stable policy gradient training
- **Curriculum learning**: Progressive difficulty
- **Evaluation**: Measuring agent performance

### ATC Concepts Demonstrated
- **Separation standards**: Minimum safe distances
- **Conflict detection**: Identifying potential collisions
- **Vectoring**: Directing aircraft with heading changes
- **Traffic flow**: Managing multiple aircraft efficiently

## Architecture Comparison

| Feature | Simple 2D | Realistic 3D |
|---------|-----------|--------------|
| Dimensions | 2D (x, y) | 3D (x, y, altitude) |
| Actions | Heading changes | Heading + Altitude + Speed |
| Physics | Simplified | Realistic (turn rates, climb rates) |
| Runways | 4 exit points | 2 intersecting runways (4 directions) |
| Max Aircraft | 5 | 10 |
| Training Time | 5 min | 30 min |
| Code Complexity | ~300 lines | ~600 lines |

## Extending the Demos

### Easy Extensions
- Add more aircraft
- Change separation standards
- Modify reward function
- Try different RL algorithms (SAC, A2C, DQN)

### Advanced Extensions
- Add wind effects
- Multiple aircraft types with different speeds
- Emergency scenarios
- Communication delays
- Weather constraints
- Fuel management

## Performance Tips

1. **Faster training**: Reduce `max_steps` and `episode_length`
2. **Better learning**: Tune reward coefficients
3. **Smoother physics**: Reduce `dt` (timestep)
4. **More aircraft**: Increase `max_aircraft` gradually

## Comparison with Main Project

These notebooks demonstrate the same RL concepts as the main OpenScope-RL project, but with simpler environments:

| Aspect | Main Project | These Notebooks |
|--------|--------------|-----------------|
| Environment | Real OpenScope game | Pure Python simulation |
| Realism | Very high | Medium-High |
| Setup | Complex (game server, browser) | Simple (just Python) |
| Speed | Slow (browser automation) | Fast (pure computation) |
| Use Case | Production training | Learning & experimentation |

## Contributing

Feel free to:
- Add new notebook variants
- Improve visualizations
- Add new RL algorithms
- Create challenge scenarios

## License

MIT License - Same as main project
