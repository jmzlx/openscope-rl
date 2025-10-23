# Quick Start Guide - Self-Contained ATC Demos

This directory contains **completely self-contained** RL demonstrations for air traffic control that don't require the OpenScope game server.

## What's Available

### 1. **Simple 2D ATC** (`simple_2d_atc.ipynb`) ⭐ Start Here!
- **Train in**: 5-10 minutes
- **Complexity**: Low
- **Perfect for**: Learning RL basics, quick demos
- **What it does**: Route aircraft through 2D airspace to exit points

### 2. **Realistic 3D ATC** (`realistic_3d_atc.ipynb`)
- **Train in**: 20-30 minutes
- **Complexity**: Medium-High
- **Perfect for**: Realistic demonstrations, research
- **What it does**: Full 3D ATC with altitude, runways, landing procedures

## Installation

```bash
# If you haven't already, install dependencies
uv sync

# Or with pip
pip install gymnasium numpy matplotlib stable-baselines3
```

## Running the Demos

### Option 1: Test the Environments (Fastest)

```bash
# Quick test to verify everything works
uv run python test_environments.py
```

You should see:
```
✅ Simple 2D ATC Environment: PASSED
✅ Realistic 3D ATC Environment: PASSED
```

### Option 2: Run in Jupyter (Recommended)

1. **Start Jupyter**:
   ```bash
   jupyter notebook
   ```

2. **Open `simple_2d_atc.ipynb`**

3. **Run all cells** (Cell → Run All)

4. **Watch it train!** You'll see:
   - Environment visualization
   - Random policy baseline
   - PPO training progress
   - Trained agent performance

### Option 3: Run in VS Code

1. Open the notebook in VS Code
2. Select Python kernel
3. Run all cells

## What to Expect

### Simple 2D ATC (`simple_2d_atc.ipynb`)

**Before training (random policy)**:
- 0-2 successful exits
- 5+ separation violations
- Total reward: -50 to -100

**After training (50k steps, ~5 min)**:
- 5+ successful exits
- 0-1 violations
- Total reward: +50 to +100

**What you'll see**:
- Top-down view of airspace
- Aircraft as blue arrows
- Separation circles (3nm radius)
- Training progress plots
- Real-time rendering

### Realistic 3D ATC (`realistic_3d_atc.ipynb`)

**Before training (random policy)**:
- Few or no successful landings
- Multiple separation violations
- Chaotic aircraft movements

**After training (100k steps, ~20-30 min)**:
- Multiple successful landings per episode
- Minimal violations
- Coordinated traffic flow
- Aircraft properly aligned with runways

**What you'll see**:
- Black canvas with white runways
- Yellow aircraft triangles with altitude labels
- Realistic turn rates and climb/descent
- Landing procedures
- Training metrics (rewards, landings, violations)

## Troubleshooting

### Import Errors
```bash
# Make sure you're in the right directory
cd poc-self-contained

# Install missing packages
uv add <package-name>
# or
pip install <package-name>
```

### Matplotlib Display Issues

If you see matplotlib errors when rendering:

**WSL/Linux without display**:
```bash
# Install virtual display
sudo apt-get install xvfb

# Start virtual display
Xvfb :99 -screen 0 1024x768x24 &
export DISPLAY=:99

# Then run notebook
```

**Jupyter in browser**:
```python
# Add to first cell
%matplotlib inline
```

**VS Code**:
- Select "Python Interactive" mode
- Or use `%matplotlib widget` for interactive plots

### Training is Slow

**Simple 2D**: Should take ~5-10 minutes for 50k steps
- If slower: Close other applications
- Try reducing `max_aircraft` or `max_steps`

**Realistic 3D**: Should take ~20-30 minutes for 100k steps
- If slower: Reduce `max_aircraft` to 4-5
- Reduce `episode_length` to 120

### Agent Not Learning

Common fixes:
1. **Train longer**: Try 2-3x more timesteps
2. **Check reward**: Make sure rewards are being collected
3. **Adjust hyperparameters**:
   - Increase `ent_coef` (more exploration)
   - Decrease `learning_rate` (more stable)
4. **Simplify environment**:
   - Reduce `max_aircraft`
   - Make rewards more generous

## Next Steps

After running the demos:

### Learn More About RL
- Modify reward functions
- Try different algorithms (SAC, A2C)
- Tune hyperparameters
- Add curriculum learning

### Extend the Environments
- Add more aircraft types
- Implement wind effects
- Create different airspace layouts
- Add fuel constraints
- Implement emergencies

### Use the Main Project
For production-grade training with the real OpenScope game:
1. Read `../CLAUDE.md` for full documentation
2. Start the OpenScope game server
3. Use the main training scripts with Playwright automation

## Comparison: Simple 2D vs Realistic 3D

| Feature | Simple 2D | Realistic 3D |
|---------|-----------|--------------|
| **Dimensions** | 2D (x, y) | 3D (x, y, altitude) |
| **Actions** | Heading only | Heading + Altitude + Speed + Land |
| **Physics** | Simplified | Realistic (turn/climb rates) |
| **Goal** | Exit airspace | Land on runways |
| **Training Time** | 5 min | 30 min |
| **Code Lines** | ~300 | ~600 |
| **Best For** | Learning RL | Realistic demos |

## Getting Help

1. **Check error messages** - They usually point to the issue
2. **Run test script first** - `uv run python test_environments.py`
3. **Start with Simple 2D** - Easier to debug
4. **Read the code** - All notebooks are well-commented

## File Structure

```
poc-self-contained/
├── README.md                    # Overview and comparison
├── QUICK_START.md              # This file
├── simple_2d_atc.ipynb         # Quick demo (5 min training)
├── realistic_3d_atc.ipynb      # Full 3D demo (30 min training)
├── test_environments.py        # Automated tests
└── utils/                      # (empty, for future extensions)
```

## Tips for Success

1. **Start simple**: Begin with `simple_2d_atc.ipynb`
2. **Watch the visualization**: Helps understand what's happening
3. **Monitor training plots**: Should see upward trend in rewards
4. **Be patient**: RL takes time to learn
5. **Experiment**: Change parameters and see what happens!

## Performance Expectations

**Your computer should handle this if it can**:
- Run Python and Jupyter
- Display matplotlib plots
- Has 4GB+ RAM

**Training speed**:
- Simple 2D: ~10,000 steps/minute
- Realistic 3D: ~3,000 steps/minute

**If slower**: This is normal! RL training is computationally intensive.

---

**Ready to start?** Run this command:

```bash
jupyter notebook simple_2d_atc.ipynb
```

Then **Run All Cells** and watch your first RL agent learn to control air traffic! 🎉
