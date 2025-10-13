# OpenScope RL Quick Start Guide

Get your AI agent training in 5 minutes!

## Prerequisites

- Python 3.8+
- Node.js 14+
- 16GB RAM minimum
- CUDA-capable GPU recommended (but not required)

## Step 1: Setup OpenScope Game

```bash
# Navigate to openScope root
cd /home/jmzlx/Projects/openscope

# Install dependencies
npm install

# Build the game
npm run build

# Start the server
npm run start
```

The game should now be running at http://localhost:3003

**Keep this terminal open!** The game server must stay running.

## Step 2: Setup RL Training Environment

Open a new terminal:

```bash
# Navigate to RL training directory
cd /home/jmzlx/Projects/openscope/rl_training

# Run automated setup
./setup.sh

# Activate virtual environment
source venv/bin/activate
```

## Step 3: Verify Setup

Test that everything works:

```bash
# Quick test (will connect to game and run for a few seconds)
python -c "
from environment.openscope_env import OpenScopeEnv
env = OpenScopeEnv(game_url='http://localhost:3003', headless=True)
obs, info = env.reset()
print('✓ Environment initialized successfully!')
env.close()
"
```

You should see: `✓ Environment initialized successfully!`

## Step 4: Start Training

### Option A: Quick Training Run (recommended for testing)

```bash
# Train with minimal settings for testing
python train.py \
  --config config/training_config.yaml \
  --device cuda
```

This will train for a while. You can stop anytime with `Ctrl+C`.

### Option B: Full Training Run

Edit `config/training_config.yaml` to your preferences, then:

```bash
# Full training with Weights & Biases logging
python train.py --wandb
```

### Option C: Use Example Script

```bash
# Pre-configured training run
./run_example.sh
```

## Step 5: Monitor Training

While training, watch for:

```
Training Progress:
  Step: 10240/10000000
  Episode: 125
  Mean Reward: 15.3
  Policy Loss: 0.0234
  Value Loss: 0.0156
```

**Good signs:**
- Mean reward increasing over time
- Policy loss decreasing initially
- No crashes or errors

**Red flags:**
- Constant negative rewards (check game is running)
- Loss = NaN (reduce learning rate)
- Browser crashes (reduce timewarp)

## Step 6: Checkpoints & Logs

Training automatically saves:

```
rl_training/
├── checkpoints/           # Model checkpoints every 10K steps
│   ├── checkpoint_10000.pt
│   ├── checkpoint_20000.pt
│   └── ...
└── logs/                  # Training logs
    └── training_log_*.jsonl
```

## Step 7: Evaluate Your Agent

After training for a while (e.g., 50K steps):

```bash
# Evaluate with 10 episodes
python evaluate.py \
  --checkpoint checkpoints/checkpoint_50000.pt \
  --n-episodes 10

# With visualization (non-headless browser)
python evaluate.py \
  --checkpoint checkpoints/checkpoint_50000.pt \
  --n-episodes 5 \
  --render
```

## Step 8: Visualize Training Progress

```bash
# Generate training plots
python visualize_training.py \
  --log-file logs/training_log_*.jsonl \
  --output-dir training_plots
```

Check `training_plots/` for graphs showing:
- Reward progression
- Loss curves
- Episode lengths
- Summary statistics

## Common Issues & Solutions

### Issue: "Connection refused"

**Problem:** Game server not running

**Solution:**
```bash
# In openScope root directory
npm run start
```

### Issue: "Playwright browser not found"

**Problem:** Chromium not installed

**Solution:**
```bash
playwright install chromium
```

### Issue: Training very slow

**Problem:** CPU-only or low timewarp

**Solutions:**
1. Use GPU: `--device cuda`
2. Increase timewarp in config: `timewarp: 10`
3. Reduce episode length: `episode_length: 1800`

### Issue: Agent does nothing

**Problem:** Poor exploration

**Solutions:**
1. Increase entropy coefficient: `entropy_coef: 0.02`
2. Check reward function is working
3. Start with simpler curriculum stage

### Issue: Browser keeps crashing

**Problem:** Timewarp too high or memory leak

**Solutions:**
1. Reduce timewarp: `timewarp: 3`
2. Add more delays in environment
3. Restart training periodically

## Training Tips

### For Quick Results (1-2 hours)
```yaml
# In config/training_config.yaml
env:
  timewarp: 10
  max_aircraft: 5
  episode_length: 1800

curriculum:
  enabled: false  # Skip curriculum
```

### For Best Performance (overnight)
```yaml
env:
  timewarp: 5
  max_aircraft: 20
  episode_length: 3600

curriculum:
  enabled: true
```

### For Debugging
```yaml
env:
  timewarp: 1
  headless: false  # See the game!
  max_aircraft: 1
```

## Next Steps

Once your agent is training well:

1. **Experiment with hyperparameters**
   - Try different learning rates
   - Adjust reward shaping
   - Modify network architecture

2. **Try different airports**
   - KSEA (Seattle)
   - KJFK (New York)
   - EGLL (London Heathrow)

3. **Advanced features**
   - Multi-airport training
   - Transfer learning
   - Imitation from expert demonstrations

4. **Share your results!**
   - Post on openScope Discord
   - Contribute improvements
   - Write about your findings

## Useful Commands Cheat Sheet

```bash
# Setup
./setup.sh && source venv/bin/activate

# Train
python train.py

# Resume training
python train.py --checkpoint checkpoints/checkpoint_50000.pt

# Evaluate
python evaluate.py --checkpoint checkpoints/checkpoint_50000.pt

# Visualize
python visualize_training.py --log-file logs/training_log_*.jsonl

# Clean up
rm -rf checkpoints/* logs/* training_plots/*
```

## Getting Help

- **README.md**: Comprehensive documentation
- **IMPLEMENTATION_SUMMARY.md**: Technical details
- **config/training_config.yaml**: All configuration options
- **GitHub Issues**: Report bugs
- **Discord**: Ask questions

---

**Ready to train?** Just run:

```bash
./setup.sh && source venv/bin/activate && python train.py
```

Good luck training your ATC AI! 🛫✈️🛬

