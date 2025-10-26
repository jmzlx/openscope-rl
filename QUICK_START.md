# Quick Start Guide

## What We Built

✅ **Complete experimental framework** for exploring 7 different ML approaches to train OpenScope ATC agents

✅ **Git worktrees** for parallel development without conflicts

✅ **Shared utilities** for consistent benchmarking and metrics

✅ **Detailed implementation guides** for each approach

## Your Next Steps

### 1. Review the Roadmap (5 minutes)
```bash
cat EXPERIMENT_ROADMAP.md
```

This explains the overall structure and all 7 approaches.

### 2. Choose Your Starting Point

**Recommended Order** (by priority):

#### Option A: Start Simple (Recommended)
```bash
cd .trees/01-baseline-ppo
cat IMPLEMENTATION_GUIDE.md
```
- Establishes baseline performance
- Uses proven PPO algorithm
- Action masking for sample efficiency
- Expected: 2-3 days to complete

#### Option B: Transformer Approach
```bash
cd .trees/05-decision-transformer
cat IMPLEMENTATION_GUIDE.md
```
- Leverages your existing transformer code
- Offline learning (no environment interaction!)
- Return conditioning is very cool
- Expected: 3-4 days to complete

#### Option C: Revolutionary (High Risk/Reward)
```bash
cd .trees/07-cosmos-world-model
cat IMPLEMENTATION_GUIDE.md
```
- NVIDIA Cosmos integration (cutting-edge!)
- 10-100x sample efficiency if it works
- Perfect for your 2x DGX hardware
- Expected: 5-7 days to complete

### 3. Verify Setup

Check that all worktrees are created:
```bash
./scripts/list_worktrees.sh
```

You should see 8 worktrees (main + 7 experimental branches).

### 4. Test Shared Utilities

```python
# In a Python shell or notebook
from experiments import OpenScopeBenchmark, ATCMetrics, plot_comparison

# Create benchmark suite
benchmark = OpenScopeBenchmark()
scenarios = benchmark.get_scenarios()
print(f"Found {len(scenarios)} benchmark scenarios")

# Create metrics tracker
tracker = ATCMetrics()
print(tracker)
```

## File Structure

```
openscope-rl/
├── EXPERIMENT_ROADMAP.md          # ← Read this for overview
├── QUICK_START.md                 # ← You are here
│
├── .trees/                        # Experimental worktrees
│   ├── 01-baseline-ppo/          # ← Start here
│   │   └── IMPLEMENTATION_GUIDE.md
│   ├── 05-decision-transformer/  # ← Or here (transformer-based)
│   │   └── IMPLEMENTATION_GUIDE.md
│   └── 07-cosmos-world-model/    # ← Or here (revolutionary)
│       └── IMPLEMENTATION_GUIDE.md
│
├── experiments/                   # Shared utilities
│   ├── benchmark.py              # Standard test scenarios
│   ├── metrics.py                # Common metrics
│   └── visualization.py          # Plotting functions
│
├── scripts/                       # Worktree management
│   ├── setup_worktrees.sh        # Create all worktrees
│   ├── create_worktree.sh        # Create single worktree
│   └── cleanup_worktrees.sh      # Remove worktrees
│
├── environment/                   # Existing OpenScope env
├── models/                        # Existing neural networks
├── poc/atc_rl/                   # Working POC environments
└── notebooks/                     # Will be created in worktrees
```

## Quick Commands

### Worktree Management
```bash
# List all worktrees
./scripts/list_worktrees.sh

# Clean up worktrees (careful!)
./scripts/cleanup_worktrees.sh
```

### Working in a Worktree
```bash
# Switch to worktree
cd .trees/01-baseline-ppo

# Check current branch
git branch

# Make changes, commit
git add .
git commit -m "Implement action masking"
git push origin experiment/01-baseline-ppo

# Return to main repo
cd ../..
```

### Shared Utilities
All worktrees can access shared utilities:
```python
# In any worktree
from experiments.benchmark import OpenScopeBenchmark
from experiments.metrics import MetricsTracker
from experiments.visualization import plot_comparison
```

## Implementation Workflow

For each approach you implement:

1. **Read the guide**:
   ```bash
   cd .trees/01-baseline-ppo
   cat IMPLEMENTATION_GUIDE.md
   ```

2. **Implement step-by-step**:
   - Follow the guide's structure
   - Use existing code from main repo
   - Test incrementally

3. **Create demo notebook**:
   - End-to-end example
   - Runnable in < 30 minutes
   - Clear visualizations

4. **Commit your work**:
   ```bash
   git add .
   git commit -m "Descriptive message"
   git push origin experiment/XX-name
   ```

5. **Document results**:
   - Update README in worktree
   - Record key metrics
   - Save model checkpoints

## Expected Timeline

### Week 1: Establish Baselines
- Day 1-2: Baseline PPO implementation
- Day 3-4: Decision Transformer implementation
- Day 5: Initial evaluation and comparison

### Week 2: Advanced Approaches
- Day 6-8: Cosmos integration (if pursuing)
- Day 9-10: Hierarchical RL (if pursuing)
- Day 11-12: Additional approaches

### Week 3: Evaluation
- Day 13-14: Run full benchmark suite
- Day 15-16: Create comparison analysis
- Day 17-18: Documentation and writeup

### Week 4: Refinement
- Day 19-21: Optimize best approach(es)
- Day 22-23: Final experiments
- Day 24-25: Merge to main, production ready

## Success Metrics

**Minimum Success**:
- ✅ One approach achieves 70%+ success rate
- ✅ Baseline established
- ✅ Reproducible pipeline

**Great Success**:
- ✅ Multiple approaches working
- ✅ 85%+ success rate
- ✅ Clear winner identified
- ✅ Sample efficiency improvement demonstrated

**Amazing Success**:
- ✅ Cosmos approach works (10-100x efficiency!)
- ✅ Publication-worthy results
- ✅ Production-ready system
- ✅ Novel insights about ATC learning

## Troubleshooting

### "Module not found" in worktree
Worktrees share the main repo code. Make sure you're in the correct worktree:
```bash
git worktree list
pwd  # Verify current directory
```

### OpenScope server not running
The main PlaywrightEnv requires OpenScope server:
```bash
cd ../openscope
npm start
```

### GPU not detected
```bash
nvidia-smi  # Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"
```

### WandB not logging
```bash
wandb login
# Use entity: "jmzlx.ai"
```

## Resources

**Main Documentation**:
- `EXPERIMENT_ROADMAP.md` - Overall structure
- `.trees/XX-name/IMPLEMENTATION_GUIDE.md` - Detailed guides

**Working Examples**:
- `poc/atc_2d_demo.ipynb` - Working PPO implementation
- `environment/` - Existing OpenScope integration
- `models/` - Existing transformer architectures

**External Resources**:
- Decision Transformer paper: https://arxiv.org/abs/2106.01345
- NVIDIA Cosmos docs: https://docs.nvidia.com/cosmos/
- Stable-Baselines3: https://stable-baselines3.readthedocs.io/

## Questions?

1. Check `EXPERIMENT_ROADMAP.md` for high-level overview
2. Check `IMPLEMENTATION_GUIDE.md` in specific worktree
3. Review existing POC implementations
4. Check experiments/ utilities

---

## Ready to Start?

Choose your approach and dive in:

```bash
# Option 1: Baseline PPO (recommended first)
cd .trees/01-baseline-ppo && cat IMPLEMENTATION_GUIDE.md

# Option 2: Decision Transformer (transformer-based)
cd .trees/05-decision-transformer && cat IMPLEMENTATION_GUIDE.md

# Option 3: Cosmos (revolutionary)
cd .trees/07-cosmos-world-model && cat IMPLEMENTATION_GUIDE.md
```

**Good luck! 🚀**
