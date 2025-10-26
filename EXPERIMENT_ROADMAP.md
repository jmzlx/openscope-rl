# OpenScope RL: Multi-Approach Experiment Roadmap

**Goal**: Train AI agents to play OpenScope (air traffic control simulator) using multiple ML approaches in parallel, then compare results to find the best method.

## 🎯 Quick Start

1. **Check worktree setup**:
   ```bash
   ./scripts/list_worktrees.sh
   ```

2. **Start with priority approaches** (in order):
   - **Approach 1**: Baseline PPO (`.trees/01-baseline-ppo/`)
   - **Approach 5**: Decision Transformer (`.trees/05-decision-transformer/`)
   - **Approach 7**: Cosmos World Model (`.trees/07-cosmos-world-model/`)

3. **Each worktree has**:
   - `IMPLEMENTATION_GUIDE.md` - Detailed instructions
   - Isolated Git branch (`experiment/XX-name`)
   - All main repo code accessible

## 📊 Experiment Structure

### Worktrees (Isolated Development Environments)

```
.trees/
├── 01-baseline-ppo/          ⭐⭐⭐ Start here!
├── 02-hierarchical-rl/        ⭐⭐ Interpretability
├── 03-behavioral-cloning/     ⭐ Pre-training
├── 04-multi-agent/            Research
├── 05-decision-transformer/   ⭐⭐⭐ Transformer-based!
├── 06-trajectory-transformer/ Research
└── 07-cosmos-world-model/     ⭐⭐⭐ Game-changer!
```

### Shared Resources (Main Repo)

```
experiments/
├── benchmark.py         # Standard test scenarios
├── metrics.py           # Common metrics (success rate, violations)
└── visualization.py     # Plotting and visualization

environment/             # OpenScope Playwright integration
models/                  # Neural network architectures
poc/atc_rl/             # Working POC environments
```

## 🚀 Approach Summaries

### 1. Baseline PPO + Action Masking ⭐ PRIORITY 1
**Branch**: `experiment/01-baseline-ppo`

**What**: Standard PPO with action masking to prevent invalid actions
**Why**: Establish performance baseline, 2-3x sample efficiency improvement
**Timeline**: 2-3 days
**Expected**: 60-75% success rate after 500k steps

**Key Features**:
- Mask invalid aircraft IDs
- Mask ILS commands without runway
- Vectorized training (8+ parallel environments)
- Curriculum learning (2→4→6→10 aircraft)

**Go to**: `.trees/01-baseline-ppo/IMPLEMENTATION_GUIDE.md`

---

### 2. Hierarchical RL
**Branch**: `experiment/02-hierarchical-rl`

**What**: Two-level policy (aircraft selection → command generation)
**Why**: Interpretability, reduced action space complexity
**Timeline**: 3-4 days
**Expected**: Similar performance to baseline, much more interpretable

**Key Features**:
- High-level: Which aircraft to command
- Low-level: What command to issue
- Attention visualization
- Temporal abstraction

---

### 3. Behavioral Cloning + RL
**Branch**: `experiment/03-behavioral-cloning`

**What**: Pre-train on expert demonstrations, fine-tune with RL
**Why**: Faster convergence, safer initial policy
**Timeline**: 3-4 days
**Expected**: 2-3x faster learning than pure RL

**Key Features**:
- Rule-based expert controller
- Supervised pre-training
- RL fine-tuning
- Mixed BC+RL loss

---

### 4. Multi-Agent RL
**Branch**: `experiment/04-multi-agent`

**What**: Each aircraft as independent cooperative agent
**Why**: Natural formulation for ATC, emergent coordination
**Timeline**: 4-5 days
**Expected**: Interesting emergent behaviors, scalability

**Key Features**:
- Shared policy across agents
- Communication via attention
- MAPPO implementation
- Handles variable aircraft count

---

### 5. Decision Transformer ⭐ PRIORITY 2
**Branch**: `experiment/05-decision-transformer`

**What**: Offline RL via sequence modeling (return-to-go conditioning)
**Why**: Learn from fixed dataset, no online interaction needed!
**Timeline**: 3-4 days
**Expected**: 5-10x sample efficiency vs PPO

**Key Features**:
- Offline learning from recorded episodes
- Return conditioning (control behavior at test time)
- Leverage existing transformer architecture
- Works with mixed-quality data

**Go to**: `.trees/05-decision-transformer/IMPLEMENTATION_GUIDE.md`

---

### 6. Trajectory Transformer
**Branch**: `experiment/06-trajectory-transformer`

**What**: Model full trajectories (state→action→reward→state...)
**Why**: Unified world model + policy, planning capabilities
**Timeline**: 4-5 days
**Expected**: Better credit assignment, multi-step planning

**Key Features**:
- Single transformer predicts everything
- Beam search planning
- World model accuracy
- Multi-task learning

---

### 7. Cosmos World Model 🚀 ⭐ PRIORITY 3
**Branch**: `experiment/07-cosmos-world-model`

**What**: Fine-tune NVIDIA Cosmos on OpenScope, train policy in simulation
**Why**: 10-100x faster training! Revolutionary if it works!
**Timeline**: 5-7 days
**Expected**: Massive sample efficiency improvement

**Key Features**:
- World Foundation Model (Cosmos)
- Fine-tune on OpenScope gameplay videos
- Train RL in simulated environment
- Transfer to real OpenScope

**Hardware**: Perfect for your 2x DGX + RTX 5090!

**Go to**: `.trees/07-cosmos-world-model/IMPLEMENTATION_GUIDE.md`

---

## 📈 Evaluation Criteria

All approaches will be compared on:

### Primary Metrics
- **Success Rate**: % aircraft that exit successfully
- **Separation Violations**: Number of conflicts
- **Sample Efficiency**: Steps to reach 70% success rate
- **Training Time**: Hours to train

### Secondary Metrics
- **Throughput**: Aircraft per hour
- **Command Efficiency**: Commands per aircraft
- **Interpretability**: Can we understand decisions?
- **Robustness**: Performance on unseen scenarios

### Standard Benchmark Scenarios
(Defined in `experiments/benchmark.py`)

1. **Simple Arrivals**: 2-3 aircraft, wide separation
2. **Mixed Traffic**: 5-7 aircraft, arrivals + departures
3. **Dense Traffic**: 10+ aircraft, high conflict probability
4. **Conflict Resolution**: Pre-configured conflict scenarios
5. **Major Airport Peak**: 15+ aircraft at KJFK

## 🛠️ Workflow

### For Each Approach:

1. **Switch to worktree**:
   ```bash
   cd .trees/01-baseline-ppo
   ```

2. **Read implementation guide**:
   ```bash
   cat IMPLEMENTATION_GUIDE.md
   ```

3. **Implement**:
   - Create necessary files
   - Use shared utilities from `../experiments/`
   - Test incrementally

4. **Create demo notebook**:
   - `notebooks/XX_approach_demo.ipynb`
   - End-to-end runnable example
   - < 30 minutes to execute

5. **Commit your work**:
   ```bash
   git add .
   git commit -m "Implement [approach name]"
   git push origin experiment/01-baseline-ppo
   ```

6. **Document results**:
   - Update README in worktree
   - Record metrics
   - Save checkpoints

### Final Comparison:

Once multiple approaches are complete:

1. **Run benchmark suite** on all approaches
2. **Create comparison notebook**: `notebooks/00_approach_comparison.ipynb`
3. **Generate plots**: Sample efficiency, final performance, training time
4. **Pick winner(s)** for production use
5. **Merge successful approaches** to main branch

## 💻 Hardware Utilization

**Your Setup**:
- 2x NVIDIA DGX with NVLink
- 1x RTX 5090 (for development/inference)
- 1x MacBook Pro M4 Max 48GB (for notebooks/development)

**Optimal Usage**:
- **DGX**: Cosmos fine-tuning, large-scale PPO training
- **RTX 5090**: Development, Decision Transformer training
- **MacBook**: Notebooks, visualization, coordination

**Distributed Training**:
- PPO: 8-16 parallel environments per GPU
- Decision Transformer: Data-parallel across GPUs
- Cosmos: Model-parallel or data-parallel fine-tuning

## 📝 Progress Tracking

### Checklist:

- [x] Infrastructure setup (worktrees, scripts, utilities)
- [ ] Approach 1: Baseline PPO
  - [ ] Action masking implemented
  - [ ] Training script working
  - [ ] Demo notebook complete
  - [ ] Baseline metrics established
- [ ] Approach 5: Decision Transformer
  - [ ] Model architecture implemented
  - [ ] Offline dataset collected
  - [ ] Training working
  - [ ] Return conditioning demonstrated
- [ ] Approach 7: Cosmos Integration
  - [ ] Data collection pipeline
  - [ ] Cosmos fine-tuning
  - [ ] Simulated environment
  - [ ] RL training in simulation
- [ ] Final comparison and analysis
- [ ] Merge best approach(es) to main

### Target Timeline:

- **Week 1**: Baseline PPO + Decision Transformer
- **Week 2**: Cosmos integration
- **Week 3**: Additional approaches (hierarchical, etc.)
- **Week 4**: Final evaluation and comparison

## 🎓 Learning Resources

### Decision Transformer
- Paper: https://arxiv.org/abs/2106.01345
- Key idea: Conditioning on desired returns

### NVIDIA Cosmos
- Docs: https://docs.nvidia.com/cosmos/
- Blog: https://blogs.nvidia.com/blog/cosmos-world-foundation-models/
- Released: January 2025 (very new!)

### PPO Resources
- Your POC: `poc/atc_2d_demo.ipynb` (working example!)
- Stable-Baselines3 docs

## 🤝 Collaboration

### Git Workflow:
- Each approach on separate branch
- Worktrees allow parallel development
- Main branch stays clean
- Merge only proven approaches

### Sharing Results:
- Use `experiments/metrics.py` for consistent tracking
- WandB for experiment logging (entity: "jmzlx.ai")
- Notebooks for reproducibility

## 🚨 Important Notes

1. **OpenScope Server**: Must be running at `localhost:3003` for main environment
2. **Action Masking**: Critical for sample efficiency in all approaches
3. **Curriculum Learning**: Start simple (2-3 aircraft), increase difficulty
4. **Checkpointing**: Save frequently, training can be interrupted
5. **Evaluation**: Use standard benchmarks for fair comparison

## 🎯 Success Criteria

**Minimum Viable**:
- At least one approach achieves 70%+ success rate
- Baseline established for comparison
- Reproducible training pipeline

**Stretch Goals**:
- 85%+ success rate on complex scenarios
- 10x sample efficiency improvement (Cosmos or DT)
- Interpretable decision-making (hierarchical)
- Publication-worthy results

## 🆘 Troubleshooting

### Common Issues:

1. **Worktree not found**:
   ```bash
   ./scripts/setup_worktrees.sh
   ```

2. **OpenScope server not running**:
   ```bash
   cd ../openscope && npm start
   ```

3. **Module not found** (in worktree):
   - Worktrees share the main repo code
   - Ensure you're in correct worktree: `git worktree list`

4. **WandB not logging**:
   - Check entity/project names
   - Ensure `wandb login` was run

### Get Help:

- Check `IMPLEMENTATION_GUIDE.md` in each worktree
- Review working POC: `poc/atc_2d_demo.ipynb`
- Consult shared utilities: `experiments/`

---

## 🎉 Let's Build Something Amazing!

This experiment framework lets you explore 7 different ML approaches systematically. Start with the priority approaches (Baseline PPO, Decision Transformer, Cosmos), establish baselines, then expand to additional methods.

**Next Steps**:
1. Review this roadmap
2. Read `./trees/01-baseline-ppo/IMPLEMENTATION_GUIDE.md`
3. Start implementing!

Good luck! 🚀
