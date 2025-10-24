# OpenScope RL Demo Environment

A demonstration environment for exploring air traffic control using the OpenScope ATC simulator.

## 🚀 Quick Start

Explore OpenScope interactively:

```bash
# Start OpenScope server (in parent directory)
cd ../openscope
npm install
npm run build
npm run start  # Must be running at localhost:3003

# Open the demo notebook
jupyter notebook openscope_async_demo.ipynb
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
│   │   ├── recorder.py       # Episode recording
│   │   ├── player.py         # Episode visualization
│   │   └── rendering.py      # Rendering utilities
│   └── *.ipynb               # Demo notebooks
└── openscope_async_demo.ipynb # Main demo notebook
```

## 🎯 Key Features

- **Modular Architecture**: Clean separation of concerns with dedicated modules
- **Interactive Demo**: Jupyter notebook for exploring OpenScope
- **Playwright Integration**: Browser automation for game interaction
- **Self-Contained POCs**: Complete ATC environments in the `poc/` directory
- **Configurable Components**: Dataclass-based configuration system
- **Multiple Reward Strategies**: Default, safety-focused, and efficiency-focused
- **Episode Recording**: Complete episode capture and visualization
- **Type Safety**: Comprehensive type hints throughout

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

## 📚 Documentation

- `docs/OPENSCOPE_SETUP_FINAL.md` - Detailed setup guide
- `docs/SCORE_AND_MESSAGES_GUIDE.md` - Understanding game scoring
- `docs/NOTEBOOK_INSTRUCTIONS.md` - Jupyter notebook usage
- `docs/WHERE_TO_SEE_COMMANDS.md` - Command reference

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

- **Issues**: Create an issue on GitHub
- **Documentation**: Check `docs/` directory for detailed usage
- **Examples**: See `poc/` directory for interactive demos

## 🏆 Acknowledgments

- OpenScope team for the excellent ATC simulator
- OpenAI for PPO and baseline implementations
- PyTorch team for the deep learning framework