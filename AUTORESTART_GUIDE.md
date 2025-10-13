# Auto-Restart Training Guide

The auto-restart wrapper automatically restarts training if it crashes, which is useful when using aggressive timewarp settings.

## Quick Start

### Basic Usage (Ultra-Fast Config)
```bash
cd /home/jmzlx/Projects/openscope/rl_training
./train_with_autorestart.sh
```

This will:
- ✅ Use `config/ultra_fast_config.yaml` by default
- ✅ Automatically restart on crashes
- ✅ Resume from latest checkpoint
- ✅ Stop after 10 crashes (prevents infinite loops)

### Custom Config
```bash
./train_with_autorestart.sh config/test_config.yaml
```

### Advanced Options
```bash
./train_with_autorestart.sh [config] [max_crashes] [restart_delay]

# Examples:
./train_with_autorestart.sh config/ultra_fast_config.yaml 20 3
#                            └─ config file            └─ max   └─ delay (seconds)
#                                                         crashes
```

## How It Works

```
┌─────────────────────────────────────────────────────────┐
│ Start Training                                          │
├─────────────────────────────────────────────────────────┤
│ Run 1: Training... → Crash!                            │
│   ↓ Wait 5 seconds...                                   │
│ Run 2: Resume from checkpoint... → Crash!              │
│   ↓ Wait 5 seconds...                                   │
│ Run 3: Resume from checkpoint... → Success!            │
└─────────────────────────────────────────────────────────┘
```

## Features

✅ **Automatic Checkpoint Resume**: Always resumes from latest checkpoint
✅ **Crash Counter**: Stops after N crashes to prevent infinite loops  
✅ **Colored Output**: Easy to see what's happening
✅ **Graceful Exit**: Ctrl+C to stop anytime
✅ **Runtime Tracking**: Shows how long each run lasted

## Example Output

```bash
$ ./train_with_autorestart.sh

============================================================
OpenScope RL Training - Auto-Restart Mode
============================================================
Config: config/ultra_fast_config.yaml
Max crashes before stopping: 10
Restart delay: 5s
============================================================

✓ Virtual environment activated

============================================================
Starting training run #1
============================================================
ℹ Starting fresh training...

Using device: cuda
Creating environment...
Creating agent...
Network parameters: 7,339,053

Starting training for 50000 timesteps...
  5%|███▌                | 2560/50000 [12:30<3:45:22, 3.51it/s]

============================================================
✗ Training crashed! (exit code: 1)
Runtime before crash: 750s
Crash count: 1 / 10
============================================================

Restarting in 5 seconds...
(Press Ctrl+C now to stop)
Restarting now!

============================================================
Starting training run #2
============================================================
✓ Found checkpoint: checkpoints/checkpoint_2500.pt
✓ Resuming from checkpoint...

[... continues automatically ...]
```

## Stopping Training

### Normal Stop
Press **Ctrl+C** once to stop gracefully after current episode

### Force Stop
Press **Ctrl+C** twice quickly to stop immediately

## Common Scenarios

### Scenario 1: Overnight Training (Aggressive Speed)
```bash
# Run ultra-fast config, allow up to 20 crashes
./train_with_autorestart.sh config/ultra_fast_config.yaml 20 5

# Leave it running overnight
# It will handle crashes automatically
```

### Scenario 2: Testing Stability
```bash
# See if a config is stable enough
./train_with_autorestart.sh config/my_test_config.yaml 3 10

# If it crashes 3 times, the config is too aggressive
```

### Scenario 3: Production Training
```bash
# Use stable config, but still protect against rare crashes
./train_with_autorestart.sh config/training_config.yaml 5 30

# Conservative settings, 30s delay between restarts
```

## Troubleshooting

### "Virtual environment not found"
```bash
# Run setup first
./setup.sh
source venv/bin/activate
```

### "Permission denied"
```bash
# Make script executable
chmod +x train_with_autorestart.sh
```

### Too Many Crashes
If you hit the crash limit quickly, your config is too aggressive:

1. **Reduce timewarp**: `30 → 20 → 10`
2. **Increase action_interval**: `2 → 3 → 5`
3. **Check browser**: Make sure game server is running
4. **Check logs**: Look at error messages in logs/

### Crashes Always at Same Point
If it crashes at the same step count repeatedly:
- Might be a specific game state causing issues
- Try different `seed` in config
- Try different airport

## Monitoring While Running

### Watch Progress
```bash
# In another terminal
tail -f logs/training_log_*.jsonl
```

### Check Checkpoints
```bash
# See saved checkpoints
ls -lth checkpoints/
```

### GPU Usage
```bash
# Monitor GPU (if available)
python monitor_gpu.py
```

## Tips for Maximum Speed

1. **Start conservative, increase gradually**
   - Test with timewarp 10, 20, 30, 40...
   - Find your stable maximum

2. **Use auto-restart for aggressive settings**
   - If crash rate < 1 per hour: Good!
   - If crash rate > 3 per hour: Too aggressive

3. **Monitor first hour**
   - If stable for 1 hour, likely stable indefinitely
   - If crashes in first 10 minutes, reduce timewarp

4. **Checkpoints save progress**
   - Set `save_interval: 2500` for frequent saves
   - Won't lose much progress on crashes

## Performance Expectations

| Config | Speed | Stability | Crash Rate | Use Case |
|--------|-------|-----------|------------|----------|
| Conservative | 1-2 steps/s | ★★★★★ | ~0/hr | Final training |
| Balanced | 3-5 steps/s | ★★★★☆ | ~0-1/hr | Development |
| Aggressive | 6-10 steps/s | ★★★☆☆ | ~2-3/hr | Rapid testing |
| Extreme | 10-15 steps/s | ★★☆☆☆ | ~5-10/hr | Quick experiments |

## When NOT to Use Auto-Restart

❌ **Debugging**: If you need to see exact error messages
❌ **First Run**: Test config manually first to ensure it works
❌ **Production**: Use stable config without auto-restart for important runs

## Best Practices

✅ **Test first**: Run config manually for 10 minutes before using auto-restart
✅ **Monitor logs**: Check that crashes are environmental, not bugs
✅ **Set reasonable limits**: `max_crashes: 10-20` is usually good
✅ **Use with aggressive configs**: Perfect for timewarp > 20

---

Happy (crash-resistant) training! 🚀

