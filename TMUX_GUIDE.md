# Tmux Training Guide

Run your training in a detached session that survives terminal disconnection!

## 🚀 Quick Start

### Start Training in Tmux

```bash
cd /home/jmzlx/Projects/openscope/rl_training

# Option 1: Simple (recommended)
./train_tmux.sh

# Option 2: With specific config
./train_tmux.sh config/test_config.yaml

# Option 3: Without auto-restart
./train_tmux.sh config/ultra_fast_config.yaml no
```

This will:
1. Create a tmux session named `openscope-rl`
2. Start training with auto-restart
3. Attach you to the session

### Detach and Close Terminal

**While in the tmux session:**
1. Press `Ctrl+B` (release both keys)
2. Then press `D`

💡 Training continues running in the background!

### Reattach Later

```bash
tmux attach -t openscope-rl
```

Or use shorthand:
```bash
tmux a -t openscope-rl
```

---

## 📋 Essential Tmux Commands

### Session Management

```bash
# List all sessions
tmux ls

# Create new session
tmux new -s my-session-name

# Attach to session
tmux attach -t openscope-rl

# Kill session
tmux kill-session -t openscope-rl

# Kill all sessions
tmux kill-server
```

### Inside Tmux Session

**All commands start with Prefix: `Ctrl+B`**

| Action | Command |
|--------|---------|
| **Detach** | `Ctrl+B` then `D` |
| **New window** | `Ctrl+B` then `C` |
| **Next window** | `Ctrl+B` then `N` |
| **Previous window** | `Ctrl+B` then `P` |
| **List windows** | `Ctrl+B` then `W` |
| **Rename window** | `Ctrl+B` then `,` |
| **Split horizontal** | `Ctrl+B` then `"` |
| **Split vertical** | `Ctrl+B` then `%` |
| **Navigate panes** | `Ctrl+B` then arrow keys |
| **Scroll mode** | `Ctrl+B` then `[` (q to exit) |

---

## 🎯 Common Workflows

### Workflow 1: Start Training, Disconnect, Check Later

```bash
# Day 1 - Start training
cd /home/jmzlx/Projects/openscope/rl_training
./train_tmux.sh
# Training starts...
# Press Ctrl+B then D to detach
# Close terminal/laptop/disconnect from SSH

# Day 2 - Check progress
tmux attach -t openscope-rl
# View progress, logs, etc.
# Ctrl+B then D to detach again
```

### Workflow 2: Monitor Training + GPU + Logs

Create a multi-pane layout:

```bash
# Start training session
./train_tmux.sh

# Inside tmux, split into 3 panes:
# 1. Training (already running)
Ctrl+B then "    # Split horizontal (creates pane below)
Ctrl+B then %    # Split vertical (creates pane to right)

# Now you have 3 panes:
# ┌─────────────────┐
# │    Training     │  ← Pane 1 (training running)
# ├─────────┬───────┤
# │  Logs   │  GPU  │  ← Pane 2 & 3 (navigate with Ctrl+B arrows)
# └─────────┴───────┘

# In pane 2 (bottom left):
Ctrl+B then ↓    # Move to bottom-left pane
source venv/bin/activate
tail -f logs/training_log_*.jsonl

# In pane 3 (bottom right):
Ctrl+B then →    # Move to bottom-right pane
source venv/bin/activate
python monitor_gpu.py

# Now you see everything at once!
# Detach with Ctrl+B then D
```

### Workflow 3: Multiple Training Runs

```bash
# Start first training
SESSION_NAME=run1 ./train_tmux.sh config/ultra_fast_config.yaml

# Detach (Ctrl+B then D)

# Start second training
SESSION_NAME=run2 ./train_tmux.sh config/test_config.yaml

# List all sessions
tmux ls
# Output:
# run1: 1 windows (created Mon Oct 13 14:00:00 2025)
# run2: 1 windows (created Mon Oct 13 14:05:00 2025)

# Switch between them
tmux attach -t run1  # View first
tmux attach -t run2  # View second
```

---

## 💡 Pro Tips

### 1. **Auto-logging**

Enable tmux logging to save everything:

```bash
# Inside tmux session
Ctrl+B then :
set -g history-limit 50000
```

### 2. **Rename Session**

```bash
# Inside tmux
Ctrl+B then $
# Type new name
```

### 3. **Quick Session Switcher**

```bash
# Inside tmux
Ctrl+B then S
# Shows list of sessions, use arrows to select
```

### 4. **Mouse Support** (optional)

```bash
# Add to ~/.tmux.conf
echo "set -g mouse on" >> ~/.tmux.conf
tmux source-file ~/.tmux.conf

# Now you can:
# - Click panes to switch
# - Scroll with mouse wheel
# - Resize panes by dragging
```

### 5. **Persist Sessions Across Reboots**

Install tmux-resurrect (optional):
```bash
# Not essential for now, but useful for power users
```

---

## 🐛 Troubleshooting

### "No session found"

```bash
# Check if session exists
tmux ls

# If not listed, start new one
./train_tmux.sh
```

### Training Not Running After Reattach

```bash
# Check if process is still alive
tmux attach -t openscope-rl
# Look at the screen - should see training progress
# If frozen, might have crashed
```

### Can't Detach

- Make sure you press `Ctrl+B` **then release**, **then** press `D`
- Not `Ctrl+B+D` all at once!

### Lost in Scroll Mode

If screen is frozen and you can't type:
- Press `q` to exit scroll mode

### Multiple Tmux Sessions Consuming Resources

```bash
# List all sessions
tmux ls

# Kill specific session
tmux kill-session -t session-name

# Kill all sessions
tmux kill-server  # WARNING: Kills ALL sessions!
```

---

## 🎮 Your Current Setup

With the training you just started, here's what you can do:

### Stop Current Training and Restart in Tmux

```bash
# 1. Stop current training
Ctrl+C  # in the training terminal

# 2. Start in tmux
cd /home/jmzlx/Projects/openscope/rl_training
./train_tmux.sh config/ultra_fast_config.yaml

# 3. Detach when you want
Ctrl+B then D

# 4. Close terminal/laptop - training continues!

# 5. Later, check progress
tmux attach -t openscope-rl
```

---

## 📊 Monitoring in Tmux

### Quick Status Check (from outside tmux)

```bash
# Capture current screen content
tmux capture-pane -t openscope-rl -p | tail -20

# See last few lines of training
tmux capture-pane -t openscope-rl -p | grep -E "steps/sec|reward"
```

### Send Commands to Detached Session

```bash
# Not usually needed, but you can send keys to detached session
tmux send-keys -t openscope-rl "some command" Enter
```

---

## 🆚 Tmux vs Screen vs Nohup

| Feature | Tmux | Screen | Nohup |
|---------|------|--------|-------|
| Detach/Reattach | ✅ | ✅ | ❌ |
| Multiple panes | ✅ | ❌ | ❌ |
| Session management | ✅ | ✅ | ❌ |
| Scripting | ✅ | ⚠️ | ✅ |
| Modern/Maintained | ✅ | ⚠️ | ✅ |

**Verdict**: Tmux is best for interactive training monitoring!

---

## Quick Reference Card

```
START:    ./train_tmux.sh
DETACH:   Ctrl+B then D
ATTACH:   tmux attach -t openscope-rl
LIST:     tmux ls
KILL:     tmux kill-session -t openscope-rl

SPLIT:    Ctrl+B then " (horizontal) or % (vertical)
NAVIGATE: Ctrl+B then arrow keys
SCROLL:   Ctrl+B then [ (q to exit)
```

---

## Summary

✅ **Use tmux for long training runs**
✅ **Detach with Ctrl+B then D**
✅ **Close terminal freely - training continues**
✅ **Reattach anytime with `tmux attach`**
✅ **Perfect for overnight/multi-day training**

Happy detached training! 🚀

