# WSL Display Fix for OpenScope RL

## Problem

You're getting this error:
```
TargetClosedError: Looks like you launched a headed browser without having a XServer running.
Set either 'headless: true' or use 'xvfb-run <your-playwright-app>' before running Playwright.
```

**Why this happens:**
1. We set `headless: false` to fix the game loop (time wasn't advancing with `headless: true`)
2. WSL2 doesn't have a display server (X Server) by default
3. Playwright can't open a browser window without a display

## Solution Options

### Option 1: Use Xvfb with Cursor/Remote Notebooks (Recommended for Cursor Users)

If you're running notebooks through Cursor or another remote editor (not launching Jupyter yourself), you need xvfb running as a background service.

**Step 1: Install xvfb**
```bash
sudo apt-get update
sudo apt-get install -y xvfb
```

**Step 2: Start xvfb in the background**
```bash
cd ~/Projects/atc-ai/rl_training
./start_xvfb.sh
```

Or manually:
```bash
Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &
```

**Step 3: Run your notebook**
The notebook (cell 3) will automatically detect no DISPLAY and set `DISPLAY=:99`. Just run the cells normally in Cursor!

**That's it!** The browser will work properly in the background.

**Note:** Xvfb will stop when you close your terminal/WSL session. To make it permanent, add to your `~/.bashrc`:
```bash
# Auto-start xvfb if not running
if ! pgrep -f "Xvfb :99" > /dev/null; then
    Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &
fi
export DISPLAY=:99
```

---

### Option 2: Use Xvfb with Jupyter Lab (For command-line Jupyter users)

If you're launching Jupyter Lab yourself from the terminal:

**Step 1: Install xvfb**
```bash
sudo apt-get update
sudo apt-get install -y xvfb
```

**Step 2: Run Jupyter with xvfb**
```bash
cd ~/Projects/atc-ai/rl_training
xvfb-run -a jupyter lab
```

The `-a` flag automatically picks an available display number.

**That's it!** Now when you run the notebook cells, the browser will work properly.

---

### Option 3: Set up X11 Forwarding (If you want to see the browser)

This allows you to actually see the browser window from Windows.

**Step 1: Install VcXsrv on Windows**
- Download from: https://sourceforge.net/projects/vcxsrv/
- Install and run "XLaunch"
- Choose "Multiple windows", Display number: 0
- Start no client
- **IMPORTANT:** Check "Disable access control"

**Step 2: Set DISPLAY in WSL**
```bash
# Add to ~/.bashrc
export DISPLAY=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}'):0

# Reload bashrc
source ~/.bashrc
```

**Step 3: Test it**
```bash
# Install x11-apps if needed
sudo apt-get install x11-apps
xclock  # Should show a clock window on Windows
```

**Step 4: Run Jupyter**
```bash
cd ~/Projects/atc-ai/rl_training
jupyter lab
```

Now the Chrome browser window will appear on Windows!

---

### Option 3: Use Headless Mode with Manual Time Updates (Not Recommended)

We could try to manually drive the game loop, but this is fragile and complex. Not recommended.

---

## Recommended: Quick Start with Xvfb

### For Cursor/Remote Notebook Users
```bash
# One-time setup
sudo apt-get update && sudo apt-get install -y xvfb

# Start xvfb (stays running in background)
cd ~/Projects/atc-ai/rl_training
./start_xvfb.sh

# Now just run your notebook cells in Cursor - cell 3 will auto-configure DISPLAY
```

### For Command-line Jupyter Users
```bash
# One-time setup
sudo apt-get update && sudo apt-get install -y xvfb

# Every time you want to run the notebook
cd ~/Projects/atc-ai/rl_training
xvfb-run -a jupyter lab
```

## Testing

After setting up xvfb or X11 forwarding:

1. Restart your notebook kernel
2. Run cell 8 (create environment)
3. Run cell 10 (game state test)
4. You should see time advancing: `Step 18: Time= 180.0s (+180.0s), Aircraft=1`

## Alternative: Update Config Back to Headless

If you can't use xvfb or X11, you can revert the config and live with the time=0 issue:

```yaml
env:
  headless: true  # Revert to true
```

But this means the game simulation won't work properly for training.

## Summary

**For WSL users, the best solution is:**
```bash
sudo apt-get install -y xvfb
xvfb-run -a jupyter lab
```

This gives you the best of both worlds:
- ✅ Game loop works (time advances)
- ✅ No need for X server on Windows
- ✅ Fast and reliable

