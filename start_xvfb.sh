#!/bin/bash
# Start Xvfb virtual display server if not already running
# This is needed for running OpenScope in non-headless mode on WSL/remote environments

DISPLAY_NUM=99

# Check if Xvfb is already running on this display
if pgrep -f "Xvfb :$DISPLAY_NUM" > /dev/null; then
    echo "✅ Xvfb already running on display :$DISPLAY_NUM"
else
    echo "🚀 Starting Xvfb on display :$DISPLAY_NUM..."
    Xvfb :$DISPLAY_NUM -screen 0 1024x768x24 > /dev/null 2>&1 &
    sleep 1
    
    if pgrep -f "Xvfb :$DISPLAY_NUM" > /dev/null; then
        echo "✅ Xvfb started successfully"
    else
        echo "❌ Failed to start Xvfb"
        exit 1
    fi
fi

echo "🖥️  DISPLAY=:$DISPLAY_NUM"
echo ""
echo "To use in Python:"
echo "  import os"
echo "  os.environ['DISPLAY'] = ':$DISPLAY_NUM'"

