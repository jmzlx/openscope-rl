#!/bin/bash
# Run OpenScope RL training in a detached tmux session

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Default values
SESSION_NAME="openscope-rl"
CONFIG="${1:-config/ultra_fast_config.yaml}"
USE_AUTORESTART="${2:-yes}"

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}OpenScope RL Training - Tmux Session Manager${NC}"
echo -e "${BLUE}============================================================${NC}"

# Check if session already exists
if tmux has-session -t $SESSION_NAME 2>/dev/null; then
    echo -e "${YELLOW}⚠  Session '$SESSION_NAME' already exists!${NC}"
    echo ""
    echo "Options:"
    echo "  1. Attach to existing session: tmux attach -t $SESSION_NAME"
    echo "  2. Kill and restart: tmux kill-session -t $SESSION_NAME && $0"
    echo "  3. Use different name: SESSION_NAME=myname $0"
    echo ""
    read -p "Kill existing session and start fresh? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Killing existing session...${NC}"
        tmux kill-session -t $SESSION_NAME
    else
        echo -e "${GREEN}Attaching to existing session...${NC}"
        tmux attach -t $SESSION_NAME
        exit 0
    fi
fi

echo -e "${GREEN}Creating new tmux session: $SESSION_NAME${NC}"
echo -e "Config: ${YELLOW}$CONFIG${NC}"
echo -e "Auto-restart: ${YELLOW}$USE_AUTORESTART${NC}"
echo ""

# Create tmux session
if [ "$USE_AUTORESTART" = "yes" ]; then
    # Use auto-restart wrapper
    tmux new-session -d -s $SESSION_NAME "cd /home/jmzlx/Projects/openscope/rl_training && ./train_with_autorestart.sh $CONFIG; read -p 'Training finished. Press enter to close...'"
else
    # Direct training
    tmux new-session -d -s $SESSION_NAME "cd /home/jmzlx/Projects/openscope/rl_training && source venv/bin/activate && python train.py --config $CONFIG; read -p 'Training finished. Press enter to close...'"
fi

echo -e "${GREEN}✓ Session created!${NC}"
echo ""
echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}Quick Commands:${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""
echo -e "${GREEN}Attach to session:${NC}"
echo -e "  tmux attach -t $SESSION_NAME"
echo ""
echo -e "${GREEN}Detach from session (while inside):${NC}"
echo -e "  Ctrl+B, then D"
echo ""
echo -e "${GREEN}List all sessions:${NC}"
echo -e "  tmux ls"
echo ""
echo -e "${GREEN}Kill session:${NC}"
echo -e "  tmux kill-session -t $SESSION_NAME"
echo ""
echo -e "${BLUE}============================================================${NC}"
echo ""
echo -e "${YELLOW}Attaching to session now...${NC}"
echo -e "${YELLOW}(Press Ctrl+B then D to detach and leave it running)${NC}"
sleep 2

# Attach to the session
tmux attach -t $SESSION_NAME

