#!/bin/bash
# Auto-restart wrapper for OpenScope RL training
# Automatically restarts training if it crashes (useful for aggressive timewarp settings)

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
CONFIG_FILE="${1:-config/ultra_fast_config.yaml}"
MAX_CRASHES="${2:-10}"
RESTART_DELAY="${3:-5}"

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}OpenScope RL Training - Auto-Restart Mode${NC}"
echo -e "${BLUE}============================================================${NC}"
echo -e "Config: ${GREEN}$CONFIG_FILE${NC}"
echo -e "Max crashes before stopping: ${YELLOW}$MAX_CRASHES${NC}"
echo -e "Restart delay: ${YELLOW}${RESTART_DELAY}s${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""
echo -e "${YELLOW}Press Ctrl+C twice quickly to stop completely${NC}"
echo ""

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
    echo -e "${GREEN}✓${NC} Virtual environment activated"
else
    echo -e "${RED}✗${NC} Virtual environment not found!"
    echo "Run ./setup.sh first"
    exit 1
fi

# Counter for crashes
CRASH_COUNT=0
RUN_COUNT=0

# Function to find latest checkpoint
find_latest_checkpoint() {
    LATEST=$(ls -t checkpoints/checkpoint_*.pt 2>/dev/null | head -1)
    if [ -n "$LATEST" ]; then
        echo "$LATEST"
    else
        echo ""
    fi
}

# Trap Ctrl+C
trap 'echo -e "\n${YELLOW}Received interrupt. Exiting gracefully...${NC}"; exit 0' INT

while [ $CRASH_COUNT -lt $MAX_CRASHES ]; do
    RUN_COUNT=$((RUN_COUNT + 1))
    
    echo -e "\n${BLUE}============================================================${NC}"
    echo -e "${GREEN}Starting training run #$RUN_COUNT${NC}"
    echo -e "${BLUE}============================================================${NC}"
    
    # Build command
    CMD="python train.py --config $CONFIG_FILE"
    
    # Check for existing checkpoint
    CHECKPOINT=$(find_latest_checkpoint)
    if [ -n "$CHECKPOINT" ] && [ $RUN_COUNT -gt 1 ]; then
        echo -e "${GREEN}✓${NC} Found checkpoint: ${YELLOW}$CHECKPOINT${NC}"
        echo -e "${GREEN}✓${NC} Resuming from checkpoint..."
        CMD="$CMD --checkpoint $CHECKPOINT"
    else
        echo -e "${BLUE}ℹ${NC} Starting fresh training..."
    fi
    
    # Record start time
    START_TIME=$(date +%s)
    
    # Run training
    if $CMD; then
        # Training completed successfully (reached total_timesteps)
        echo -e "\n${GREEN}============================================================${NC}"
        echo -e "${GREEN}✓ Training completed successfully!${NC}"
        echo -e "${GREEN}============================================================${NC}"
        exit 0
    else
        # Training crashed or was interrupted
        EXIT_CODE=$?
        END_TIME=$(date +%s)
        RUNTIME=$((END_TIME - START_TIME))
        
        CRASH_COUNT=$((CRASH_COUNT + 1))
        
        echo -e "\n${RED}============================================================${NC}"
        echo -e "${RED}✗ Training crashed! (exit code: $EXIT_CODE)${NC}"
        echo -e "${RED}Runtime before crash: ${RUNTIME}s${NC}"
        echo -e "${RED}Crash count: $CRASH_COUNT / $MAX_CRASHES${NC}"
        echo -e "${RED}============================================================${NC}"
        
        # Check if we should continue
        if [ $CRASH_COUNT -ge $MAX_CRASHES ]; then
            echo -e "\n${RED}Maximum crash count reached. Stopping.${NC}"
            echo -e "${YELLOW}Consider:${NC}"
            echo -e "  - Reducing timewarp in config"
            echo -e "  - Increasing action_interval"
            echo -e "  - Using a more stable configuration"
            exit 1
        fi
        
        # Wait before restart
        echo -e "\n${YELLOW}Restarting in ${RESTART_DELAY} seconds...${NC}"
        echo -e "${YELLOW}(Press Ctrl+C now to stop)${NC}"
        
        for i in $(seq $RESTART_DELAY -1 1); do
            echo -ne "\r${YELLOW}$i...${NC}"
            sleep 1
        done
        echo -e "\r${GREEN}Restarting now!${NC}"
    fi
done

echo -e "\n${RED}Training stopped after $CRASH_COUNT crashes.${NC}"
exit 1

