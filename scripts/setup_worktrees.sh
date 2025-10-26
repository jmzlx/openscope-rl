#!/bin/bash
# Set up all experimental worktrees
#
# Usage: ./scripts/setup_worktrees.sh

set -e

echo "🚀 Setting up all experimental worktrees..."
echo ""

# Define all approaches
APPROACHES=(
    "01-baseline-ppo"
    "02-hierarchical-rl"
    "03-behavioral-cloning"
    "04-multi-agent"
    "05-decision-transformer"
    "06-trajectory-transformer"
    "07-cosmos-world-model"
)

# Create each worktree
for approach in "${APPROACHES[@]}"; do
    echo "════════════════════════════════════════"
    echo "Creating worktree: $approach"
    echo "════════════════════════════════════════"
    ./scripts/create_worktree.sh "$approach" || echo "⚠️  Failed to create $approach (may already exist)"
    echo ""
done

echo "✅ All worktrees created!"
echo ""
echo "📋 List of worktrees:"
git worktree list
