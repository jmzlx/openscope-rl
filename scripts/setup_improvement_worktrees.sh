#!/bin/bash
# Set up all improvement worktrees for parallel development
#
# Usage: ./scripts/setup_improvement_worktrees.sh

set -e

echo "🚀 Setting up all improvement worktrees..."
echo ""

# Define all improvements
IMPROVEMENTS=(
    "improve-01-progress-reward"
    "improve-02-hyperparam"
    "improve-03-perf-bench"
    "improve-04-config"
    "improve-05-demo"
)

# Create each worktree
for imp in "${IMPROVEMENTS[@]}"; do
    echo "════════════════════════════════════════"
    echo "Creating worktree: $imp"
    echo "════════════════════════════════════════"
    ./scripts/create_worktree.sh "$imp" || echo "⚠️  Failed to create $imp (may already exist)"
    echo ""
done

echo "✅ All worktrees created!"
echo ""
echo "📋 List of worktrees:"
git worktree list

