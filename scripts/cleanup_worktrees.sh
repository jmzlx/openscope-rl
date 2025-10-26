#!/bin/bash
# Clean up all experimental worktrees
#
# Usage: ./scripts/cleanup_worktrees.sh [--force]

FORCE=false
if [ "$1" == "--force" ]; then
    FORCE=true
fi

echo "🧹 Cleaning up experimental worktrees..."
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

if [ "$FORCE" = false ]; then
    echo "⚠️  This will remove all worktrees and their branches!"
    echo "Files will be preserved in .trees/ but worktrees will be unlinked."
    echo ""
    read -p "Are you sure? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "❌ Aborted"
        exit 1
    fi
fi

# Remove each worktree
for approach in "${APPROACHES[@]}"; do
    WORKTREE=".trees/${approach}"
    BRANCH="experiment/${approach}"

    echo "🗑️  Removing worktree: $approach"

    if [ -d "$WORKTREE" ]; then
        git worktree remove "$WORKTREE" --force || echo "⚠️  Failed to remove worktree"
    fi

    # Optionally delete branch
    if [ "$FORCE" = true ]; then
        if git show-ref --verify --quiet "refs/heads/${BRANCH}"; then
            git branch -D "${BRANCH}" || echo "⚠️  Failed to delete branch"
        fi
    fi
done

echo ""
echo "✅ Cleanup complete!"
echo ""
echo "📋 Remaining worktrees:"
git worktree list
