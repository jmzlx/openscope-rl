#!/bin/bash
# Create a Git worktree for an experimental approach
#
# Usage: ./scripts/create_worktree.sh 01-baseline-ppo

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <approach-name>"
    echo "Example: $0 01-baseline-ppo"
    exit 1
fi

APPROACH=$1
BRANCH="experiment/${APPROACH}"
WORKTREE=".trees/${APPROACH}"

# Check if worktree already exists
if [ -d "$WORKTREE" ]; then
    echo "❌ Worktree already exists: $WORKTREE"
    exit 1
fi

# Create branch if it doesn't exist
if git show-ref --verify --quiet "refs/heads/${BRANCH}"; then
    echo "ℹ️  Branch already exists: ${BRANCH}"
else
    echo "📝 Creating branch: ${BRANCH}"
    git branch "${BRANCH}"
fi

# Create worktree
echo "🌳 Creating worktree: ${WORKTREE}"
git worktree add "${WORKTREE}" "${BRANCH}"

echo "✅ Worktree created successfully!"
echo "   Branch: ${BRANCH}"
echo "   Worktree: ${WORKTREE}"
echo ""
echo "To work in this worktree:"
echo "   cd ${WORKTREE}"
