#!/usr/bin/env bash
# git_push.sh — commit & push shortcut with no-change check
# Usage: ./git_push.sh "commit message"

if [ -z "$1" ]; then
    echo "Usage: $0 \"commit message\""
    exit 1
fi

COMMIT_MSG="$1"

# Add all changes
git add .

# Check if there is anything to commit
if git diff --cached --quiet; then
    echo "[INFO] Nothing to commit."
    exit 0
fi

# Commit with the provided message
git commit -m "$COMMIT_MSG"

# Push to origin main
git push -u origin main

