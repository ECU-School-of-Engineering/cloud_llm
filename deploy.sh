#!/bin/bash
# Deploy to Barry via GitHub Actions
# Usage: ./deploy.sh [message]
#
# Commits any uncommitted changes, pushes to main, and the
# self-hosted runner on Barry picks up the job automatically.

MSG=${1:-"deploy $(date '+%Y-%m-%d %H:%M')"}

echo "==> Committing and pushing..."
git add -A
git commit -m "$MSG" 2>/dev/null || echo "(nothing to commit)"
git push origin main

echo ""
echo "==> Push done. GitHub Actions will now:"
echo "    1. Run deploy.yml on Barry's self-hosted runner"
echo "    2. git pull + docker compose restart clm"
echo ""
echo "Monitor at: https://github.com/$(git remote get-url origin | sed 's/.*github.com[:/]//' | sed 's/\.git$//')/actions"
