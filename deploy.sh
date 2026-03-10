#!/bin/bash
# Sync code from laptop to Barry and restart the specified service
# Usage: ./deploy.sh [clm|llm|all]   default: clm
#
# Set GPU_HOST in your environment to override, e.g.:
#   export GPU_HOST=fhh@barry

GPU_HOST=${GPU_HOST:-fhh@barry}
DEPLOY_DIR=/opt/ivade
TARGET=${1:-clm}

echo "==> Syncing to $GPU_HOST:$DEPLOY_DIR ..."
rsync -avz \
    --exclude='.git' \
    --exclude='models' \
    --exclude='.env' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='database/*.db' \
    ./ $GPU_HOST:$DEPLOY_DIR/

if [ "$TARGET" = "all" ]; then
    echo "==> Restarting llm and clm..."
    ssh $GPU_HOST "cd $DEPLOY_DIR && docker compose restart llm && docker compose restart clm"
else
    echo "==> Restarting $TARGET ..."
    ssh $GPU_HOST "cd $DEPLOY_DIR && docker compose restart $TARGET"
fi

echo "==> Tailing logs for $TARGET (Ctrl+C to stop)..."
ssh $GPU_HOST "cd $DEPLOY_DIR && docker compose logs -f $TARGET"
