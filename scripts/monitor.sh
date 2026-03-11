#!/bin/bash
# Open a tmux session with 4 panes, each tailing logs for one Docker service
# Usage: ./scripts/monitor.sh
# Requires: tmux

cd "$(dirname "$0")/.."

if ! command -v tmux &>/dev/null; then
    echo "tmux not found. Install with: sudo apt install tmux"
    exit 1
fi

SESSION="ivade"

# Kill existing session if present
tmux kill-session -t $SESSION 2>/dev/null

tmux new-session -d -s $SESSION -x 220 -y 50

# Split into 4 panes: 2x2 grid
tmux rename-window -t $SESSION "IVADE Logs"
tmux send-keys -t $SESSION "docker compose logs -f llm" Enter

tmux split-window -h -t $SESSION
tmux send-keys -t $SESSION "docker compose logs -f clm" Enter

tmux split-window -v -t $SESSION:0.0
tmux send-keys -t $SESSION "docker compose logs -f ngrok" Enter

tmux split-window -v -t $SESSION:0.1
tmux send-keys -t $SESSION "docker compose logs -f tunnel" Enter

# Even out pane sizes
tmux select-layout -t $SESSION tiled

tmux attach-session -t $SESSION
