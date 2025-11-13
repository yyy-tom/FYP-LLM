#!/bin/bash
# Helper script to run training in tmux session
# Usage: ./run_training_tmux.sh

SESSION_NAME="qwen_training"

# Check if tmux session already exists
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Session '$SESSION_NAME' already exists. Attach with: tmux attach -t $SESSION_NAME"
    echo "Or kill it first with: tmux kill-session -t $SESSION_NAME"
    exit 1
fi

# Create new tmux session
tmux new-session -d -s "$SESSION_NAME"

# Send training command to tmux session
tmux send-keys -t "$SESSION_NAME" "python scripts/train_qwen_counsel.py --config configs/config.json" C-m

echo "Training started in tmux session '$SESSION_NAME'"
echo "To attach: tmux attach -t $SESSION_NAME"
echo "To detach: Press Ctrl+B, then D"
echo "To kill session: tmux kill-session -t $SESSION_NAME"

