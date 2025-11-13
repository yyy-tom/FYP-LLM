#!/bin/bash
# Helper script to run training in screen session
# Usage: ./run_training_screen.sh

SESSION_NAME="qwen_training"

# Check if screen session already exists
if screen -list | grep -q "$SESSION_NAME"; then
    echo "Session '$SESSION_NAME' already exists. Attach with: screen -r $SESSION_NAME"
    echo "Or kill it first with: screen -S $SESSION_NAME -X quit"
    exit 1
fi

# Create new screen session and run training
screen -dmS "$SESSION_NAME" bash -c "python scripts/train_qwen_counsel.py --config configs/config.json; exec bash"

echo "Training started in screen session '$SESSION_NAME'"
echo "To attach: screen -r $SESSION_NAME"
echo "To detach: Press Ctrl+A, then D"
echo "To kill session: screen -S $SESSION_NAME -X quit"

