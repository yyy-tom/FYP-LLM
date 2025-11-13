#!/bin/bash
# Interactive training with tmux - Best of both worlds!
# This gives you an interactive session (like tmux) but with SLURM resource management
# Usage: ./run_interactive_training.sh

SESSION_NAME="qwen_training"

echo "Requesting interactive SLURM session with 8 GPUs..."
echo "This may take a moment to allocate resources..."

# Request interactive session and immediately start tmux
srun -p gpu_8h --qos gpu --account gpu --gres=gpu:8 --cpus-per-task=8 --pty bash -c "
    echo 'SLURM session allocated!'
    echo 'Node: \$SLURM_NODELIST'
    echo 'Job ID: \$SLURM_JOB_ID'
    echo ''
    echo 'Starting tmux session: $SESSION_NAME'
    echo 'To detach: Press Ctrl+B, then D'
    echo 'To reattach later: tmux attach -t $SESSION_NAME'
    echo ''
    
    # Check if tmux session exists
    if tmux has-session -t '$SESSION_NAME' 2>/dev/null; then
        echo 'Session exists, attaching...'
        tmux attach -t '$SESSION_NAME'
    else
        # Create new session and run training
        tmux new-session -s '$SESSION_NAME' -d 'python scripts/train_qwen_counsel.py --config configs/config.json; exec bash'
        tmux attach -t '$SESSION_NAME'
    fi
"

