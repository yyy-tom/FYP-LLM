#!/bin/bash
#SBATCH --job-name=train_blazing
#SBATCH --output=logs/train_blazing_%j.out
#SBATCH --error=logs/train_blazing_%j.err
#SBATCH --partition=gpu_72h
#SBATCH --gres=rtx2080:gpu:8
#SBATCH --cpus-per-task=30
#SBATCH --time=02:00:00

################################################################################
# Qwen2.5-1.5B-Instruct BLAZING FAST Training
#
# EXTREME SPEED MODE - Minimal quality, maximum speed
# Expected completion time: 30-45 minutes on 8 GPUs
#
# Speed optimizations:
# - Only 1 epoch
# - Max sequence length: 256 tokens
# - Minimal LoRA (rank 8, only 2 modules)
# - No gradient checkpointing
# - Rare evaluation and saves
# - High batch size
#
# Usage:
#   sbatch scripts/jobs/train_1.5b_blazing_fast.sh
################################################################################

set -e
set -u

echo "=========================================="
echo "🔥 BLAZING FAST TRAINING MODE 🔥"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=========================================="

# Configuration
BASE_DIR="/research/d7/fyp25/yyyu2/FYP-LLM/"
CONFIG_FILE="$BASE_DIR/configs/config_1.5b_blazing_fast.json"

echo ""
echo "Configuration:"
echo "  Config: config_1.5b_blazing_fast.json"
echo "  Expected time: 30-45 minutes"
echo ""

# Set cache directories
export HF_HOME="$BASE_DIR/.cache/huggingface"
export TRANSFORMERS_CACHE="$BASE_DIR/.cache/huggingface/transformers"
export HF_DATASETS_CACHE="$BASE_DIR/.cache/huggingface/datasets"
export HF_HUB_CACHE="$BASE_DIR/.cache/huggingface/hub"
export XET_CACHE="$BASE_DIR/.cache/huggingface/xet"

mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$HF_HUB_CACHE" "$XET_CACHE"
mkdir -p "$BASE_DIR/logs"

# Setup environment
cd "$BASE_DIR"

if [ -f "$BASE_DIR/.venv/bin/activate" ]; then
    source "$BASE_DIR/.venv/bin/activate"
elif command -v uv &> /dev/null; then
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# Environment check
echo "Python: $(python --version)"
if command -v nvidia-smi &> /dev/null; then
    echo "GPUs detected:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
    echo ""
fi

# Verify config
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Distributed training setup
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_P2P_DISABLE=0

NUM_GPUS=${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}
echo "Training with $NUM_GPUS GPUs"
echo ""

TRAINING_SCRIPT="$BASE_DIR/scripts/training/train_qwen_counsel_multi_gpu.py"

if [ ! -f "$TRAINING_SCRIPT" ]; then
    echo "ERROR: Training script not found: $TRAINING_SCRIPT"
    exit 1
fi

# Start training
echo "=========================================="
echo "🚀 Starting BLAZING FAST training at $(date)"
echo "=========================================="
echo ""

if [ "$NUM_GPUS" -gt 1 ]; then
    accelerate launch \
        --multi_gpu \
        --num_processes=$NUM_GPUS \
        --num_machines=1 \
        --mixed_precision=bf16 \
        --dynamo_backend=no \
        "$TRAINING_SCRIPT" \
        --config "$CONFIG_FILE"
else
    python "$TRAINING_SCRIPT" --config "$CONFIG_FILE"
fi

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "✅ Training finished at $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "Final GPU memory:"
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
fi

exit $EXIT_CODE

