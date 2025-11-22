#!/bin/bash
#SBATCH --job-name=train_1.5b_fast
#SBATCH --output=logs/train_1.5b_fast_%j.out
#SBATCH --error=logs/train_1.5b_fast_%j.err
#SBATCH --partition=gpu_72h
#SBATCH --gres=rtx2080:gpu:8
#SBATCH --cpus-per-task=30

#SBATCH --time=72:00:00


################################################################################
# Qwen2.5-1.5B-Instruct Fast Training - Optimized for 8 GPUs
#
# This script trains the 1.5B model with maximum speed optimizations:
# - High batch sizes (8 per GPU)
# - Aggressive LoRA settings (rank 32)
# - Fast learning schedule
# - Expected time: 1.5-2 hours for 3 epochs
#
# Usage:
#   sbatch scripts/jobs/train_1.5b_fast.sh
#
# Or with custom config:
#   sbatch scripts/jobs/train_1.5b_fast.sh configs/config_1.5b_custom.json
################################################################################

set -e  # Exit on error
set -u  # Exit on undefined variable

# Print job information
echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"

echo "CPUs per task: $SLURM_CPUS_PER_TASK"

echo "Start time: $(date)"
echo "=========================================="

# Configuration
BASE_DIR="/research/d7/fyp25/yyyu2/FYP-LLM/"
CONFIG_FILE="$BASE_DIR/configs/config_1.5b_fast.json"



echo ""
echo "Configuration:"
echo "  Config file: $CONFIG_FILE"
echo "  Base directory: $BASE_DIR"

echo ""

# Set cache directories
export HF_HOME="$BASE_DIR/.cache/huggingface"
export TRANSFORMERS_CACHE="$BASE_DIR/.cache/huggingface/transformers"
export HF_DATASETS_CACHE="$BASE_DIR/.cache/huggingface/datasets"
export HF_HUB_CACHE="$BASE_DIR/.cache/huggingface/hub"
export XET_CACHE="$BASE_DIR/.cache/huggingface/xet"

# Create cache directories
mkdir -p "$HF_HOME"
mkdir -p "$TRANSFORMERS_CACHE"
mkdir -p "$HF_DATASETS_CACHE"
mkdir -p "$HF_HUB_CACHE"
mkdir -p "$XET_CACHE"

# Create logs directory if it doesn't exist
mkdir -p "$BASE_DIR/logs"

# Environment setup
echo "=========================================="
echo "Setting up Python environment..."
echo "=========================================="

# Load required modules (adjust based on your HPC setup)
# Uncomment and modify as needed:
# module load cuda/11.8
# module load python/3.10
# module load gcc/11.2.0

# Activate virtual environment or use uv
cd "$BASE_DIR"

if [ -f "$BASE_DIR/.venv/bin/activate" ]; then
    echo "Activating virtual environment..."
    source "$BASE_DIR/.venv/bin/activate"
elif command -v uv &> /dev/null; then
    echo "Using uv for Python environment..."
    export PATH="$HOME/.cargo/bin:$PATH"
else
    echo "Warning: No virtual environment found and uv not available"
fi

# Verify CUDA and Python
echo ""
echo "Environment check:"
python --version
which python
echo ""

if command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader,nounits
    echo ""
fi

# Verify config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "Using config file: $CONFIG_FILE"
echo ""
cat "$CONFIG_FILE"
echo ""

# Set distributed training environment variables
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_P2P_DISABLE=0

# Detect number of GPUs
NUM_GPUS=${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}
echo "Training with $NUM_GPUS GPUs"
echo ""

# Training script
TRAINING_SCRIPT="$BASE_DIR/scripts/training/train_qwen_counsel_multi_gpu.py"

if [ ! -f "$TRAINING_SCRIPT" ]; then
    echo "ERROR: Training script not found: $TRAINING_SCRIPT"
    exit 1
fi

# Start training
echo "=========================================="
echo "Starting training at $(date)"
echo "=========================================="
echo ""

# Use accelerate for multi-GPU training
if [ "$NUM_GPUS" -gt 1 ]; then
    echo "Launching multi-GPU training with accelerate..."
    accelerate launch \
        --multi_gpu \
        --num_processes=$NUM_GPUS \
        --num_machines=1 \
        --mixed_precision=bf16 \
        --dynamo_backend=no \
        "$TRAINING_SCRIPT" \
        --config "$CONFIG_FILE"
else
    echo "Launching single-GPU training..."
    python "$TRAINING_SCRIPT" --config "$CONFIG_FILE"
fi

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Training finished at $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

# Print GPU stats at the end
if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "Final GPU memory usage:"
    nvidia-smi
fi

exit $EXIT_CODE

