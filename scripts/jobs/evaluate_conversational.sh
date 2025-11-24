#!/bin/bash
#SBATCH --job-name=eval_1.5b
#SBATCH --chdir=/research/d7/fyp25/yyyu2/FYP-LLM
#SBATCH --output=logs/eval_1.5b_%j.out
#SBATCH --error=logs/eval_1.5b_%j.err
#SBATCH --partition=gpu_72h
#SBATCH --gres=gpu:rtx2080:8
#SBATCH --cpus-per-task=30
#SBATCH --time=72:00:00

# ==========================================
# Conversational Model Evaluation Script
# ==========================================
# This script evaluates the fine-tuned model
# in conversational (multi-turn) mode
#
# IMPORTANT: Submit this script from the project root directory:
#   cd /research/d7/fyp25/yyyu2/FYP-LLM
#   sbatch scripts/jobs/evaluate_conversational.sh
#
# The logs/ directory must exist in the submission directory
# ==========================================

# Don't use set -e yet - we need to handle directory setup first
# Get base directory FIRST (before any other operations)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# CRITICAL: Change to BASE_DIR immediately and create logs directory
# This ensures relative paths in SBATCH directives work correctly
# Note: SLURM processes #SBATCH directives before script runs,
# so logs directory must exist or be created in the submission directory
cd "$BASE_DIR" || {
    echo "ERROR: Cannot change to BASE_DIR: $BASE_DIR"
    exit 1
}
mkdir -p "$BASE_DIR/logs" || {
    echo "ERROR: Cannot create logs directory: $BASE_DIR/logs"
    exit 1
}

# Now we can safely use set -e
set -e  # Exit on error

# Get job information
JOB_ID=${SLURM_JOB_ID:-"local"}
NODE_NAME=$(hostname)
START_TIME=$(date)

echo "=========================================="
echo "📊 Conversational Model Evaluation"
echo "=========================================="
echo "SLURM Job ID: $JOB_ID"
echo "Node: $NODE_NAME"
echo "Start time: $START_TIME"
echo "Working directory: $(pwd)"
echo "=========================================="

# Configuration
MODEL_PATH="models/qwen2.5-1.5b-blazing-fast"
BASE_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
TEST_DATASET="datasets/esconv_processed"
OUTPUT_FILE="evaluation/results/qwen2.5-1.5b-blazing-fast_esconv_conversational.json"
MAX_SAMPLES=100
MIN_CONVERSATION_TURNS=2
MAX_CONVERSATION_TURNS=5

echo ""
echo "Configuration:"
echo "  Model: $MODEL_PATH"
echo "  Base Model: $BASE_MODEL"
echo "  Test Dataset: $TEST_DATASET"
echo "  Output: $OUTPUT_FILE"
echo "  Max Samples: $MAX_SAMPLES"
echo "  Min Conversation Turns: $MIN_CONVERSATION_TURNS"
echo "  Max Conversation Turns: $MAX_CONVERSATION_TURNS"
echo "  Mode: CONVERSATIONAL (multi-turn)"
echo "=========================================="

# ==========================================
# 🔍 GPU and CUDA Detection
# ==========================================
echo ""
echo "=========================================="
echo "🔍 GPU and CUDA Detection"
echo "=========================================="

# Set CUDA environment
if [ -z "$CUDA_HOME" ]; then
    if [ -d "/usr/local/cuda-12.8" ]; then
        export CUDA_HOME="/usr/local/cuda-12.8"
    elif [ -d "/usr/local/cuda" ]; then
        export CUDA_HOME="/usr/local/cuda"
    fi
fi

if [ -n "$CUDA_HOME" ]; then
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
    echo "✓ CUDA_HOME set to: $CUDA_HOME"
else
    echo "⚠️  CUDA_HOME not found, using system CUDA"
fi

# Check for GPUs
if command -v nvidia-smi &> /dev/null; then
    echo "✓ nvidia-smi found"
    echo "GPU Information:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader | while IFS=, read -r index name mem_total mem_free; do
        echo "  $index, $name, $mem_total, $mem_free"
    done
    
    # Count GPUs
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    echo "Detected GPUs: $GPU_COUNT"
    
    # Set CUDA_VISIBLE_DEVICES if not already set by SLURM
    if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
        # Use first GPU by default
        export CUDA_VISIBLE_DEVICES=0
        echo "CUDA_VISIBLE_DEVICES set to: 0"
    else
        echo "CUDA_VISIBLE_DEVICES already set: $CUDA_VISIBLE_DEVICES"
    fi
else
    echo "⚠️  nvidia-smi not found - GPU detection unavailable"
fi

# Check CUDA compiler
if command -v nvcc &> /dev/null; then
    NVCC_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | sed 's/,//')
    echo "CUDA Information: Cuda compilation tools, release $NVCC_VERSION"
else
    echo "⚠️  nvcc not found in PATH"
fi

# Check PyTorch CUDA availability (simple check)
echo "PyTorch CUDA Check:"
echo "  Note: CUDA availability will be checked by evaluation script"
echo "✓ GPUs detected - will attempt GPU evaluation"
echo "  (Evaluation script will handle CUDA initialization and fallback to CPU if needed)"

# ==========================================
# Set Cache Directories
# ==========================================
echo ""
echo "=========================================="
echo "Setting up cache directories..."
echo "=========================================="

# Set cache directories to use large disk space
LARGE_DISK_PATH="/research/d7/fyp25/yyyu2"
if [ -d "$LARGE_DISK_PATH" ]; then
    export HF_HOME="$LARGE_DISK_PATH/.cache/huggingface"
    export TRANSFORMERS_CACHE="$LARGE_DISK_PATH/.cache/huggingface/transformers"
    export HF_DATASETS_CACHE="$LARGE_DISK_PATH/.cache/huggingface/datasets"
    export HF_HUB_CACHE="$LARGE_DISK_PATH/.cache/huggingface/hub"
    export XET_CACHE="$LARGE_DISK_PATH/.cache/huggingface/xet"
    
    # CRITICAL: Set TMPDIR to large disk to avoid quota issues during download
    export TMPDIR="$LARGE_DISK_PATH/.cache/tmp"
    export TMP="$LARGE_DISK_PATH/.cache/tmp"
    export TEMP="$LARGE_DISK_PATH/.cache/tmp"
    
    mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$HF_HUB_CACHE" "$XET_CACHE" "$TMPDIR"
    echo "✓ Using large disk cache: $HF_HOME"
    echo "✓ Temporary files directory: $TMPDIR"
else
    echo "⚠️  Large disk path $LARGE_DISK_PATH not found. Using default cache directories."
    mkdir -p "$BASE_DIR/.cache/huggingface" "$BASE_DIR/.cache/uv"
fi

# ==========================================
# Check Dependencies
# ==========================================
echo ""
echo "=========================================="
echo "Checking evaluation dependencies..."
echo "=========================================="

# Check if virtual environment exists
if [ -d ".venv" ]; then
    echo "✓ Virtual environment found"
    source .venv/bin/activate
else
    echo "⚠️  Virtual environment not found, using system Python"
fi

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "⚠️  Warning: Model path $MODEL_PATH does not exist"
    echo "   Evaluation will proceed but may fail if model is not cached"
fi

# Check if dataset exists
if [ ! -d "$TEST_DATASET" ]; then
    echo "⚠️  Warning: Dataset path $TEST_DATASET does not exist"
    echo "   Evaluation will fail if dataset cannot be loaded"
    exit 1
else
    echo "✓ Test dataset found: $TEST_DATASET"
fi

# Create output directory if it doesn't exist
OUTPUT_DIR=$(dirname "$OUTPUT_FILE")
mkdir -p "$OUTPUT_DIR"
echo "✓ Output directory: $OUTPUT_DIR"

# ==========================================
# Run Evaluation
# ==========================================
echo ""
echo "=========================================="
echo "🚀 Starting conversational evaluation at $(date)"
echo "=========================================="

# Run the evaluation script
python evaluation/scripts/evaluate_model.py \
    --model_path "$MODEL_PATH" \
    --base_model "$BASE_MODEL" \
    --test_dataset "$TEST_DATASET" \
    --output "$OUTPUT_FILE" \
    --max_samples "$MAX_SAMPLES" \
    --device auto \
    --compare_with_base \
    --conversational_mode \
    --min_conversation_turns "$MIN_CONVERSATION_TURNS" \
    --max_conversation_turns "$MAX_CONVERSATION_TURNS" \
    --num_comparison_examples 20

EXIT_CODE=$?

# ==========================================
# Summary
# ==========================================
echo ""
echo "=========================================="
echo "Evaluation Summary"
echo "=========================================="
END_TIME=$(date)
echo "End time: $END_TIME"
echo "Exit code: $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Evaluation completed successfully!"
    echo "  Results saved to: $OUTPUT_FILE"
    
    # Check if output file exists and show size
    if [ -f "$OUTPUT_FILE" ]; then
        FILE_SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)
        echo "  Output file size: $FILE_SIZE"
    fi
else
    echo "✗ Evaluation failed with exit code: $EXIT_CODE"
    echo "  Check the error log for details"
fi

echo "=========================================="
exit $EXIT_CODE

