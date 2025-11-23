#!/bin/bash
#SBATCH --job-name=eval_1.5b
#SBATCH --output=logs/eval_1.5b_%j.out
#SBATCH --error=logs/eval_1.5b_%j.err
#SBATCH --partition=gpu_72h
#SBATCH --gres=gpu:rtx2080:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00

################################################################################
# Evaluate Qwen2.5-1.5B-Instruct Blazing Fast Model
#
# This script evaluates the fine-tuned model on the validation dataset.
# It automatically detects GPU/CUDA availability and configures accordingly.
#
# Usage:
#   sbatch scripts/jobs/evaluate_1.5b_blazing_fast.sh
#   OR run directly: bash scripts/jobs/evaluate_1.5b_blazing_fast.sh
################################################################################

set -e
set -u

echo "=========================================="
echo "📊 Model Evaluation"
echo "SLURM Job ID: ${SLURM_JOB_ID:-N/A (running interactively)}"
echo "Node: ${SLURM_NODELIST:-$(hostname)}"
echo "Start time: $(date)"
echo "=========================================="

# Configuration
BASE_DIR="/research/d7/fyp25/yyyu2/FYP-LLM"
MODEL_PATH="$BASE_DIR/models/qwen2.5-1.5b-blazing-fast"
BASE_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
TEST_DATASET="$BASE_DIR/datasets/all_mental_health_combined"
OUTPUT_FILE="$BASE_DIR/evaluation/results/qwen2.5-1.5b-blazing-fast_val.json"
MAX_SAMPLES=${1:-100}  # Allow override via command line argument

# If running from different location, try to detect
if [ ! -d "$BASE_DIR" ]; then
    # Try relative path from script location
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    BASE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
    MODEL_PATH="$BASE_DIR/models/qwen2.5-1.5b-blazing-fast"
    TEST_DATASET="$BASE_DIR/datasets/all_mental_health_combined"
    OUTPUT_FILE="$BASE_DIR/evaluation/results/qwen2.5-1.5b-blazing-fast_val.json"
fi

echo ""
echo "Configuration:"
echo "  Model: $MODEL_PATH"
echo "  Base Model: $BASE_MODEL"
echo "  Test Dataset: $TEST_DATASET"
echo "  Output: $OUTPUT_FILE"
echo "  Max Samples: $MAX_SAMPLES"
echo ""

# Set cache directories
export HF_HOME="$BASE_DIR/.cache/huggingface"
export TRANSFORMERS_CACHE="$BASE_DIR/.cache/huggingface/transformers"
export HF_DATASETS_CACHE="$BASE_DIR/.cache/huggingface/datasets"
export HF_HUB_CACHE="$BASE_DIR/.cache/huggingface/hub"
export XET_CACHE="$BASE_DIR/.cache/huggingface/xet"
export UV_CACHE_DIR="$BASE_DIR/.cache/uv"

mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$HF_HUB_CACHE" "$XET_CACHE" "$UV_CACHE_DIR"
mkdir -p "$BASE_DIR/logs"
mkdir -p "$(dirname "$OUTPUT_FILE")"

# Setup environment
cd "$BASE_DIR"

if [ -f "$BASE_DIR/.venv/bin/activate" ]; then
    source "$BASE_DIR/.venv/bin/activate"
elif command -v uv &> /dev/null; then
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# GPU and CUDA Detection
echo "=========================================="
echo "🔍 GPU and CUDA Detection"
echo "=========================================="

# Set CUDA environment variables (similar to training scripts)
# Try to detect CUDA_HOME if not set
if [ -z "${CUDA_HOME:-}" ]; then
    # Common CUDA installation paths
    for cuda_path in /usr/local/cuda-12.8 /usr/local/cuda-12.0 /usr/local/cuda-11.8 /usr/local/cuda; do
        if [ -d "$cuda_path" ]; then
            export CUDA_HOME="$cuda_path"
            break
        fi
    done
fi

# Set CUDA paths if CUDA_HOME is found
if [ -n "${CUDA_HOME:-}" ] && [ -d "${CUDA_HOME}" ]; then
    export PATH="${CUDA_HOME}/bin:${PATH}"
    echo "✓ CUDA_HOME set to: $CUDA_HOME"
else
    echo "⚠️  CUDA_HOME not found - using system CUDA if available"
fi

# Detect number of GPUs available
# For multi-GPU evaluation, we'll use all available GPUs
# IMPORTANT: Handle CUDA_VISIBLE_DEVICES carefully to avoid PyTorch initialization errors
if command -v nvidia-smi &> /dev/null; then
    # Check if GPUs are available
    NUM_GPUS_DETECTED=$(nvidia-smi -L 2>/dev/null | wc -l || echo "0")
    if [ "$NUM_GPUS_DETECTED" -gt 0 ]; then
        # If CUDA_VISIBLE_DEVICES is set by SLURM, preserve it but note it
        # PyTorch can have issues if CUDA_VISIBLE_DEVICES is changed after import
        if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
            echo "✓ CUDA_VISIBLE_DEVICES set by SLURM: $CUDA_VISIBLE_DEVICES"
            # Count how many GPUs are visible
            NUM_VISIBLE=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
            echo "  Visible GPUs: $NUM_VISIBLE"
        else
            # If not set, create list for all GPUs
            GPU_LIST=$(seq -s, 0 $((NUM_GPUS_DETECTED - 1)))
            export CUDA_VISIBLE_DEVICES="$GPU_LIST"
            echo "✓ Set CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES (using all $NUM_GPUS_DETECTED GPUs)"
        fi
        
        # Set additional environment variables to help with CUDA initialization
        export CUDA_LAUNCH_BLOCKING=0
        # Don't set TORCH_CUDA_ARCH_LIST as it might cause issues
    fi
fi

# Check for nvidia-smi
NUM_GPUS=0
USE_CPU=false

if command -v nvidia-smi &> /dev/null; then
    echo "✓ nvidia-smi found"
    echo ""
    echo "GPU Information:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader 2>/dev/null || {
        echo "⚠️  nvidia-smi command failed"
    }
    echo ""
    
    # Detect number of GPUs (before CUDA_VISIBLE_DEVICES restriction)
    NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l || echo "0")
    echo "Total GPUs detected: $NUM_GPUS"
    
    # Use SLURM GPU count if available
    if [ -n "${SLURM_GPUS_ON_NODE:-}" ]; then
        echo "SLURM GPUs allocated: $SLURM_GPUS_ON_NODE"
    fi
    
    # Display GPU usage
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        NUM_VISIBLE_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
        echo "Using $NUM_VISIBLE_GPUS GPU(s): $CUDA_VISIBLE_DEVICES"
    else
        echo "Using all available GPUs"
    fi
else
    echo "⚠️  nvidia-smi not found - GPU may not be available"
    NUM_GPUS=0
fi

# Check for CUDA compiler
if command -v nvcc &> /dev/null; then
    echo ""
    echo "CUDA Information:"
    nvcc --version 2>/dev/null | grep "release" || echo "nvcc found but version info unavailable"
else
    echo "⚠️  nvcc not found"
fi

# Check PyTorch CUDA availability (with timeout to avoid hanging)
echo ""
echo "PyTorch CUDA Check:"
PYTORCH_CUDA_AVAILABLE=false

# Quick check: just verify PyTorch is installed and get version
# Skip detailed CUDA check if it might hang - let evaluation script handle it
if command -v timeout &> /dev/null; then
    # Use timeout command if available (Linux)
    TIMEOUT_CMD="timeout 10s"
elif command -v gtimeout &> /dev/null; then
    # Use gtimeout on macOS if coreutils is installed
    TIMEOUT_CMD="gtimeout 10s"
else
    # No timeout available - use a simpler check
    TIMEOUT_CMD=""
fi

# Simple PyTorch version check (should be fast)
PYTORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "")
if [ -n "$PYTORCH_VERSION" ]; then
    echo "PyTorch version: $PYTORCH_VERSION"
    
    # If we have GPUs detected via nvidia-smi, assume CUDA might work
    # But don't test it here to avoid hanging - let the evaluation script handle it
    if [ "$NUM_GPUS" -gt 0 ]; then
        echo "GPUs detected via nvidia-smi: $NUM_GPUS"
        echo "Note: CUDA availability will be checked by evaluation script"
        PYTORCH_CUDA_AVAILABLE=true  # Optimistic - let Python script verify
    else
        echo "No GPUs detected - will use CPU"
        PYTORCH_CUDA_AVAILABLE=false
    fi
else
    echo "⚠️  PyTorch not found or import failed"
    PYTORCH_CUDA_AVAILABLE=false
fi

# Determine device strategy
# Use "auto" mode to let Python evaluation script handle CUDA detection
# This avoids hanging on CUDA initialization errors
if [ "$NUM_GPUS" -eq 0 ] || [ "$PYTORCH_CUDA_AVAILABLE" = "false" ]; then
    USE_CPU=true
    echo ""
    echo "⚠️  No GPUs detected or PyTorch not available - will use CPU"
    echo "   Note: CPU evaluation will be slower but will work correctly"
else
    USE_CPU=false
    echo ""
    echo "✓ GPUs detected - will attempt GPU evaluation"
    echo "   (Evaluation script will handle CUDA initialization and fallback to CPU if needed)"
fi

echo ""
echo "=========================================="

# Verify paths
if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model path not found: $MODEL_PATH"
    exit 1
fi

if [ ! -d "$TEST_DATASET" ]; then
    echo "ERROR: Test dataset not found: $TEST_DATASET"
    exit 1
fi

# Install evaluation dependencies if needed
echo ""
echo "Checking evaluation dependencies..."
uv pip install rouge-score nltk --quiet || echo "⚠️  Failed to install dependencies (may already be installed)"

# Run evaluation
echo ""
echo "=========================================="
echo "🚀 Starting evaluation at $(date)"
echo "=========================================="
echo ""

EVALUATION_SCRIPT="$BASE_DIR/evaluation/scripts/evaluate_model.py"

if [ ! -f "$EVALUATION_SCRIPT" ]; then
    echo "ERROR: Evaluation script not found: $EVALUATION_SCRIPT"
    exit 1
fi

# Determine if we should use multi-GPU
USE_MULTI_GPU=false
if [ "$USE_CPU" = "false" ] && [ "$NUM_GPUS" -gt 1 ]; then
    USE_MULTI_GPU=true
    echo "✓ Multi-GPU mode: Using $NUM_GPUS GPUs for parallel evaluation"
else
    echo "✓ Single-GPU or CPU mode"
fi

# Use "auto" device mode to let Python script handle device detection
# This avoids CUDA initialization issues from shell environment
if [ "$USE_CPU" = "true" ]; then
    DEVICE_ARG="cpu"
else
    # Use "auto" to let the evaluation script detect CUDA properly
    DEVICE_ARG="auto"
fi

# Build evaluation command with optional flags
EVAL_CMD="uv run $EVALUATION_SCRIPT"
EVAL_CMD="$EVAL_CMD --model_path $MODEL_PATH"
EVAL_CMD="$EVAL_CMD --base_model $BASE_MODEL"
EVAL_CMD="$EVAL_CMD --test_dataset $TEST_DATASET"
EVAL_CMD="$EVAL_CMD --output $OUTPUT_FILE"
EVAL_CMD="$EVAL_CMD --max_samples $MAX_SAMPLES"
EVAL_CMD="$EVAL_CMD --device $DEVICE_ARG"

# Add multi-GPU flag if enabled
if [ "$USE_MULTI_GPU" = "true" ]; then
    echo ""
    echo "🚀 Launching multi-GPU evaluation (DataParallel)..."
    echo "   This will use all $NUM_GPUS GPUs to accelerate evaluation"
    echo ""
    EVAL_CMD="$EVAL_CMD --multi_gpu"
else
    echo ""
    echo "🚀 Launching evaluation..."
    echo ""
fi

# Add comparison flags (can be enabled via environment variables or script modification)
# To enable comparison, set: COMPARE_WITH_BASE=true before running
if [ "${COMPARE_WITH_BASE:-false}" = "true" ]; then
    echo "📊 Comparison mode enabled: Will compare with base model"
    EVAL_CMD="$EVAL_CMD --compare_with_base"
    EVAL_CMD="$EVAL_CMD --num_comparison_examples ${NUM_COMPARISON_EXAMPLES:-10}"
fi

# Add save responses flag (can be enabled via environment variable)
if [ "${SAVE_RESPONSES:-false}" = "true" ]; then
    echo "💾 Response saving enabled: Will save all individual responses"
    EVAL_CMD="$EVAL_CMD --save_responses"
fi

# Run the evaluation
eval $EVAL_CMD

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "✅ Evaluation finished at $(date)"
echo "Exit code: $EXIT_CODE"
echo "Results saved to: $OUTPUT_FILE"
echo "=========================================="

if command -v nvidia-smi &> /dev/null && [ "$NUM_GPUS" -gt 0 ]; then
    echo ""
    echo "Final GPU memory:"
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
fi

exit $EXIT_CODE

