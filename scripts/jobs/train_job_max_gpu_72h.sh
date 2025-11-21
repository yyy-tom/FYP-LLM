#!/bin/bash
#SBATCH --job-name=yyy_model_train_max
#SBATCH --partition=gpu_72h
#SBATCH --qos=gpu
#SBATCH --account=gpu
#SBATCH --gres=gpu:rtx2080:2
#SBATCH --cpus-per-task=30
#SBATCH --time=72:00:00
#SBATCH --output=logs/train_max_%j.out
#SBATCH --error=logs/train_max_%j.err
#SBATCH --ntasks=1


# Set base directory to avoid disk quota issues
BASE_DIR="/research/d7/fyp25/yyyu2"
cd "$BASE_DIR/FYP-LLM" || exit 1

# Create logs directory if it doesn't exist
mkdir -p logs

# Set HuggingFace cache directories to use quota path (CRITICAL for avoiding quota errors)
export HF_HOME="$BASE_DIR/.cache/huggingface"
# Note: TRANSFORMERS_CACHE is deprecated, using HF_HOME instead
export HF_DATASETS_CACHE="$BASE_DIR/.cache/huggingface/datasets"
export HF_HUB_CACHE="$BASE_DIR/.cache/huggingface/hub"
export XET_CACHE="$BASE_DIR/.cache/huggingface/xet"

# Set uv cache directory to use quota path (CRITICAL for avoiding quota errors with uv)
export UV_CACHE_DIR="$BASE_DIR/.cache/uv"

# Create cache directories
mkdir -p "$HF_HOME"
mkdir -p "$HF_DATASETS_CACHE"
mkdir -p "$HF_HUB_CACHE"
mkdir -p "$XET_CACHE"
mkdir -p "$UV_CACHE_DIR"

# Print job information
echo "=========================================="
echo "MAXIMUM RESOURCES JOB (GPU_72H)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "GPUs: 8"
echo "Time Limit: 72 hours (3 days)"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "=========================================="
echo "Cache Directories:"
echo "HF_HOME: $HF_HOME"
echo "HF_DATASETS_CACHE: $HF_DATASETS_CACHE"
echo "HF_HUB_CACHE: $HF_HUB_CACHE"
echo "XET_CACHE: $XET_CACHE"
echo "UV_CACHE_DIR: $UV_CACHE_DIR"
echo "=========================================="

# Set CUDA environment variables (needed for PyTorch to find CUDA at runtime)
export CUDA_HOME=/usr/local/cuda-10.0
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Set PyTorch memory allocator to reduce fragmentation (helps with OOM during DDP init)
# Note: PYTORCH_CUDA_ALLOC_CONF is deprecated, use PYTORCH_ALLOC_CONF instead
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Print CUDA information
echo "CUDA_HOME: $CUDA_HOME"
echo "CUDA version: $(nvcc --version 2>/dev/null | grep 'release' || echo 'nvcc not found')"

# Print GPU information
echo "=========================================="
echo "GPU Information:"
echo "=========================================="
nvidia-smi

# Print Python and PyTorch versions
echo "=========================================="
echo "Environment Information:"
echo "=========================================="
echo "Python version: $(uv run python --version)"
echo "PyTorch version: $(uv run python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'PyTorch not installed')"
echo "CUDA available: $(uv run python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'N/A')"
echo "Number of GPUs: $(uv run python -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo 'N/A')"
echo "CPU count: $SLURM_CPUS_PER_TASK"
echo "=========================================="

# Run training
echo "Starting training with maximum resources..."
echo "Config: configs/config_7b_optimized.json"
echo "=========================================="

# Check and install bitsandbytes if needed
echo "Checking for bitsandbytes..."
if ! uv run python -c "import bitsandbytes" 2>/dev/null; then
    echo "bitsandbytes not found. Installing..."
    uv pip install bitsandbytes
else
    echo "bitsandbytes is already installed."
fi

# Count number of GPUs available (use actual GPU count from PyTorch if available, otherwise count CUDA_VISIBLE_DEVICES)
NUM_GPUS=$(uv run python -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null)
if [ -z "$NUM_GPUS" ] || [ "$NUM_GPUS" = "0" ]; then
    # Fallback: count GPUs in CUDA_VISIBLE_DEVICES
    NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | grep -v '^$' | wc -l)
fi
echo "Number of GPUs detected: $NUM_GPUS"

# Launch training with accelerate for proper multi-GPU support
uv run accelerate launch \
    --num_processes $NUM_GPUS \
    --num_machines 1 \
    --mixed_precision bf16 \
    scripts/training/train_qwen_counsel_multi_gpu.py \
    --config configs/config_7b_8gpu.json \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --dataset_path datasets/all_mental_health_combined


# Print completion time
echo "=========================================="
echo "End Time: $(date)"
echo "Training completed!"
echo "=========================================="

