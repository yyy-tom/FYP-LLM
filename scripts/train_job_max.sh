#!/bin/bash
#SBATCH --job-name=qwen_train_max
#SBATCH --partition=batch_168h
#SBATCH --qos=ex_batch
#SBATCH --account=gpu
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=80
#SBATCH --time=168:00:00
#SBATCH --output=logs/train_max_%j.out
#SBATCH --error=logs/train_max_%j.err
#SBATCH --ntasks=1
#SBATCH --nodes=1

# Set base directory to avoid disk quota issues
BASE_DIR="/research/d7/fyp25/yyyu2"
cd "$BASE_DIR/FYP-LLM" || exit 1

# Create logs directory if it doesn't exist
mkdir -p logs

# Set HuggingFace cache directories to use quota path (CRITICAL for avoiding quota errors)
export HF_HOME="$BASE_DIR/.cache/huggingface"
export TRANSFORMERS_CACHE="$BASE_DIR/.cache/huggingface/transformers"
export HF_DATASETS_CACHE="$BASE_DIR/.cache/huggingface/datasets"
export HF_HUB_CACHE="$BASE_DIR/.cache/huggingface/hub"
export XET_CACHE="$BASE_DIR/.cache/huggingface/xet"

# Set uv cache directory to use quota path (CRITICAL for avoiding quota errors with uv)
export UV_CACHE_DIR="$BASE_DIR/.cache/uv"

# Create cache directories
mkdir -p "$HF_HOME"
mkdir -p "$TRANSFORMERS_CACHE"
mkdir -p "$HF_DATASETS_CACHE"
mkdir -p "$HF_HUB_CACHE"
mkdir -p "$XET_CACHE"
mkdir -p "$UV_CACHE_DIR"

# Print job information
echo "=========================================="
echo "MAXIMUM RESOURCES JOB"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "GPUs: 8"
echo "Time Limit: 168 hours (7 days)"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "=========================================="
echo "Cache Directories:"
echo "HF_HOME: $HF_HOME"
echo "TRANSFORMERS_CACHE: $TRANSFORMERS_CACHE"
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
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'PyTorch not installed')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'N/A')"
echo "Number of GPUs: $(python -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo 'N/A')"
echo "CPU count: $SLURM_CPUS_PER_TASK"
echo "=========================================="

# Run training
echo "Starting training with maximum resources..."
echo "Config: configs/config.json"
echo "=========================================="
uv run python scripts/train_qwen_counsel.py --config configs/config.json

# Print completion time
echo "=========================================="
echo "End Time: $(date)"
echo "Training completed!"
echo "=========================================="

