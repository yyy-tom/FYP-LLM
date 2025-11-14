#!/bin/bash
# Interactive training script for 10 CPUs (max for srun with highcpucount)
# Usage: srun -p gpu_24h --qos gpu --account gpu --gres=gpu:1 --cpus-per-task=10 --constraint=highcpucount --nodelist=gpu54 --pty bash scripts/train_job_interactive_10cpu.sh

# Set base directory
BASE_DIR="/research/d7/fyp25/yyyu2"
cd "$BASE_DIR/FYP-LLM" || exit 1

# Create logs directory if it doesn't exist
mkdir -p logs

# Set HuggingFace cache directories
export HF_HOME="$BASE_DIR/.cache/huggingface"
export TRANSFORMERS_CACHE="$BASE_DIR/.cache/huggingface/transformers"
export HF_DATASETS_CACHE="$BASE_DIR/.cache/huggingface/datasets"
export HF_HUB_CACHE="$BASE_DIR/.cache/huggingface/hub"
export XET_CACHE="$BASE_DIR/.cache/huggingface/xet"
export UV_CACHE_DIR="$BASE_DIR/.cache/uv"

# Create cache directories
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$HF_HUB_CACHE" "$XET_CACHE" "$UV_CACHE_DIR"

# Print job information
echo "=========================================="
echo "Interactive Training Session"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "CPUs: $SLURM_CPUS_ON_NODE"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "=========================================="

# Check GPU
echo "GPU Information:"
nvidia-smi

# Check CUDA
echo ""
echo "CUDA Information:"
nvcc --version 2>/dev/null || echo "nvcc not found"
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')" 2>/dev/null || echo "PyTorch not installed"

# Note: With only 10 CPUs, training will be slower
# Consider using batch job with 80 CPUs for faster training
echo ""
echo "Note: You have 10 CPUs. For faster training with 80 CPUs, use:"
echo "  sbatch scripts/train_job_gpu.sh"
echo ""

# Model size selection (change config file as needed):
# - configs/config.json (0.5B)
# - configs/config_3b_cpu.json (3B)
# - configs/config_7b_cpu.json (7B)

# Train with GPU
echo "Starting training..."
echo "You can modify the config file in the command below:"
echo ""
echo "To train, run:"
echo "  uv run python scripts/train_qwen_counsel.py --config configs/config_7b_cpu.json"
echo ""

# Keep shell open for interactive use
exec /bin/bash

