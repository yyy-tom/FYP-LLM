#!/bin/bash
#SBATCH --job-name=qwen-train
#SBATCH --partition=ex_batch
#SBATCH --qos=ex_gpu
#SBATCH --account=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=80
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# Recommended nodes (uncomment one):
# For Titan RTX (24GB, no CPU limit) - BEST for 80 CPUs:
#SBATCH --nodelist=gpu54
# For RTX 3090 (24GB, 6:1 CPU limit):
#SBATCH --nodelist=projgpu12
# For RTX 2080 (8-11GB, 10:1 CPU limit):
#SBATCH --nodelist=gpu40

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
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
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

# Model size selection (change config file as needed):
# - configs/config.json (0.5B)
# - configs/config_3b_cpu.json (3B)
# - configs/config_7b_cpu.json (7B)

# Train with GPU
echo ""
echo "Starting training..."
uv run python scripts/train_qwen_counsel.py --config configs/config_14b_optimized

echo ""
echo "Training completed at: $(date)"

