#!/bin/bash
#SBATCH --job-name=qwen_train
#SBATCH --partition=gpu_8h
#SBATCH --qos=gpu
#SBATCH --account=gpu
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"

# Load environment (adjust if needed)
# module load python/3.10  # Uncomment and adjust if needed
# source venv/bin/activate  # Uncomment if using virtual environment

# Set CUDA visible devices (SLURM handles this automatically, but explicit is good)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Print GPU information
echo "GPU Information:"
nvidia-smi

# Print Python and PyTorch versions
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "Number of GPUs: $(python -c 'import torch; print(torch.cuda.device_count())')"

# Run training
echo "Starting training..."
python train_qwen_counsel.py --config config.json

# Print completion time
echo "End Time: $(date)"
echo "Training completed!"

