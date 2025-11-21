# SRUN Commands for Interactive GPU Access

## Recommended Commands

### Option 1: Titan RTX (gpu54-59) - BEST for 80 CPUs ✅

**Why**: No CPU limit, 24GB VRAM, perfect for 7B models

**IMPORTANT**: For CPU:GPU ratio > 10:1 (80 CPUs : 1 GPU), you must use **batch job (sbatch)** with `ex_gpu` partition/QOS, not interactive `srun`.

```bash
# For interactive access with 80 CPUs, use batch job instead (see below)
# OR use fewer CPUs (10 CPUs max for interactive srun with highcpucount)

# Interactive with 10 CPUs (for testing/setup)
srun -p gpu_24h --qos gpu --account gpu --gres=gpu:1 --cpus-per-task=10 --constraint=highcpucount --nodelist=gpu54 --pty /bin/bash
```

**For 80 CPUs, use batch job (see "Batch Job for 80 CPUs" section below)**

### Option 2: RTX 3090 (projgpu12-14) - Good Alternative ✅

**Why**: 24GB VRAM, modern GPU, but CPU limit is 6:1 or 10:1

```bash
# For CPU:GPU ratio 6:1 to 10:1, need highcpucount constraint
# Single GPU, 10 CPUs (max for interactive with highcpucount)
srun -p gpu_24h --qos gpu --account gpu --gres=gpu:1 --cpus-per-task=10 --constraint=highcpucount --nodelist=projgpu12 --pty /bin/bash

# Alternative nodes
srun -p gpu_24h --qos gpu --account gpu --gres=gpu:1 --cpus-per-task=10 --constraint=highcpucount --nodelist=projgpu13 --pty /bin/bash
srun -p gpu_24h --qos gpu --account gpu --gres=gpu:1 --cpus-per-task=10 --constraint=highcpucount --nodelist=projgpu14 --pty /bin/bash
```

### Option 3: RTX 2080 (gpu40-51) - For Smaller Models ✅

**Why**: 8-11GB VRAM, good for 0.5B-3B models, many available nodes

```bash
# Single GPU, 80 CPUs
srun -p gpu_8h --qos gpu --account gpu --gres=gpu:1 --cpus-per-task=80 --nodelist=gpu40 --pty /bin/bash

# Alternative nodes (gpu40-51 available)
srun -p gpu_8h --qos gpu --account gpu --gres=gpu:1 --cpus-per-task=80 --nodelist=gpu41 --pty /bin/bash
```

## Quick Reference by Model Size

### For 7B Model Training

```bash
# Best: Titan RTX (no CPU limit)
srun -p gpu_24h --qos gpu --account gpu --gres=gpu:1 --cpus-per-task=80 --nodelist=gpu54 --pty /bin/bash

# Alternative: RTX 3090
srun -p gpu_24h --qos gpu --account gpu --gres=gpu:1 --cpus-per-task=80 --nodelist=projgpu12 --pty /bin/bash
```

### For 3B Model Training

```bash
# RTX 2080 or RTX 3090
srun -p gpu_8h --qos gpu --account gpu --gres=gpu:1 --cpus-per-task=80 --nodelist=gpu40 --pty /bin/bash
```

### For 0.5B-1.5B Model Training

```bash
# Any modern GPU
srun -p gpu_8h --qos gpu --account gpu --gres=gpu:1 --cpus-per-task=80 --nodelist=gpu40 --pty /bin/bash
```

## Time Limits by Partition

- `gpu_2h`: 2 hours (quick testing)
- `gpu_8h`: 8 hours (good for training)
- `gpu_24h`: 24 hours (best for long training)
- `gpu_72h`: 72 hours (very long training)

## After Getting Interactive Shell

Once you get the interactive shell, verify GPU access:

```bash
# Check GPU
nvidia-smi

# Check CUDA
nvcc --version

# Check PyTorch CUDA
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

# Navigate to project
cd /research/d7/fyp25/yyyu2/FYP-LLM

# Install PyTorch with CUDA (if not already installed)
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Train
uv run python scripts/train_qwen_counsel.py --config configs/config_7b_cpu.json
```

## Troubleshooting

### If node is busy, try:

1. Different node in same category
2. Different partition (gpu_8h instead of gpu_24h)
3. Remove `--nodelist` to let SLURM choose

### If CPU limit error:

- Titan RTX (gpu54-59): No limit, should work
- RTX 3090: Try `--cpus-per-task=60` instead of 80
- RTX 2080: Try `--cpus-per-task=60` instead of 80

### Check available nodes:

```bash
sinfo -p gpu_24h -o "%N %G %e %m"
```
