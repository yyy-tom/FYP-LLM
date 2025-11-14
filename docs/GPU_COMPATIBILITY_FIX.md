# GPU Compatibility Fix for GTX TITAN X

## Problem

Your GPUs are **NVIDIA GeForce GTX TITAN X** with CUDA capability **sm_52** (Maxwell architecture, 2015).

The current PyTorch installation only supports newer architectures:
- sm_70, sm_75, sm_80, sm_86, sm_90, sm_100, sm_120

## Impact

⚠️ **This is a WARNING, not necessarily a fatal error:**
- Training might still work but will be **very slow** (CPU fallback)
- GPU acceleration may not work properly
- Memory usage might be inefficient

## Solutions

### Option 1: Install PyTorch with sm_52 Support (Recommended if you must use these GPUs)

Install an older PyTorch version that supports Maxwell architecture:

```bash
# On your HPC cluster
cd /research/d7/fyp25/yyyu2/FYP-LLM

# Uninstall current PyTorch
uv pip uninstall torch torchvision torchaudio

# Install PyTorch 1.13.1 (last version with sm_52 support)
# For CUDA 10.0
uv pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

# Or try PyTorch 2.0.0 with CUDA 11.8 (might work)
uv pip install torch==2.0.0+cu118 torchvision==0.15.0+cu118 torchaudio==2.0.0 --extra-index-url https://download.pytorch.org/whl/cu118
```

**Note**: This will limit you to older PyTorch versions, which may have compatibility issues with newer transformers.

### Option 2: Request Different GPU Nodes (Best Solution) ⭐

Check if your HPC cluster has newer GPUs available. From your SLURM table, you might have access to:
- `gpu{1-53}` or `gpu{54-59}` - Check what GPUs these are
- `projgpu{1-15}` - Project-specific GPUs

**Check available GPUs:**
```bash
# Check what GPUs are available on different nodes
sinfo -o "%N %G"  # Show nodes and GPUs

# Or check specific partition
sinfo -p gpu_8h -o "%N %G %f"
```

**Request specific GPU type:**
```bash
# If you have access to newer GPUs, request them
sbatch --partition=gpu_8h --gres=gpu:8 --constraint="gpu_type:V100|gpu_type:A100" train_job.sh
```

### Option 3: Use CPU Training (Not Recommended)

If you must use these GPUs and can't change:
- Training will be **extremely slow** (days/weeks instead of hours)
- Not practical for 7B model

### Option 4: Check if Training Actually Works

Sometimes PyTorch will still use GPU despite the warning. Test it:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

If it returns `True` and shows your GPU, it might work (just slower).

## Recommended Action

**Best approach**: Contact your HPC administrator or check if you can request nodes with newer GPUs (V100, A100, RTX series).

**If you must use TITAN X**: Install older PyTorch (Option 1), but be aware of compatibility limitations.

## Checking Your Current Setup

```bash
# Check CUDA version
nvcc --version

# Check PyTorch CUDA support
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')"

# Check GPU compute capability
nvidia-smi --query-gpu=compute_cap --format=csv
```

## Alternative: Use Smaller Model

If you're stuck with TITAN X GPUs, consider using a smaller model:
- Qwen2.5-1.5B or Qwen2.5-3B instead of 7B
- Will train faster and use less memory
- Update `config.json`: `"model_name": "Qwen/Qwen2.5-1.5B-Instruct"`


