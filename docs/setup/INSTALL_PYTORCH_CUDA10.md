# Installing PyTorch with CUDA 10.0 Support

## Problem

`uv add` doesn't support PyTorch's version format with CUDA suffix (e.g., `1.7.1+cu100`). You need to use `uv pip install` instead.

## Solution: Use `uv pip install` with PyTorch Index

### Quick Install

```bash
cd /research/d7/fyp25/yyyu2/FYP-LLM

# Set CUDA paths
export CUDA_HOME=/usr/local/cuda-10.0
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Install PyTorch for CUDA 10.0 using uv pip (not uv add)
uv pip install torch==1.7.1+cu100 torchvision==0.8.2+cu100 torchaudio==0.7.2 \
    --index-url https://download.pytorch.org/whl/cu100
```

### Or Use the Helper Script

```bash
chmod +x scripts/setup/install_pytorch_cuda10.sh
./scripts/setup/install_pytorch_cuda10.sh
```

## Why `uv add` Doesn't Work

- `uv add` uses standard PyPI versioning
- PyTorch uses custom version format: `1.7.1+cu100` (CUDA suffix)
- You need to install from PyTorch's custom index URL

## Important Compatibility Issues

⚠️ **PyTorch 1.7.1 has major compatibility issues:**

1. **Python Version**: PyTorch 1.7.1 supports Python 3.6-3.9 only

   - Your project requires Python 3.11-3.13
   - **Solution**: Use Python 3.9 or downgrade Python requirement

2. **Transformers**: transformers>=4.36.0 requires PyTorch 2.0+

   - PyTorch 1.7.1 is from 2020
   - **Solution**: Downgrade transformers to <4.0

3. **Modern Dependencies**: Many packages require newer PyTorch

## Recommended Approach

### Option 1: Use Python 3.9 (Best if Available)

```bash
# Update pyproject.toml
requires-python = ">=3.9,<3.10"

# Then install PyTorch 1.7.1
uv pip install torch==1.7.1+cu100 torchvision==0.8.2+cu100 torchaudio==0.7.2 \
    --index-url https://download.pytorch.org/whl/cu100

# Downgrade transformers
uv pip install "transformers<4.0"
```

### Option 2: Request Newer GPU Nodes (Best Long-term)

Check if your cluster has nodes with:

- Newer GPUs (V100, A100, RTX series)
- CUDA 11.x or 12.x
- Better PyTorch support

### Option 3: Use CPU-Only PyTorch (Very Slow)

```bash
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## Verify Installation

After installation:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')"
```

## Update pyproject.toml

I've updated your `pyproject.toml` to comment out the torch dependency since it needs to be installed separately via pip for CUDA support.

## Summary

- ✅ Use `uv pip install` (not `uv add`) for PyTorch with CUDA
- ⚠️ PyTorch 1.7.1 only supports Python 3.6-3.9
- ⚠️ You'll need to downgrade transformers and other dependencies
- 💡 Best solution: Get access to newer GPU nodes with modern CUDA
