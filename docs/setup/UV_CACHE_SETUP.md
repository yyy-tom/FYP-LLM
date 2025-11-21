# Setting Up UV Cache to Avoid Disk Quota Issues

## Problem

When running `uv sync` or other uv commands, you get:
```
error: Failed to create download directory
Caused by: Disk quota exceeded (os error 122) at path "/uac/y22/yyyu2/.local/share/uv/..."
```

This is because `uv` is using the default cache location which hits your disk quota.

## Solution: Set UV Cache Environment Variables

### Quick Fix (Temporary)

Before running any `uv` commands, set the cache directory:

```bash
export BASE_DIR="/research/d7/fyp25/yyyu2"
export UV_HOME="$BASE_DIR/.cache/uv/home"      # For Python installations
export UV_CACHE_DIR="$BASE_DIR/.cache/uv/cache" # For package cache

# Create directories
mkdir -p "$UV_HOME" "$UV_CACHE_DIR"

# Now run uv commands
uv sync
```

### Permanent Fix

Add to your `~/.bashrc` or `~/.zshrc`:

```bash
export BASE_DIR="/research/d7/fyp25/yyyu2"
export UV_HOME="$BASE_DIR/.cache/uv/home"
export UV_CACHE_DIR="$BASE_DIR/.cache/uv/cache"
mkdir -p "$UV_HOME" "$UV_CACHE_DIR"
```

Then reload your shell:
```bash
source ~/.bashrc  # or source ~/.zshrc
```

### Or Use the Helper Script

```bash
# Source the script to set environment variables
source scripts/setup/setup_uv_cache.sh

# Then run uv commands
uv sync
```

## Installing PyTorch with CUDA 10.0

After setting up the cache, install PyTorch separately:

```bash
# Set CUDA paths
export CUDA_HOME=/usr/local/cuda-10.0
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Install PyTorch for CUDA 10.0
uv pip install torch==1.7.1+cu100 torchvision==0.8.2+cu100 torchaudio==0.7.2 \
    --index-url https://download.pytorch.org/whl/cu100
```

## Complete Setup Sequence

```bash
# 1. Set UV cache directories (CRITICAL - do this first!)
export BASE_DIR="/research/d7/fyp25/yyyu2"
export UV_HOME="$BASE_DIR/.cache/uv/home"      # For Python installations
export UV_CACHE_DIR="$BASE_DIR/.cache/uv/cache" # For package cache
mkdir -p "$UV_HOME" "$UV_CACHE_DIR"

# 2. Pin Python version (if not already done)
uv python pin 3.9

# 3. Sync dependencies (without torch)
uv sync

# 4. Install PyTorch separately with CUDA support
export CUDA_HOME=/usr/local/cuda-10.0
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

uv pip install torch==1.7.1+cu100 torchvision==0.8.2+cu100 torchaudio==0.7.2 \
    --index-url https://download.pytorch.org/whl/cu100

# 5. Verify installation
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

## Why PyTorch Must Be Installed Separately

- `pyproject.toml` doesn't support version suffixes like `+cu100`
- PyTorch with CUDA must be installed from PyTorch's custom index
- Use `uv pip install` (not `uv add`) for PyTorch with CUDA

## Verify Cache Location

Check where uv is caching:

```bash
# Check package cache
uv cache dir
# Should show: /research/d7/fyp25/yyyu2/.cache/uv/cache

# Check Python installation location
echo $UV_HOME
# Should show: /research/d7/fyp25/yyyu2/.cache/uv/home
```

## Troubleshooting

If you still get quota errors:

1. **Make sure you export the variables in the same shell session:**
   ```bash
   source scripts/setup/setup_uv_cache.sh
   # Then immediately run uv commands in the same shell
   ```

2. **Check if variables are set:**
   ```bash
   echo $UV_HOME
   echo $UV_CACHE_DIR
   ```

3. **If still failing, try setting XDG directories:**
   ```bash
   export XDG_DATA_HOME="/research/d7/fyp25/yyyu2/.local/share"
   export XDG_CACHE_HOME="/research/d7/fyp25/yyyu2/.cache"
   ```

