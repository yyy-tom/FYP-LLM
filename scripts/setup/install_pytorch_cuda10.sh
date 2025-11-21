#!/bin/bash
# Script to install PyTorch with CUDA 10.0 support
# Run this on your HPC cluster

set -e

echo "Installing PyTorch with CUDA 10.0 support..."
echo "=========================================="

# Set CUDA paths
export CUDA_HOME=/usr/local/cuda-10.0
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Verify CUDA is accessible
if ! command -v nvcc &> /dev/null; then
    echo "ERROR: nvcc not found. CUDA 10.0 may not be properly installed."
    exit 1
fi

echo "CUDA version:"
nvcc --version

echo ""
echo "WARNING: CUDA 10.0 is very old (2018)."
echo "PyTorch 1.7.1 is the last version with CUDA 10.0 support."
echo "This may have compatibility issues with newer transformers and Python 3.12+."
echo ""

read -p "Continue with installation? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Installation cancelled."
    exit 1
fi

# Navigate to project directory
cd /research/d7/fyp25/yyyu2/FYP-LLM || exit 1

# Uninstall current PyTorch if exists
echo "Uninstalling current PyTorch..."
uv pip uninstall -y torch torchvision torchaudio 2>/dev/null || true

# Install PyTorch 1.7.1 with CUDA 10.0 support
# Use uv pip install (not uv add) because uv add doesn't support +cu100 version suffix
echo "Installing PyTorch 1.7.1+cu100..."
echo "Using PyTorch's official wheel repository..."

# Method 1: Direct URL installation (most reliable)
uv pip install torch==1.7.1+cu100 torchvision==0.8.2+cu100 torchaudio==0.7.2 \
    --index-url https://download.pytorch.org/whl/cu100

# If that doesn't work, try alternative method:
# uv pip install torch==1.7.1+cu100 torchvision==0.8.2+cu100 torchaudio==0.7.2 \
#     -f https://download.pytorch.org/whl/torch_stable.html

# Verify installation
echo ""
echo "Verifying installation..."
python -c "
import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('CUDA version:', torch.version.cuda)
    print('GPU count:', torch.cuda.device_count())
    print('GPU name:', torch.cuda.get_device_name(0))
else:
    print('WARNING: CUDA not available!')
"

echo ""
echo "Installation complete!"
echo ""
echo "IMPORTANT NOTES:"
echo "  - PyTorch 1.7.1 may have compatibility issues with:"
echo "    * transformers>=4.36.0 (you may need to downgrade to transformers<4.0)"
echo "    * Python 3.12+ (PyTorch 1.7.1 supports Python 3.6-3.9)"
echo ""
echo "  - Consider using Python 3.9 if possible for better compatibility"
echo "  - Or request access to nodes with newer GPUs/CUDA versions"
