# Installing autoawq for AWQ Models

If you're using AWQ quantized models (like `Qwen/Qwen2.5-7B-Instruct-AWQ`), you need to install the `autoawq` library.

## Quick Install

On your HPC cluster, activate your virtual environment and install:

```bash
# If using uv (recommended)
uv pip install autoawq

# Or if using pip directly
pip install autoawq
```

## For CUDA Environments

If you have CUDA available, you can install the CUDA-enabled version:

```bash
# With uv
uv pip install autoawq

# Or with pip
pip install autoawq
```

## Verify Installation

After installing, verify it works:

```bash
python -c "import awq; print('AWQ installed successfully')"
```

## Troubleshooting

### Installation fails with compilation errors

If you encounter compilation errors, try:

```bash
# Install build dependencies first
pip install ninja

# Then install autoawq
pip install autoawq
```

### CUDA version mismatch

If you get CUDA-related errors, make sure your CUDA version matches:

```bash
# Check CUDA version
nvcc --version

# Check PyTorch CUDA version
python -c "import torch; print(torch.version.cuda)"
```

### Alternative: Use non-AWQ model

If you continue having issues with AWQ, you can switch to a regular model:

Update `configs/config.json`:
```json
{
  "model_name": "Qwen/Qwen2.5-7B-Instruct",  // Remove -AWQ
  "use_4bit": true,  // This will use BitsAndBytes instead
  ...
}
```

Then you'll need `bitsandbytes` instead of `autoawq`:
```bash
uv pip install bitsandbytes
```

