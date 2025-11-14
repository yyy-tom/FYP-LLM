# CPU-Only Training Setup for 80 CPU Cores

## Overview

All training configurations have been updated for CPU-only training optimized for 80 CPU cores. Since you're not using CUDA, all configs use:
- **Smaller models** (0.5B-1.5B) suitable for CPU training
- **No quantization** (use_4bit: false)
- **Float32 precision** (fp16/bf16: false)
- **High dataloader workers** (24-32) to utilize multiple CPUs
- **Gradient accumulation** to simulate larger batch sizes

## Updated Configurations

### Main Configs

1. **`configs/config.json`** - Default config (0.5B model, 32 workers)
2. **`configs/config_cpu_80cores.json`** - Optimized for 80 cores (recommended)
3. **`configs/config_memory_efficient.json`** - Lower memory usage (1 batch, 24 workers)
4. **`configs/config_7b_optimized.json`** - 1.5B model variant
5. **`configs/config_14b_optimized.json`** - 1.5B model variant
6. **`configs/config_32b_tesla_t4.json`** - 0.5B model, minimal config

## Key Settings for 80 CPU Cores

```json
{
  "dataloader_num_workers": 32,  // Uses 32 CPUs for data loading
  "batch_size": 2,                // Small batch size
  "gradient_accumulation_steps": 16,  // Effective batch = 32
  "use_4bit": false,              // No quantization (requires CUDA)
  "fp16": false,                  // Use float32
  "bf16": false,                  // Use float32
  "optim": "adamw_torch",         // Standard optimizer (not fused)
  "dataloader_pin_memory": false  // Not needed for CPU
}
```

## Installation

### Step 1: Install PyTorch CPU Version

```bash
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Step 2: Verify CPU-Only Mode

```bash
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

Should output: `CUDA available: False`

### Step 3: Install Dependencies

```bash
uv sync
```

## Training

### Recommended Config (80 Cores)

```bash
uv run python scripts/train_qwen_counsel.py --config configs/config_cpu_80cores.json
```

### Default Config

```bash
uv run python scripts/train_qwen_counsel.py --config configs/config.json
```

### Memory-Efficient Config

```bash
uv run python scripts/train_qwen_counsel.py --config configs/config_memory_efficient.json
```

## Model Recommendations

For CPU training with 80 cores:

| Model Size | Training Time (approx) | RAM Required | Recommended |
|------------|------------------------|--------------|-------------|
| 0.5B       | 10-20 hours            | 2-4 GB       | ✅ Best     |
| 1.5B       | 30-50 hours            | 6-8 GB       | ✅ Good     |
| 3B         | 100+ hours             | 12-16 GB     | ⚠️ Slow     |
| 7B+        | Not practical          | 28+ GB       | ❌ Avoid    |

## CPU Utilization

With 80 CPUs:
- **32 workers** for data loading (configs/config.json, config_cpu_80cores.json)
- **24 workers** for memory-efficient configs
- **16 workers** for minimal configs
- Remaining CPUs used for model training operations

## Performance Tips

1. **Use smaller models**: 0.5B-1.5B are practical for CPU
2. **Increase gradient accumulation**: Compensates for small batch sizes
3. **Use gradient checkpointing**: Saves memory (already enabled)
4. **Monitor CPU usage**: Should see high utilization across all cores
5. **Be patient**: CPU training is 10-100x slower than GPU

## Updated Dependencies

All dependencies in `pyproject.toml` have been updated to latest versions:
- `transformers>=4.45.0` (latest)
- `accelerate>=1.0.0` (latest)
- `peft>=0.12.0` (latest)
- `datasets>=2.20.0` (latest)
- And other packages to latest compatible versions

## HPC Job Submission

If using SLURM, you can request 80 CPUs:

```bash
#!/bin/bash
#SBATCH --job-name=qwen-cpu-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --mem=64G
#SBATCH --time=48:00:00

cd /path/to/FYP-LLM
uv run python scripts/train_qwen_counsel.py --config configs/config_cpu_80cores.json
```

## Summary

✅ All configs updated for CPU-only training
✅ Optimized for 80 CPU cores
✅ Using latest package versions (no CUDA 10.0 limitation)
✅ Smaller models (0.5B-1.5B) for practical CPU training
✅ High dataloader workers (24-32) to utilize multiple CPUs

Ready to train on CPU with 80 cores!

