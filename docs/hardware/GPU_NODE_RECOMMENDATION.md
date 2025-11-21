# GPU Node Recommendation for Training

## Available GPU Nodes (from your cluster)

Based on the `sinfo` output and GPU specifications:

### Best Options (Recommended) ✅

#### 1. **RTX 3090 Nodes** (BEST CHOICE)

- **Nodes**: `projgpu7`, `projgpu8`, `projgpu12-14`, `projgpu21-26`
- **GPU**: RTX 3090 (24 GB VRAM)
- **OS**: Ubuntu 18
- **CPU:GPU Ratio**: 6:1 (projgpu7,8,12-14) or 10:1 (projgpu21-26 with highcpucount)
- **CUDA Support**: CUDA 11.0+ (supports PyTorch 2.0+)
- **Why Best**:
  - ✅ Modern GPU with 24GB VRAM (can fit 7B models with quantization)
  - ✅ Supports latest PyTorch/Transformers versions
  - ✅ High performance
  - ✅ Some nodes have highcpucount (projgpu21-26)

**Available nodes from sinfo:**

- `projgpu7` - Currently allocated (cwfu_gpu partition)
- `projgpu8` - Mix state (lwwang_gpu partition)
- `projgpu12-14` - Mix state (pheng_gpu, lwwang_gpu partitions)
- `projgpu21-26` - Not shown in sinfo (may be in different partition)

#### 2. **Titan RTX Nodes** (GOOD ALTERNATIVE)

- **Nodes**: `gpu54-59`
- **GPU**: Titan RTX (24 GB VRAM)
- **OS**: Ubuntu 24
- **CPU:GPU Ratio**: No Limit (Recommend 20:1) - **Great for 80 CPUs!**
- **CUDA Support**: CUDA 11.0+
- **Why Good**:
  - ✅ 24GB VRAM (same as RTX 3090)
  - ✅ **No CPU limit** - perfect for your 80 CPU requirement!
  - ✅ Modern OS (Ubuntu 24)
  - ✅ Highcpucount feature

**Available nodes from sinfo:**

- `gpu54` - Idle (ct2401 partition)
- `gpu55-56, 58-59` - Mix state (multiple partitions)
- `gpu57` - Idle (multiple partitions)

#### 3. **RTX 2080 Nodes** (GOOD FOR SMALLER MODELS)

- **Nodes**: `gpu36-39`, `gpu40-53`, `projgpu3-6`
- **GPU**: RTX 2080 (8-11 GB VRAM)
- **OS**: Ubuntu 18/24
- **CPU:GPU Ratio**: 6:1 or 10:1
- **CUDA Support**: CUDA 11.0+
- **Why Good**:
  - ✅ Modern CUDA support
  - ✅ Good for 0.5B-3B models
  - ✅ Some have highcpucount

**Available nodes from sinfo:**

- `gpu40-51` - Idle (gpu_2h, gpu_8h partitions)
- `projgpu3-6` - Idle (gpu_24h, pheng_gpu partitions)

### Avoid (Old CUDA 10.0) ❌

#### Titan X/XP/V Nodes

- **Nodes**: `gpu7-9` (Titan X), `gpu30-35` (Titan XP), `projgpu9,11,15-20` (Titan V/XP)
- **GPU**: Titan X/XP/V (12 GB VRAM)
- **OS**: CentOS 7 / Ubuntu 18
- **CUDA Support**: CUDA 10.0 only
- **Why Avoid**:
  - ❌ Only supports CUDA 10.0 (PyTorch 1.7.1)
  - ❌ Incompatible with transformers>=4.36.0
  - ❌ Cannot use Qwen2.5 models
  - ❌ Old hardware

## Recommendation by Use Case

### For 7B Model Training (Recommended)

**Best Choice**: RTX 3090 or Titan RTX nodes

- **Partition**: `gpu_24h` or `gpu_72h` (longer time limits)
- **Nodes**:
  - `projgpu12-14` (RTX 3090, if available)
  - `gpu54-59` (Titan RTX, **no CPU limit** - perfect for 80 CPUs!)
- **Request**: `--gres=gpu:1 --cpus-per-task=80`

### For 3B Model Training

**Good Choice**: RTX 2080 or RTX 3090 nodes

- **Partition**: `gpu_8h` or `gpu_24h`
- **Nodes**: `gpu40-51` (RTX 2080) or any RTX 3090
- **Request**: `--gres=gpu:1 --cpus-per-task=80`

### For 0.5B-1.5B Model Training

**Any Modern GPU**: RTX 2080, RTX 3090, or Titan RTX

- **Partition**: `gpu_2h` or `gpu_8h`
- **Nodes**: Any available modern GPU node
- **Request**: `--gres=gpu:1 --cpus-per-task=80`

## SLURM Job Script Examples

### Example 1: RTX 3090 (projgpu12-14) - 7B Model

```bash
#!/bin/bash
#SBATCH --job-name=qwen-7b-train
#SBATCH --partition=gpu_24h
#SBATCH --nodelist=projgpu12  # or projgpu13, projgpu14
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=80
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

cd /path/to/FYP-LLM
uv run python scripts/training/train_qwen_counsel.py --config configs/config_7b_cpu.json
```

### Example 2: Titan RTX (gpu54-59) - 7B Model (BEST FOR 80 CPUs)

```bash
#!/bin/bash
#SBATCH --job-name=qwen-7b-train
#SBATCH --partition=gpu_24h
#SBATCH --nodelist=gpu54  # or gpu55-59
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=80  # No limit on Titan RTX!
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

cd /path/to/FYP-LLM
uv run python scripts/training/train_qwen_counsel.py --config configs/config_7b_cpu.json
```

### Example 3: RTX 2080 (gpu40-51) - 3B Model

```bash
#!/bin/bash
#SBATCH --job-name=qwen-3b-train
#SBATCH --partition=gpu_8h
#SBATCH --nodelist=gpu40  # or any gpu40-51
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=80
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

cd /path/to/FYP-LLM
uv run python scripts/training/train_qwen_counsel.py --config configs/config_3b_cpu.json
```

## Quick Decision Guide

**If you want to train 7B models:**

1. ✅ **Titan RTX (gpu54-59)** - Best choice (no CPU limit, 24GB VRAM)
2. ✅ **RTX 3090 (projgpu12-14)** - Good alternative (24GB VRAM)

**If you want to train 3B models:**

1. ✅ **RTX 2080 (gpu40-51)** - Good choice (8-11GB VRAM, many available)
2. ✅ **RTX 3090** - Overkill but works

**If you want to train 0.5B-1.5B models:**

1. ✅ **Any modern GPU** - RTX 2080, RTX 3090, or Titan RTX

## Summary

**Top Recommendation**: **Titan RTX nodes (gpu54-59)**

- ✅ 24GB VRAM (can fit 7B models)
- ✅ **No CPU limit** (perfect for your 80 CPU requirement!)
- ✅ Modern CUDA support
- ✅ Available nodes: gpu54 (idle), gpu57 (idle)

**Alternative**: **RTX 3090 nodes (projgpu12-14)**

- ✅ 24GB VRAM
- ✅ Modern CUDA support
- ⚠️ CPU limit: 6:1 or 10:1 (may need to request fewer CPUs)

**Avoid**: **Titan X/XP/V nodes**

- ❌ CUDA 10.0 only
- ❌ Cannot use Qwen2.5 models
