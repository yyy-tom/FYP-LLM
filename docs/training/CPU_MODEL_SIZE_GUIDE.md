# CPU Training: Model Size vs RAM Requirements

## Key Point: CPU Cores ≠ Memory

**80 CPU cores** = Parallel computation power (helps with speed)
**RAM** = Memory storage (limits model size)

## Memory Requirements with LoRA (Parameter-Efficient Fine-Tuning)

With LoRA, you only train a small fraction of parameters, dramatically reducing memory:

| Model Size | RAM Required (LoRA) | Training Time (80 CPUs) | Config File |
|------------|---------------------|-------------------------|-------------|
| **0.5B**   | 2.5-4 GB           | 2-4 hours/epoch         | `config.json` |
| **1.5B**   | 7-10 GB            | 6-12 hours/epoch        | `config_7b_optimized.json` (currently 1.5B) |
| **3B**     | 18-25 GB           | 20-40 hours/epoch       | `config_3b_cpu.json` |
| **7B**     | 40-50 GB           | 60-120 hours/epoch      | `config_7b_cpu.json`, `config_7b_optimized.json` |
| **14B**    | 80-100 GB          | 150-300 hours/epoch     | `config_14b_optimized.json` |

## Can You Train 7B Model with 80 CPU Cores?

### ✅ YES, if you have enough RAM:

**Minimum Requirements:**
- **RAM**: 40-50 GB (with LoRA)
- **CPU Cores**: 80 ✅ (you have this)
- **Training Time**: 2.5-5 days per epoch (slow but doable)

**Recommended:**
- **RAM**: 64-128 GB
- **CPU Cores**: 80 ✅ (you have this)
- **Training Time**: Still slow but manageable

### ❌ NO, if you don't have enough RAM:

- Even with 80 CPU cores, if you only have 16-32 GB RAM, you can only train 0.5B-1.5B models
- CPU cores don't help if you run out of memory

## How to Check Your Available RAM

```bash
# On Linux/HPC
free -h

# Or
cat /proc/meminfo | grep MemTotal

# In Python
python3 -c "import psutil; print(f'Total RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB')"
```

## Config Files by Model Size

### Small Models (0.5B-1.5B)
- `config.json` - 0.5B (default)
- `config_cpu_80cores.json` - 0.5B (optimized)
- `config_memory_efficient.json` - 0.5B (low memory)
- `config_7b_optimized.json` - Currently 1.5B (can be changed to 7B)

### Medium Models (3B)
- `config_3b_cpu.json` - 3B model

### Large Models (7B-14B)
- `config_7b_cpu.json` - 7B model (NEW)
- `config_7b_optimized.json` - Can be changed to 7B
- `config_14b_optimized.json` - 14B model

## Recommendations by RAM

### If you have 16-32 GB RAM:
- ✅ **0.5B model**: Best choice
- ✅ **1.5B model**: Good choice
- ❌ **3B+ models**: Not enough RAM

### If you have 32-64 GB RAM:
- ✅ **0.5B model**: Fast
- ✅ **1.5B model**: Good
- ✅ **3B model**: Possible
- ⚠️ **7B model**: Tight, may work with small batch size

### If you have 64-128 GB RAM:
- ✅ **0.5B model**: Fast
- ✅ **1.5B model**: Good
- ✅ **3B model**: Good
- ✅ **7B model**: Definitely possible
- ⚠️ **14B model**: Possible but very slow

### If you have 128+ GB RAM:
- ✅ **7B model**: Great
- ✅ **14B model**: Possible

## Training Commands

### 7B Model (if you have 40-50 GB+ RAM):
```bash
uv run python scripts/training/train_qwen_counsel.py --config configs/config_7b_cpu.json
```

### 3B Model (if you have 18-25 GB+ RAM):
```bash
uv run python scripts/training/train_qwen_counsel.py --config configs/config_3b_cpu.json
```

### 1.5B Model (if you have 7-10 GB+ RAM):
```bash
uv run python scripts/training/train_qwen_counsel.py --config configs/config_7b_optimized.json
```

### 0.5B Model (if you have 2.5-4 GB+ RAM):
```bash
uv run python scripts/training/train_qwen_counsel.py --config configs/config.json
```

## Why I Initially Used 0.5B Models

I was being **too conservative** because:
1. I didn't know your available RAM
2. CPU training is much slower, so smaller models are more practical
3. But with 80 CPUs and enough RAM, you CAN train larger models!

## Summary

- **80 CPU cores** = Great for parallel computation ✅
- **RAM** = The real limiting factor for model size
- **With LoRA**, 7B model only needs ~40-50 GB RAM (not 128+ GB)
- **If you have 64+ GB RAM**, you can definitely train 7B models
- **Training will be slow** (days) but possible with 80 CPUs

**Check your RAM first, then choose the appropriate model size!**

