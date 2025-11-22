# 🔥 Blazing Fast Training Guide

This guide shows you how to train the model as **FAST AS POSSIBLE** with minimal quality.

## ⚠️ Warning

These configurations prioritize **SPEED OVER QUALITY**. The resulting models will have significantly lower performance than standard training. Use these for:
- Quick prototyping
- Testing pipelines
- Rapid iteration
- Code verification

## 🚀 Available Blazing Fast Configs

### 1. Single GPU Blazing Fast (10-15 minutes)
**Config**: `configs/config_1.5b_single_gpu_blazing.json`

```bash
# Direct Python
python scripts/training/train_qwen_counsel_multi_gpu.py \
    --config configs/config_1.5b_single_gpu_blazing.json

# Or with tmux
./scripts/jobs/run_training_tmux.sh configs/config_1.5b_single_gpu_blazing.json
```

**Optimizations**:
- Max length: 128 tokens (8x faster)
- LoRA rank: 4 (minimal)
- Only 2 LoRA modules
- Batch size: 32 (maximum throughput)
- 1 epoch only
- No gradient checkpointing

---

### 2. Multi-GPU Blazing Fast (30-45 minutes on 8 GPUs)
**Config**: `configs/config_1.5b_blazing_fast.json`

```bash
# Submit to SLURM
sbatch scripts/jobs/train_1.5b_blazing_fast.sh

# Or use accelerate directly
accelerate launch --multi_gpu --num_processes=8 \
    scripts/training/train_qwen_counsel_multi_gpu.py \
    --config configs/config_1.5b_blazing_fast.json
```

**Optimizations**:
- Max length: 256 tokens (4x faster)
- LoRA rank: 8 (minimal)
- Only 2 LoRA modules (q_proj, v_proj)
- Batch size: 8 per GPU
- 1 epoch only
- No gradient checkpointing

---

## 📊 Speed Comparison

| Config | GPUs | Time | Max Length | LoRA Rank | Modules | Quality |
|--------|------|------|------------|-----------|---------|---------|
| **Single GPU Blazing** | 1 | 10-15 min | 128 | 4 | 2 | ⭐ |
| **Multi-GPU Blazing** | 8 | 30-45 min | 256 | 8 | 2 | ⭐⭐ |
| Single GPU Ultra Fast | 1 | 45-60 min | 1024 | 32 | 7 | ⭐⭐⭐ |
| Multi-GPU Fast | 8 | 1.5-2 hr | 768 | 16 | 7 | ⭐⭐⭐⭐ |
| Standard Training | 8 | 4-6 hr | 1024 | 32 | 7 | ⭐⭐⭐⭐⭐ |

## 🎯 What Makes It Fast?

### 1. **Minimal Sequence Length**
- Single GPU: 128 tokens (vs 1024 standard)
- Multi-GPU: 256 tokens (vs 1024 standard)
- **Impact**: 4-8x faster per batch

### 2. **Tiny LoRA**
- Rank 4-8 instead of 16-32
- Only 2 modules (q_proj, v_proj) instead of 7
- **Impact**: 3-4x fewer trainable parameters

### 3. **No Gradient Checkpointing**
- Uses more memory but much faster
- **Impact**: 20-30% speed boost

### 4. **Aggressive Batching**
- Maximum batch size that fits in memory
- Minimal gradient accumulation
- **Impact**: Better GPU utilization

### 5. **Minimal Evaluation**
- Rare eval steps (200-500 steps)
- No early stopping
- Only 1 checkpoint saved
- **Impact**: 10-15% less overhead

### 6. **Simple Training**
- Only 1 epoch
- No warmup
- Constant learning rate
- High initial LR (5e-4)
- **Impact**: Minimal schedule overhead

## 🛠️ Customization

Want even faster? Edit the config:

```json
{
  "num_epochs": 1,           // Already at minimum
  "max_length": 64,          // 🚀 Even shorter (2x faster, very poor quality)
  "lora_r": 2,               // 🚀 Minimum rank (may break)
  "batch_size": 64,          // 🚀 Increase if you have memory
  "eval_steps": 1000,        // 🚀 Never evaluate (save more time)
  "save_steps": 10000,       // 🚀 Never save during training
  "gradient_checkpointing": false,  // Already disabled
  "lora_target_modules": ["q_proj"]  // 🚀 Only 1 module (50% faster, worse quality)
}
```

## 📝 Notes

1. **Memory Usage**: 
   - Single GPU: ~8-9 GB VRAM
   - Multi-GPU: ~6-7 GB per GPU

2. **Quality Trade-offs**:
   - Short sequences = model can't learn long context
   - Small LoRA = less adaptation capacity
   - 1 epoch = underfitting
   - No warmup = unstable initial training

3. **Best Use Cases**:
   - Pipeline testing
   - Hyperparameter search (structure, not values)
   - Code debugging
   - Quick proof-of-concept

4. **NOT Recommended For**:
   - Production models
   - Final evaluation
   - Research experiments requiring quality
   - Comparing model architectures

## 🎮 Quick Start Commands

**Test pipeline in 10 minutes:**
```bash
python scripts/training/train_qwen_counsel_multi_gpu.py \
    --config configs/config_1.5b_single_gpu_blazing.json
```

**Fast training on cluster (30 min):**
```bash
sbatch scripts/jobs/train_1.5b_blazing_fast.sh
```

**Monitor progress:**
```bash
# Watch logs
tail -f logs/train_blazing_*.out

# Check GPU usage
watch -n 1 nvidia-smi
```

## 🔄 Checkpoint Behavior

By default, the blazing fast configs have **`ignore_checkpoints: true`**, which means:
- ✅ **Always starts fresh training** - ignores any existing checkpoints
- ✅ **No resume overhead** - faster startup
- ✅ **Clean slate** - perfect for quick iterations

If you want to **resume from checkpoints** instead:
```json
{
  "ignore_checkpoints": false
}
```

To **manually delete old checkpoints** before training:
```bash
# Remove old model outputs
rm -rf models/qwen2.5-1.5b-blazing-fast/checkpoint-*

# Or delete the entire output directory
rm -rf models/qwen2.5-1.5b-blazing-fast
```

## 🔍 Troubleshooting

**Out of memory?**
- Reduce `batch_size` (16 → 8)
- Reduce `max_length` (256 → 128)
- Enable `gradient_checkpointing: true`

**Training too slow?**
- Check GPU utilization: `nvidia-smi dmon`
- Increase `dataloader_num_workers` (8 → 16)
- Disable all evaluation: `eval_steps: 100000`

**Model quality terrible?**
- This is expected! Use standard configs for quality
- Or increase to 2-3 epochs
- Or increase max_length to 512

**Accidentally resuming from old checkpoint?**
- Set `ignore_checkpoints: true` in config
- Or delete old checkpoints manually (see above)

---

**Remember**: These configs are for SPEED ONLY. For production-quality models, use the standard training configurations!

