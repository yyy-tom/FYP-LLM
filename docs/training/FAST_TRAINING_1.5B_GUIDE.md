# Fast Training Guide: Qwen2.5-1.5B-Instruct ⚡

## 🚀 Quick Start

The **Qwen2.5-1.5B-Instruct** model is optimized for **maximum training speed** while maintaining excellent quality for mental health conversations.

### Why 1.5B for Fast Training?

- **2-3x faster** than 3B models
- **4-6x faster** than 7B models  
- Higher batch sizes = faster convergence
- Lower memory = more room for optimization
- **Same architecture quality** as larger Qwen2.5 models

---

## 📊 Training Speed Comparison

| Model | Relative Speed | Time per Epoch* | Quality | Memory/GPU |
|-------|---------------|-----------------|---------|------------|
| **Qwen2.5-1.5B** | **3.0x** ⚡⚡⚡ | ~30 min | ⭐⭐⭐ Good | ~6 GB |
| Qwen2.5-3B | 1.5x | ~60 min | ⭐⭐⭐⭐ Better | ~10 GB |
| Qwen2.5-7B | 1.0x | ~90 min | ⭐⭐⭐⭐⭐ Best | ~18 GB |

*Estimated on 8x RTX 2080 Ti GPUs with your dataset (~30K samples)

---

## 🎯 Configuration Options

### Option 1: Multi-GPU (8 GPUs) - RECOMMENDED
**Config:** `configs/config_1.5b_fast.json`

**Features:**
- Batch size: 8 per GPU
- Effective batch: 256 (8 GPUs × 8 batch × 4 accumulation)
- Higher LoRA rank (32) for better quality
- Training time: ~1.5 hours for 3 epochs

**Usage:**
```bash
# Submit to SLURM with 8 GPUs
sbatch scripts/jobs/train_job_max_gpu_72h.sh configs/config_1.5b_fast.json

# Or run directly with accelerate
accelerate launch --multi_gpu --num_processes=8 \
  scripts/training/train_qwen_counsel_multi_gpu.py \
  --config configs/config_1.5b_fast.json
```

### Option 2: Single GPU - ULTRA FAST PROTOTYPING
**Config:** `configs/config_1.5b_single_gpu_ultra_fast.json`

**Features:**
- Batch size: 16 per GPU (very high!)
- Effective batch: 32 (1 GPU × 16 batch × 2 accumulation)
- Perfect for testing and iteration
- Training time: ~8-10 hours for 3 epochs

**Usage:**
```bash
# Single GPU training
python scripts/training/train_qwen_counsel_multi_gpu.py \
  --config configs/config_1.5b_single_gpu_ultra_fast.json
```

---

## ⚙️ Key Optimizations Explained

### 1. **High Batch Size (8-16)**
- Smaller model = more samples fit in memory
- Fewer optimizer steps = faster training
- Better gradient estimates = faster convergence

### 2. **Aggressive LoRA Settings**
- Rank 32 (vs 8 for 7B models)
- Alpha 64 for strong adaptation
- All attention + MLP layers targeted
- No slowdown on small models!

### 3. **Fast Learning Schedule**
- Learning rate: 3e-4 (higher than typical)
- Warmup: 5% (quick start)
- Cosine decay for smooth convergence

### 4. **Optimized Data Loading**
- 4 workers for parallel loading
- Pin memory for faster GPU transfer
- Prediction loss only for faster eval

### 5. **Memory Efficiency**
- 4-bit quantization still used
- Gradient checkpointing enabled
- Can still use higher batch sizes!

---

## 📈 Expected Results

### Training Metrics (3 epochs)
- **Training time:** 1.5-2 hours (8 GPUs) or 8-10 hours (1 GPU)
- **Final eval loss:** ~1.8-2.0 (similar to 3B model)
- **Perplexity:** ~6-7 on mental health domain
- **Trainable params:** ~25M (with LoRA rank 32)

### Quality Comparison
- **Mental health responses:** Comparable to 3B for most conversations
- **Following instructions:** Excellent (inherits from Qwen2.5 training)
- **Empathy & tone:** Very good for 1.5B size
- **Complex reasoning:** Slightly behind 3B/7B (expected)

---

## 🎨 Customization Tips

### Make it Even Faster
```json
{
  "batch_size": 12,              // Increase if you have memory
  "gradient_accumulation_steps": 2,  // Reduce for more frequent updates
  "max_length": 768,             // Reduce if most responses are shorter
  "eval_steps": 100,             // Evaluate less often
  "save_steps": 300,             // Save less often
  "warmup_ratio": 0.03          // Even shorter warmup
}
```

### Improve Quality (Slower)
```json
{
  "batch_size": 4,               // Smaller batches
  "gradient_accumulation_steps": 8,  // More accumulation
  "lora_r": 64,                  // Higher rank
  "max_length": 1536,            // Longer context
  "learning_rate": 2e-4,         // Lower learning rate
  "num_epochs": 5                // More epochs
}
```

### For Longer Conversations
```json
{
  "max_length": 2048,            // Support longer context
  "batch_size": 4,               // Reduce to fit memory
  "gradient_accumulation_steps": 8
}
```

---

## 🔍 Monitoring Training

### Key Metrics to Watch
1. **Training Loss:** Should drop to ~1.5-2.0
2. **Eval Loss:** Should be ~1.8-2.2 (best model)
3. **Learning Rate:** Peaks after warmup, then decays
4. **GPU Memory:** Should stay under 8GB per GPU

### Signs of Good Training
✅ Training loss decreases steadily  
✅ Eval loss tracks training loss (no huge gap)  
✅ Early stopping triggers after 3-5 checkpoints  
✅ Generated responses are coherent and empathetic

### Warning Signs
⚠️ Eval loss increases while training decreases (overfitting)  
⚠️ Loss plateaus very early (learning rate too low)  
⚠️ Loss oscillates wildly (learning rate too high)  
⚠️ OOM errors (reduce batch_size or max_length)

---

## 🚨 Troubleshooting

### Out of Memory (OOM)
```json
// Reduce memory usage:
{
  "batch_size": 4,               // Smaller batches
  "max_length": 768,             // Shorter sequences
  "gradient_checkpointing": true,  // Already enabled
  "lora_r": 16                   // Reduce rank
}
```

### Training Too Slow
```json
// Speed up:
{
  "dataloader_num_workers": 8,   // More workers
  "eval_steps": 100,             // Evaluate less
  "logging_steps": 10,           // Log less
  "prediction_loss_only": true   // Already enabled
}
```

### Quality Not Good Enough
- Try 3B model instead: `config_3b_cpu.json`
- Increase LoRA rank to 64
- Train for more epochs (5-6)
- Use lower learning rate (2e-4)

---

## 📝 Quick Comparison with Your Current Configs

| Config | Model | GPUs | Batch | Eff. Batch | Speed | Best For |
|--------|-------|------|-------|-----------|-------|----------|
| **config_1.5b_fast.json** | 1.5B | 8 | 8 | 256 | ⚡⚡⚡ Fastest | **Rapid development** |
| **config_1.5b_single_gpu.json** | 1.5B | 1 | 16 | 32 | ⚡⚡ Very Fast | **Testing/prototyping** |
| config_3b_cpu.json | 3B | 1-4 | 4 | 16 | ⚡ Fast | Balanced quality |
| config_7b_8gpu.json | 7B | 8 | 1 | 64 | Standard | Production quality |

---

## 🎯 Recommended Workflow

### Phase 1: Fast Experimentation (1.5B)
1. Use **config_1.5b_single_gpu_ultra_fast.json** on 1 GPU
2. Test different hyperparameters quickly
3. Validate your dataset and preprocessing
4. **Time:** Few hours per run

### Phase 2: Multi-GPU Training (1.5B)
1. Switch to **config_1.5b_fast.json** with 8 GPUs
2. Train with best hyperparameters from Phase 1
3. Get a production-ready model fast
4. **Time:** 1-2 hours per run

### Phase 3: Optional - Scale to 3B/7B
1. If quality needs improvement, use 3B or 7B
2. Use same hyperparameters as 1.5B
3. Expect 2-3x longer training time
4. **Time:** 4-6 hours (3B) or 8-12 hours (7B)

---

## 🎉 Summary

**For maximum training speed, use Qwen2.5-1.5B-Instruct:**

✅ **2-3x faster** than your current 3B/7B models  
✅ **Higher batch sizes** for faster convergence  
✅ **Same quality** for most mental health conversations  
✅ **Perfect for rapid iteration** and experimentation  
✅ **Ready to scale** to 3B/7B if needed

**Start with:**
```bash
# 8 GPU training (fastest)
sbatch scripts/jobs/train_job_max_gpu_72h.sh configs/config_1.5b_fast.json

# OR 1 GPU testing (ultra fast prototyping)
python scripts/training/train_qwen_counsel_multi_gpu.py \
  --config configs/config_1.5b_single_gpu_ultra_fast.json
```

Happy fast training! 🚀

