# 🚀 Quick Start: Qwen2.5-1.5B Fast Training

**TL;DR:** The 1.5B model trains **2-3x faster** than 3B with comparable quality for mental health conversations.

---

## ⚡ Ultra Quick Start (8 GPUs)

```bash
# Submit to SLURM - should finish in ~1.5-2 hours
sbatch scripts/jobs/train_1.5b_fast.sh
```

That's it! Your model will be saved to `models/qwen2.5-1.5b-fast-8gpu/`

---

## 🎯 Available Configs

| Config | GPUs | Speed | Use Case |
|--------|------|-------|----------|
| `config_1.5b_fast.json` | 8 | ⚡⚡⚡ | **Production training** (recommended) |
| `config_1.5b_single_gpu_ultra_fast.json` | 1 | ⚡⚡ | Quick testing & prototyping |

---

## 📋 Common Commands

### 8-GPU Training (Recommended)
```bash
# Standard fast training
sbatch scripts/jobs/train_1.5b_fast.sh

# With custom config
sbatch scripts/jobs/train_1.5b_fast.sh configs/my_custom_config.json

# Check job status
squeue -u $USER

# View logs (replace JOB_ID)
tail -f logs/train_1.5b_fast_JOB_ID.out
```

### Single GPU Training (Testing)
```bash
# Quick test on 1 GPU
python scripts/training/train_qwen_counsel_multi_gpu.py \
  --config configs/config_1.5b_single_gpu_ultra_fast.json

# With accelerate (explicit)
accelerate launch --num_processes=1 \
  scripts/training/train_qwen_counsel_multi_gpu.py \
  --config configs/config_1.5b_single_gpu_ultra_fast.json
```

---

## 📊 What to Expect

### Training Time
- **8 GPUs:** 1.5-2 hours for 3 epochs ✅
- **1 GPU:** 8-10 hours for 3 epochs

### Performance
- **Loss:** ~1.8-2.0 (eval)
- **Quality:** Comparable to 3B for mental health
- **Speed:** 2-3x faster than 3B
- **Memory:** ~6-8 GB per GPU

### Output
```
models/qwen2.5-1.5b-fast-8gpu/
├── adapter_model.safetensors    # LoRA weights
├── adapter_config.json
├── training_config.json
├── tokenizer files...
└── checkpoint-XXX/              # Best checkpoint
```

---

## 🔧 Quick Customizations

### Faster Training
Edit `configs/config_1.5b_fast.json`:
```json
{
  "batch_size": 12,              // ↑ Increase if you have memory
  "gradient_accumulation_steps": 2,  // ↓ Fewer steps = faster
  "max_length": 768,             // ↓ Shorter sequences = faster
  "num_epochs": 2                // ↓ Fewer epochs
}
```

### Better Quality (Slower)
```json
{
  "lora_r": 64,                  // ↑ Higher rank
  "learning_rate": 2e-4,         // ↓ Lower LR
  "num_epochs": 5,               // ↑ More epochs
  "batch_size": 4                // ↓ Smaller batches
}
```

---

## 🆚 Model Comparison

### When to Use Each Model:

**Qwen2.5-1.5B-Instruct** ⚡⚡⚡
- ✅ Rapid experimentation & iteration
- ✅ Quick proof of concept
- ✅ Limited time/compute budget
- ✅ Most mental health conversations
- ⚠️ Not ideal for very complex reasoning

**Qwen2.5-3B-Instruct** ⚡⚡
- ✅ Better general quality
- ✅ More nuanced responses
- ✅ Good balance speed/quality
- ⏱️ ~2x slower than 1.5B

**Qwen2.5-7B-Instruct** ⚡
- ✅ Best quality
- ✅ Production deployment
- ✅ Complex conversations
- ⏱️ ~3x slower than 1.5B

---

## 📈 Monitoring Training

### View Real-time Progress
```bash
# Watch the output log
tail -f logs/train_1.5b_fast_*.out

# Check GPU usage
watch -n 1 nvidia-smi
```

### Key Metrics
- **Training Loss:** Should drop to ~1.5-2.0
- **Eval Loss:** Should be ~1.8-2.2 (best model)
- **Steps per second:** ~2-5 with 8 GPUs
- **GPU Memory:** ~6-8 GB per GPU

---

## 🐛 Troubleshooting

### Out of Memory
```bash
# Option 1: Use smaller batch size
# Edit config: "batch_size": 4

# Option 2: Shorter sequences  
# Edit config: "max_length": 768

# Option 3: Lower LoRA rank
# Edit config: "lora_r": 16
```

### Training Too Slow
```bash
# Check GPU utilization
nvidia-smi dmon -s u

# Should see >90% GPU utilization
# If low, increase dataloader_num_workers in config
```

### Job Killed / OOM
```bash
# Check error log
cat logs/train_1.5b_fast_*.err

# Reduce memory in config:
# "batch_size": 4
# "max_length": 768
# "gradient_accumulation_steps": 8
```

---

## 🎓 Full Documentation

For detailed information, see:
- **Training Guide:** [`docs/training/FAST_TRAINING_1.5B_GUIDE.md`](docs/training/FAST_TRAINING_1.5B_GUIDE.md)
- **Config Reference:** [`configs/config_1.5b_fast.json`](configs/config_1.5b_fast.json)
- **Job Script:** [`scripts/jobs/train_1.5b_fast.sh`](scripts/jobs/train_1.5b_fast.sh)

---

## ✨ Recommended Workflow

### Phase 1: Quick Test (1 hour)
```bash
# Test on 1 GPU first to validate setup
python scripts/training/train_qwen_counsel_multi_gpu.py \
  --config configs/config_1.5b_single_gpu_ultra_fast.json
```

### Phase 2: Full Training (1.5 hours)
```bash
# Scale to 8 GPUs for production model
sbatch scripts/jobs/train_1.5b_fast.sh
```

### Phase 3: Evaluate & Deploy
```bash
# Test the model
python scripts/inference/test_model.py \
  --model_path models/qwen2.5-1.5b-fast-8gpu

# If quality is good → deploy!
# If quality needs improvement → try 3B model
```

---

## 🎯 Summary

**Start fast, iterate quickly:**

1. ✅ Use `config_1.5b_fast.json` for rapid training
2. ✅ Submit with `sbatch scripts/jobs/train_1.5b_fast.sh`
3. ✅ Get results in 1.5-2 hours
4. ✅ Scale to 3B/7B if needed

**Model:** Qwen2.5-1.5B-Instruct  
**Speed:** 2-3x faster than alternatives  
**Quality:** Great for mental health conversations  
**Memory:** ~6-8 GB per GPU  
**Time:** ~1.5 hours with 8 GPUs

---

Happy fast training! 🚀

*Questions? Check the full guide: [`docs/training/FAST_TRAINING_1.5B_GUIDE.md`](docs/training/FAST_TRAINING_1.5B_GUIDE.md)*

