# CPU-Only Training Guide

## Understanding CUDA and GPU Training

**Important**: CUDA is required to use GPUs for training. Without CUDA, you can only train on CPU.

- ✅ **With CUDA**: Can use GPU (fast training)
- ❌ **Without CUDA**: Can only use CPU (very slow training)

## CPU-Only Training Setup

### Step 1: Install PyTorch CPU Version

```bash
# Install PyTorch CPU-only version
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Step 2: Verify CPU-Only Installation

```bash
python3 << 'EOF'
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
EOF
```

Expected output for CPU-only:
```
PyTorch version: 2.x.x
CUDA available: False
Device: CPU
```

### Step 3: Update Config for CPU Training

Update your `configs/config.json`:

```json
{
  "model_name": "Qwen/Qwen2.5-0.5B-Instruct",  // Use smaller model for CPU
  "use_4bit": false,  // Disable quantization (requires CUDA)
  "batch_size": 1,  // Very small batch size for CPU
  "gradient_accumulation_steps": 8,  // Compensate with gradient accumulation
  "fp16": false,
  "bf16": false,  // bfloat16 not well supported on CPU
  ...
}
```

### Step 4: Train on CPU

```bash
uv run python scripts/training/train_qwen_counsel.py --config configs/config.json
```

## CPU Training Considerations

### Model Size Recommendations

For CPU training, use smaller models:
- ✅ **Qwen2.5-0.5B-Instruct** (recommended for CPU)
- ✅ **Qwen2.5-1.5B-Instruct** (if you have enough RAM)
- ⚠️ **Qwen2.5-3B-Instruct** (very slow on CPU)
- ❌ **Qwen2.5-7B-Instruct+** (not practical on CPU)

### Training Speed

**Expected training times (approximate):**

| Model Size | CPU Training Time | GPU Training Time |
|------------|-------------------|-------------------|
| 0.5B       | ~10-20 hours      | ~1-2 hours        |
| 1.5B       | ~30-50 hours      | ~2-4 hours        |
| 3B         | ~100+ hours       | ~4-8 hours        |
| 7B         | Not practical     | ~8-16 hours       |

### Memory Requirements

CPU training uses system RAM instead of GPU memory:
- **0.5B model**: ~2-4 GB RAM
- **1.5B model**: ~6-8 GB RAM
- **3B model**: ~12-16 GB RAM
- **7B model**: ~28-32 GB RAM

### Optimized CPU Training Config

```json
{
  "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
  "dataset_path": "datasets/all_mental_health_combined",
  "output_dir": "models/qwen2.5-counsel-chat-cpu",
  "batch_size": 1,
  "gradient_accumulation_steps": 16,  // Effective batch size = 16
  "num_epochs": 1,  // Start with 1 epoch
  "learning_rate": 5e-4,
  "use_4bit": false,  // No quantization on CPU
  "max_length": 512,  // Shorter sequences for CPU
  "fp16": false,
  "bf16": false,
  "dataloader_num_workers": 2,  // Fewer workers for CPU
  "gradient_checkpointing": true,  // Save memory
  ...
}
```

## Can You Use GPU Without CUDA?

**No.** CUDA is the interface that allows PyTorch to communicate with NVIDIA GPUs. Without CUDA:
- ❌ Cannot use NVIDIA GPUs
- ❌ Cannot use GPU acceleration
- ✅ Can only train on CPU

## Alternatives

### Option 1: Get CUDA Working (Best)

If you have NVIDIA GPUs available:
1. Install CUDA toolkit
2. Install PyTorch with CUDA support
3. Use GPU for fast training

### Option 2: Use Cloud GPU Services

- Google Colab (free GPU)
- Kaggle Notebooks (free GPU)
- AWS/GCP/Azure (paid GPU instances)

### Option 3: CPU Training (Current)

- Use smaller models (0.5B-1.5B)
- Be patient (much slower)
- Use gradient accumulation to simulate larger batches

## Quick Start: CPU Training

```bash
# 1. Install PyTorch CPU
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 2. Verify CPU-only mode
python3 -c "import torch; print('CUDA:', torch.cuda.is_available())"

# 3. Update config.json (use smaller model, disable quantization)

# 4. Train
uv run python scripts/training/train_qwen_counsel.py --config configs/config.json
```

## Summary

- **CUDA = GPU access**: Without CUDA, no GPU training possible
- **CPU training is possible**: But much slower (10-100x slower)
- **Use smaller models**: 0.5B-1.5B recommended for CPU
- **Be patient**: CPU training takes much longer

The training script will automatically detect CPU-only mode and adjust accordingly.

