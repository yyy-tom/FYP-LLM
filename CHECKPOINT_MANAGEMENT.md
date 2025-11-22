# 📁 Checkpoint Management Guide

This guide explains how to control checkpoint loading and resumption behavior in your training runs.

## 🔄 Default Behavior

By default, the training script **automatically resumes from checkpoints** if they exist in the output directory:

```python
# Training script automatically checks for checkpoints
output_dir = "models/my-model/"
# If checkpoint-100 exists → automatically resumes from step 100
```

This is useful for:
- ✅ Recovering from crashes
- ✅ Continuing long training runs
- ✅ Adding more epochs to existing training

## 🚫 Ignoring Checkpoints (Starting Fresh)

If you want to **ignore existing checkpoints** and start fresh training, add this to your config:

```json
{
  "ignore_checkpoints": true
}
```

### When to Use `ignore_checkpoints: true`

Use this when you want to:
- 🔄 **Start completely fresh** - ignore all previous training
- 🧪 **Test hyperparameters** - don't resume from old runs
- ⚡ **Quick iterations** - blazing fast configs use this by default
- 🐛 **Debug training** - eliminate checkpoint confusion

### Configs with `ignore_checkpoints: true` by Default

These configs start fresh every time:
- `config_1.5b_blazing_fast.json` - Fast 8-GPU training
- `config_1.5b_single_gpu_blazing.json` - Fast single GPU

### Configs with `ignore_checkpoints: false` by Default

These configs will resume if checkpoints exist:
- `config_1.5b_fast.json` - Standard fast training
- `config_1.5b_single_gpu_ultra_fast.json` - Ultra fast single GPU
- All other standard configs

## 🗑️ Manually Deleting Checkpoints

Sometimes you want to physically delete old checkpoints:

### Delete all checkpoints (keep final model)
```bash
# Remove checkpoint directories only
rm -rf models/qwen2.5-1.5b-fast-8gpu/checkpoint-*
```

### Delete entire output directory
```bash
# Remove everything (model + checkpoints)
rm -rf models/qwen2.5-1.5b-fast-8gpu
```

### Clean up all model outputs
```bash
# Remove all trained models (use with caution!)
rm -rf models/qwen2.5-*
```

## 📊 Checkpoint Directory Structure

Typical checkpoint structure:
```
models/my-model/
├── checkpoint-100/          # Checkpoint at step 100
│   ├── adapter_model.safetensors
│   ├── optimizer.pt
│   ├── scheduler.pt
│   └── trainer_state.json
├── checkpoint-200/          # Checkpoint at step 200
├── checkpoint-300/          # Latest checkpoint
├── adapter_model.safetensors  # Final model
└── training_config.json
```

## 🎯 Common Scenarios

### Scenario 1: Continue interrupted training
**Config**: `ignore_checkpoints: false` (or omit it)
```bash
# Training was interrupted at step 450
# Just rerun - it will resume automatically
sbatch scripts/jobs/train_1.5b_fast.sh
```

### Scenario 2: Start fresh with same config
**Option A**: Use `ignore_checkpoints: true` in config
```json
{
  "ignore_checkpoints": true,
  ...
}
```

**Option B**: Delete checkpoints manually
```bash
rm -rf models/qwen2.5-1.5b-fast-8gpu/checkpoint-*
sbatch scripts/jobs/train_1.5b_fast.sh
```

### Scenario 3: Quick testing (don't resume)
Use blazing fast configs - they ignore checkpoints by default:
```bash
sbatch scripts/jobs/train_1.5b_blazing_fast.sh
```

### Scenario 4: Resume but with different hyperparameters
**Warning**: This may cause issues if LoRA dimensions changed!

**Safe approach**:
```bash
# Delete old checkpoints
rm -rf models/qwen2.5-1.5b-fast-8gpu/checkpoint-*

# Edit config with new hyperparameters
vim configs/config_1.5b_fast.json

# Start fresh training
sbatch scripts/jobs/train_1.5b_fast.sh
```

## ⚠️ Important Notes

1. **LoRA Dimension Mismatch**
   - If you change `lora_r` or `lora_alpha` between runs
   - The script will try to adapt, but it's safer to start fresh
   - Use `ignore_checkpoints: true` or delete checkpoints

2. **Max Length Changes**
   - Changing `max_length` is usually safe
   - The script will update to match checkpoint config

3. **Output Directory Conflicts**
   - If `output_dir` has checkpoints from a different model
   - Either change `output_dir` or set `ignore_checkpoints: true`

4. **Save Limits**
   - `save_total_limit: 3` keeps only the 3 most recent checkpoints
   - Older checkpoints are automatically deleted

## 🔧 Configuration Examples

### Example 1: Blazing Fast (Always Fresh)
```json
{
  "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
  "output_dir": "models/test-run",
  "ignore_checkpoints": true,
  "num_epochs": 1,
  "save_steps": 1000
}
```

### Example 2: Long Training (Resume-Friendly)
```json
{
  "model_name": "Qwen/Qwen2.5-7B-Instruct",
  "output_dir": "models/production-model",
  "ignore_checkpoints": false,
  "num_epochs": 5,
  "save_steps": 100,
  "save_total_limit": 5
}
```

### Example 3: Debugging (No Checkpoints Saved)
```json
{
  "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
  "output_dir": "models/debug-run",
  "ignore_checkpoints": true,
  "save_steps": 999999,
  "save_total_limit": 1
}
```

## 🚀 Quick Commands

```bash
# Check if checkpoints exist
ls -la models/qwen2.5-1.5b-fast-8gpu/checkpoint-*/

# Count checkpoints
ls models/qwen2.5-1.5b-fast-8gpu/ | grep checkpoint | wc -l

# Find latest checkpoint
ls -t models/qwen2.5-1.5b-fast-8gpu/checkpoint-* | head -1

# Clean up all checkpoints across all models
find models/ -type d -name "checkpoint-*" -exec rm -rf {} +

# Get disk usage of checkpoints
du -sh models/*/checkpoint-*
```

---

**Quick Reference**:
- Want to resume? → `ignore_checkpoints: false` (default)
- Want fresh training? → `ignore_checkpoints: true`
- Changed hyperparameters? → Delete checkpoints or set `ignore_checkpoints: true`

