# Multilingual Training Guide (English + Chinese)

Yes, you can absolutely train with both Chinese and English datasets! This guide explains how to do it.

## Benefits of Multilingual Training

1. **Broader Coverage**: Model can handle both English and Chinese mental health conversations
2. **Better Generalization**: Multilingual training often improves model robustness
3. **Larger Dataset**: More training data = better performance
4. **Qwen2.5 Support**: Qwen2.5 models are designed for multilingual use

## Your Available Datasets

### English Datasets:
- `counsel_chat_processed/`
- `mentalchat16k_processed/`
- `kaggle_mental_health_nguyen_processed_combined/`
- `esconv_processed/`
- `amod_processed/`

### Chinese Dataset:
- `psydial_processed/` (~65K samples)

## How to Combine All Datasets (Including Chinese)

### Option 1: Combine Everything (Recommended)

```bash
# Combine ALL datasets including Chinese
uv run combine_all_datasets.py \
    --output_dir all_mental_health_multilingual
```

This will combine:
- All English datasets
- PsyDial (Chinese) dataset
- **Total**: ~207K+ training samples

### Option 2: Custom Combination

```bash
# Combine specific datasets
uv run combine_all_datasets.py \
    --output_dir custom_multilingual \
    --datasets counsel_chat_processed mentalchat16k_processed psydial_processed
```

### Option 3: Separate English and Chinese, Then Combine

```python
from datasets import load_from_disk, concatenate_datasets, DatasetDict

# Load English datasets
english_datasets = [
    load_from_disk("counsel_chat_processed"),
    load_from_disk("mentalchat16k_processed"),
    load_from_disk("kaggle_mental_health_nguyen_processed_combined"),
    load_from_disk("esconv_processed"),
    load_from_disk("amod_processed"),
]

# Load Chinese dataset
chinese_dataset = load_from_disk("psydial_processed")

# Combine all
all_datasets = english_datasets + [chinese_dataset]

combined_train = concatenate_datasets([ds["train"] for ds in all_datasets])
combined_val = concatenate_datasets([ds["validation"] for ds in all_datasets])

combined = DatasetDict({
    "train": combined_train,
    "validation": combined_val
})

combined.save_to_disk("all_mental_health_multilingual")
```

## Training Configuration

### Update config.json

```json
{
  "model_name": "Qwen/Qwen2.5-7B-Instruct-AWQ",
  "dataset_path": "all_mental_health_multilingual",
  "output_dir": "qwen2.5-counsel-chat-multilingual",
  "batch_size": 4,
  "num_epochs": 3,
  "learning_rate": 2e-4,
  ...
}
```

### Start Training

```bash
uv run train_qwen_counsel.py --config config.json
```

## Important Considerations

### 1. Model Support
✅ **Qwen2.5 models support Chinese natively** - No special configuration needed!

### 2. Tokenizer
The Qwen2.5 tokenizer handles both English and Chinese automatically. No changes needed.

### 3. Data Balance
Your combined dataset will have:
- **English**: ~142K samples
- **Chinese**: ~65K samples
- **Ratio**: ~68% English, ~32% Chinese

This is a reasonable balance. If you want more balance, you could:
- Upsample Chinese data
- Downsample English data
- Use weighted sampling during training

### 4. Evaluation
When evaluating, test on both:
- English test cases
- Chinese test cases

## Training Tips for Multilingual Models

### 1. Learning Rate
Multilingual training may benefit from slightly lower learning rates:
```json
{
  "learning_rate": 1.5e-4  // Slightly lower than English-only
}
```

### 2. Batch Size
With more data, you might need to adjust:
```json
{
  "batch_size": 4,
  "gradient_accumulation_steps": 4  // Effective batch size = 16
}
```

### 3. Training Time
Expect longer training time due to larger dataset:
- Monitor training progress
- Use appropriate `save_steps` and `eval_steps`
- Consider using wandb for tracking

## Quick Start Commands

### Complete Multilingual Training Workflow

```bash
# 1. Combine all datasets (including Chinese)
uv run combine_all_datasets.py \
    --output_dir all_mental_health_multilingual

# 2. Verify the combined dataset
python3 << 'EOF'
from datasets import load_from_disk
ds = load_from_disk("all_mental_health_multilingual")
print(f"Train: {len(ds['train']):,}")
print(f"Val: {len(ds['validation']):,}")
print(f"\nSample (checking for Chinese):")
print(ds['train'][0]['instruction'][:200])
EOF

# 3. Update config.json
# Set dataset_path to "all_mental_health_multilingual"

# 4. Train
uv run train_qwen_counsel.py --config config.json
```

## Testing Multilingual Model

After training, test with both languages:

```python
# English test
prompt = "I've been feeling anxious lately. Can you help me?"

# Chinese test  
prompt = "我最近一直感到焦虑。你能帮助我吗？"
```

## Expected Results

With multilingual training, your model should:
- ✅ Respond appropriately to English mental health queries
- ✅ Respond appropriately to Chinese mental health queries
- ✅ Understand context in both languages
- ✅ Apply counseling techniques regardless of language

## Troubleshooting

### Model Only Responds in One Language
- Check if training data is properly mixed
- Verify tokenizer supports both languages
- Try adjusting learning rate

### Poor Performance in One Language
- Check data quality for that language
- Consider balancing dataset sizes
- Use language-specific evaluation metrics

## Summary

**Yes, you can and should train with both Chinese and English!**

1. ✅ Qwen2.5 supports multilingual training natively
2. ✅ Your datasets are already processed and ready
3. ✅ Simply combine all datasets (including PsyDial)
4. ✅ Train as normal - no special configuration needed

The model will learn to handle mental health counseling in both languages! 🚀

