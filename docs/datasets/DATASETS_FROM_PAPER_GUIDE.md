# Datasets from Clinical Mental Health AI Systems Paper

This guide documents the datasets cloned from the comprehensive review paper: "A Comprehensive Review of Datasets for Clinical Mental Health AI Systems" (arXiv:2508.09809v2).

## Cloned Datasets

### 1. ESConv (Empathetic Conversations)
- **Source**: https://huggingface.co/datasets/thu-coai/esconv
- **Description**: Empathetic conversations dataset with counseling strategies
- **Size**: 
  - Training: 8,388 samples
  - Validation: 933 samples
  - Total: 9,321 samples
- **Features**: Multi-turn conversations with counseling strategies (Question, Reflection of feelings, Restatement, Providing Suggestions, etc.)
- **Processed location**: `esconv_processed/`

**Process with:**
```bash
uv run prepare_esconv_dataset.py
```

### 2. Amod Mental Health Counseling Conversations
- **Source**: https://huggingface.co/datasets/Amod/mental_health_counseling_conversations
- **Description**: Mental health counseling conversation pairs
- **Size**:
  - Training: 2,862 samples
  - Validation: 318 samples
  - Total: 3,180 samples
- **Features**: Context-Response pairs for counseling conversations
- **Processed location**: `amod_processed/`

**Process with:**
```bash
uv run prepare_amod_dataset.py
```

## Complete Dataset Collection

You now have access to the following mental health counseling datasets:

1. **Counsel Chat** (existing)
   - Location: `counsel_chat_processed/`

2. **MentalChat16K**
   - Location: `mentalchat16k_processed/`
   - Size: 59,095 training + 6,567 validation

3. **Kaggle nguyenletruongthien**
   - Location: `kaggle_mental_health_nguyen_processed_combined/`
   - Size: 72,013 training + 8,002 validation

4. **PsyDial** (Chinese)
   - Location: `psydial_processed/`
   - Size: 59,095 training + 6,567 validation

5. **ESConv** (NEW)
   - Location: `esconv_processed/`
   - Size: 8,388 training + 933 validation

6. **Amod** (NEW)
   - Location: `amod_processed/`
   - Size: 2,862 training + 318 validation

## Combining All Datasets

You can combine all datasets for maximum training data:

```python
from datasets import load_from_disk, concatenate_datasets, DatasetDict

# Load all datasets
datasets = [
    load_from_disk("counsel_chat_processed"),
    load_from_disk("mentalchat16k_processed"),
    load_from_disk("kaggle_mental_health_nguyen_processed_combined"),
    load_from_disk("psydial_processed"),  # Chinese - consider separately
    load_from_disk("esconv_processed"),
    load_from_disk("amod_processed"),
]

# Combine training sets (excluding Chinese if needed)
combined_train = concatenate_datasets([ds["train"] for ds in datasets])
combined_val = concatenate_datasets([ds["validation"] for ds in datasets])

# Save combined dataset
combined = DatasetDict({
    "train": combined_train,
    "validation": combined_val
})
combined.save_to_disk("all_mental_health_combined")
```

## Total Dataset Statistics

**English-only datasets:**
- Total training samples: ~150,000+
- Total validation samples: ~18,000+

**Including Chinese (PsyDial):**
- Total training samples: ~210,000+
- Total validation samples: ~25,000+

## Usage

### Train on Individual Dataset
```bash
# Update config.json
{
  "dataset_path": "esconv_processed",  # or any other dataset
  ...
}

# Train
uv run train_qwen_counsel.py --config config.json
```

### Train on Combined Dataset
```bash
# First combine datasets (see Python code above)
# Then update config.json
{
  "dataset_path": "all_mental_health_combined",
  ...
}

# Train
uv run train_qwen_counsel.py --config config.json
```

## Notes

- **Language**: Most datasets are in English. PsyDial is in Chinese - use separately or translate if needed.
- **Format**: All datasets are processed to the same format compatible with your training pipeline.
- **Quality**: ESConv includes counseling strategies which can be valuable for training.
- **Size**: The combined dataset provides substantial training data for robust model training.

## References

- Paper: https://arxiv.org/html/2508.09809v2
- ESConv: https://huggingface.co/datasets/thu-coai/esconv
- Amod: https://huggingface.co/datasets/Amod/mental_health_counseling_conversations

