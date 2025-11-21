# Kaggle Mental Health Dataset (nguyenletruongthien) Setup Guide

This guide explains how to download and process the Kaggle Mental Health dataset from `nguyenletruongthien/mental-health` for training your mental health counseling model.

## About the Dataset

This dataset contains multiple files with mental health conversation data:
- **conversations_training.csv**: 40,237 conversation pairs (input/output format)
- **mental_health_conversations.csv**: 40,000 conversation pairs (question/answer format)
- **Total**: ~80,000 conversation pairs after processing

**Dataset Source**: https://www.kaggle.com/datasets/nguyenletruongthien/mental-health

## Quick Start

### Step 1: Download the Dataset

The dataset has already been downloaded using the Kaggle API:

```bash
uv run download_kaggle_dataset.py \
    --dataset nguyenletruongthien/mental-health \
    --output_dir kaggle_mental_health_nguyen
```

### Step 2: Process the Dataset

The dataset has been processed and combined. The processed dataset is available at:
- **Combined dataset**: `kaggle_mental_health_nguyen_processed_combined/`
  - Training samples: 72,013
  - Validation samples: 8,002

If you need to reprocess:

```bash
# Process conversations_training.csv
uv run prepare_kaggle_dataset.py \
    --input_dir kaggle_mental_health_nguyen/conversations_training.csv \
    --output_dir kaggle_mental_health_nguyen_processed \
    --question_col input \
    --answer_col output

# Process mental_health_conversations.csv
uv run prepare_kaggle_dataset.py \
    --input_dir kaggle_mental_health_nguyen/mental_health_conversations.csv \
    --output_dir kaggle_mental_health_nguyen_processed2 \
    --question_col question \
    --answer_col answer

# Combine both datasets
uv run combine_datasets.py \
    --input_dirs kaggle_mental_health_nguyen_processed kaggle_mental_health_nguyen_processed2 \
    --output_dir kaggle_mental_health_nguyen_processed_combined
```

### Step 3: Train Your Model

Update your `config.json` to use the combined dataset:

```json
{
  "dataset_path": "kaggle_mental_health_nguyen_processed_combined",
  ...
}
```

Then train:

```bash
uv run train_qwen_counsel.py --config config.json
```

## Dataset Files

The downloaded dataset contains multiple files:

1. **conversations_training.csv** (18 MB)
   - Columns: `input`, `output`
   - Format: Direct conversation pairs
   - Rows: 40,237

2. **mental_health_conversations.csv** (20 MB)
   - Columns: `question`, `answer`, `source`, `source_dataset`, `statement`, `status`
   - Format: Question-answer pairs with metadata
   - Rows: 40,000

3. **Other files** (not used for training):
   - `dialogues_training.csv`: Different format (emotion/act/topic)
   - `mental_health_comprehensive.csv`: Statistics/indicators data
   - `sentiment_analysis.csv`: Sentiment analysis data
   - `reddit_mental_health_combined.csv`: Reddit data
   - `combined_intents.json`: Intent classification data
   - `conversations_training.json`: JSON version of conversations

## Dataset Statistics

After processing and combining:
- **Total training samples**: 72,013
- **Total validation samples**: 8,002
- **Total samples**: 80,015

## Combining with Other Datasets

You can combine this dataset with your other datasets for more diverse training data:

```python
from datasets import load_from_disk, concatenate_datasets, DatasetDict

# Load all datasets
counsel_chat = load_from_disk("counsel_chat_processed")
mentalchat16k = load_from_disk("mentalchat16k_processed")
kaggle_nguyen = load_from_disk("kaggle_mental_health_nguyen_processed_combined")

# Combine training sets
combined_train = concatenate_datasets([
    counsel_chat["train"],
    mentalchat16k["train"],
    kaggle_nguyen["train"]
])

# Combine validation sets
combined_val = concatenate_datasets([
    counsel_chat["validation"],
    mentalchat16k["validation"],
    kaggle_nguyen["validation"]
])

# Save combined dataset
combined = DatasetDict({
    "train": combined_train,
    "validation": combined_val
})
combined.save_to_disk("all_mental_health_combined")
```

## Troubleshooting

### Processing Issues
- The script automatically detects column names, but you can specify them manually
- Both CSV files use UTF-8 encoding
- Some rows may be filtered out if they don't meet length requirements (50-2000 chars for answers, 20+ chars for questions)

### File Structure
- The dataset contains multiple files with different formats
- Only `conversations_training.csv` and `mental_health_conversations.csv` are processed for conversational training
- Other files may be useful for different tasks (sentiment analysis, intent classification, etc.)

## Notes

- The dataset has been cleaned and filtered to remove very short or very long responses
- Train/validation split is 90/10
- The combined dataset maintains the same format as your other processed datasets for compatibility

