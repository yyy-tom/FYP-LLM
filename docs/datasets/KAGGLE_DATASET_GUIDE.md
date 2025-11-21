# Kaggle Mental Health Dataset Setup Guide

This guide explains how to download and process the Kaggle Mental Health Counseling Conversations dataset for training.

## Prerequisites

1. **Kaggle Account**: You need a Kaggle account to download datasets
2. **Kaggle API Credentials**: Set up your Kaggle API credentials

### Setting up Kaggle API

1. Go to https://www.kaggle.com/account
2. Scroll down to the "API" section
3. Click "Create New Token" to download `kaggle.json`
4. Place the file in `~/.kaggle/kaggle.json`
5. Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

Alternatively, you can set environment variables:
```bash
export KAGGLE_USERNAME='your_username'
export KAGGLE_KEY='your_api_key'
```

## Step 1: Download the Dataset

Use `uv` to run the download script:

```bash
uv run download_kaggle_dataset.py
```

Or with custom options:

```bash
uv run download_kaggle_dataset.py \
    --dataset melissamonfared/mental-health-counseling-conversations-k \
    --output_dir kaggle_mental_health
```

The script will:
- Check if Kaggle API is installed (installs if missing with `--install-kaggle`)
- Verify your credentials
- Download the dataset to the specified directory
- Unzip the files automatically

## Step 2: Process the Dataset

After downloading, process the dataset into the training format:

```bash
uv run prepare_kaggle_dataset.py \
    --input_dir kaggle_mental_health \
    --output_dir kaggle_mental_health_processed
```

The script will:
- Auto-detect CSV files in the input directory
- Auto-detect column names (question, answer, topic)
- Clean and filter the data
- Create train/validation splits (90/10)
- Save in the format compatible with your training script

### Custom Column Mapping

If the auto-detection doesn't work, specify columns manually:

```bash
uv run prepare_kaggle_dataset.py \
    --input_dir kaggle_mental_health \
    --output_dir kaggle_mental_health_processed \
    --question_col "question" \
    --answer_col "answer" \
    --topic_col "topic"
```

## Step 3: Train Your Model

Update your `config.json` to use the new dataset:

```json
{
  "dataset_path": "kaggle_mental_health_processed",
  ...
}
```

Then train:

```bash
uv run train_qwen_counsel.py --config config.json
```

## Combining Datasets (Optional)

If you want to combine the Kaggle dataset with your existing Counsel Chat dataset, you can:

1. Process both datasets separately
2. Use the `datasets` library to concatenate them:

```python
from datasets import load_from_disk, concatenate_datasets

# Load both datasets
counsel_chat = load_from_disk("counsel_chat_processed")
kaggle_data = load_from_disk("kaggle_mental_health_processed")

# Combine
combined_train = concatenate_datasets([
    counsel_chat["train"],
    kaggle_data["train"]
])

combined_val = concatenate_datasets([
    counsel_chat["validation"],
    kaggle_data["validation"]
])

# Save combined dataset
from datasets import DatasetDict
combined = DatasetDict({
    "train": combined_train,
    "validation": combined_val
})
combined.save_to_disk("combined_mental_health_processed")
```

## Troubleshooting

### Kaggle API Issues
- Make sure you've accepted the dataset's terms of use on Kaggle
- Verify your credentials are correct
- Check that the dataset name is correct

### Column Detection Issues
- Inspect the CSV file first: `head -5 kaggle_mental_health/*.csv`
- Manually specify column names using `--question_col`, `--answer_col`, `--topic_col`

### Encoding Issues
- The script tries multiple encodings automatically
- If issues persist, check the CSV file encoding and convert if needed

