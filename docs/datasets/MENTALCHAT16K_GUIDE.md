# MentalChat16K Dataset Setup Guide

This guide explains how to download and process the MentalChat16K dataset for training your mental health counseling model.

## About MentalChat16K

MentalChat16K is a benchmark dataset designed for conversational mental health assistance. It consists of **16,113 question-answer pairs**:

- **Real Interview Data**: 6,338 pairs from anonymized transcripts of interactions between behavioral health coaches and caregivers
- **Synthetic Data**: 9,775 pairs generated using GPT-3.5 Turbo, covering 33 mental health topics

**Dataset Source**: 
- Hugging Face: https://huggingface.co/datasets/ShenLab/MentalChat16K
- GitHub: https://github.com/PennShenLab/MentalChat16K
- Paper: https://dl.acm.org/doi/10.1145/3711896.3737393

## Quick Start

### Step 1: Process the Dataset

The dataset is available on Hugging Face, so you can download and process it directly:

```bash
uv run prepare_mentalchat16k_dataset.py
```

This will:
- Download the dataset from Hugging Face (first time only)
- Process and clean the data
- Create train/validation splits (90/10)
- Save to `mentalchat16k_processed/` directory

### Step 2: Process with Custom Options

```bash
uv run prepare_mentalchat16k_dataset.py \
    --output_dir mentalchat16k_processed \
    --max_samples 1000  # Optional: limit for testing
```

### Step 3: Train Your Model

Update your `config.json` to use the new dataset:

```json
{
  "dataset_path": "mentalchat16k_processed",
  ...
}
```

Then train:

```bash
uv run train_qwen_counsel.py --config config.json
```

## Dataset Structure

The MentalChat16K dataset has the following structure:
- `instruction`: System prompt for the mental health assistant
- `input`: User's question/concern
- `output`: Assistant's response

The processing script converts this to the format expected by your training pipeline:
- `instruction`: Formatted prompt with the question
- `input`: Empty (for compatibility)
- `output`: Assistant's response
- `topic`: Topic/category (if available)
- `upvotes`: 0 (not available in this dataset)
- `question_id`: Unique identifier

## Combining with Other Datasets

You can combine MentalChat16K with your other datasets (Counsel Chat, Kaggle dataset) for more diverse training data:

```python
from datasets import load_from_disk, concatenate_datasets, DatasetDict

# Load all datasets
counsel_chat = load_from_disk("counsel_chat_processed")
kaggle_data = load_from_disk("kaggle_mental_health_processed")
mentalchat16k = load_from_disk("mentalchat16k_processed")

# Combine training sets
combined_train = concatenate_datasets([
    counsel_chat["train"],
    kaggle_data["train"],
    mentalchat16k["train"]
])

# Combine validation sets
combined_val = concatenate_datasets([
    counsel_chat["validation"],
    kaggle_data["validation"],
    mentalchat16k["validation"]
])

# Save combined dataset
combined = DatasetDict({
    "train": combined_train,
    "validation": combined_val
})
combined.save_to_disk("combined_mental_health_processed")
```

## Script Options

```bash
uv run prepare_mentalchat16k_dataset.py --help
```

Options:
- `--dataset_name`: Hugging Face dataset name (default: `ShenLab/MentalChat16K`)
- `--output_dir`: Output directory (default: `mentalchat16k_processed`)
- `--max_samples`: Limit number of samples for testing
- `--cache_dir`: Cache directory for Hugging Face datasets
- `--split`: Dataset split to load (default: `train`)

## Troubleshooting

### Download Issues
- Make sure you have internet connection
- The first download may take a few minutes
- Dataset is cached locally after first download

### Processing Issues
- The script automatically detects column names
- If issues occur, check the dataset structure on Hugging Face
- Verify the dataset name is correct: `ShenLab/MentalChat16K`

## Citation

If you use MentalChat16K in your research, please cite:

```bibtex
@article{MentalChat16K,
  author    = {Jia Xu, Tianyi Wei, Bojian Hou, Patryk Orzechowski, Shu Yang, Ruochen Jin, Rachael Paulbeck, Joost Wagenaar, George Demiris, Li Shen},
  title     = {MentalChat16K: A Benchmark Dataset for Conversational Mental Health Assistance},
  year      = {2024},
  url       = {https://huggingface.co/datasets/ShenLab/MentalChat16K},
}
```

