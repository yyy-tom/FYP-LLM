# PsyDial Dataset Setup Guide

This guide explains how to download and process the PsyDial dataset for training your mental health counseling model.

## About PsyDial

PsyDial is a large-scale, long-term conversational dataset for mental health support in Chinese. It contains:
- **2,382 dialogues** with an average of **37.8 turns per dialogue**
- Semi-real data created by reconstructing real-world, long-term counseling dialogues
- Privacy-preserving data using the RMRR (Retrieve, Mask, Reconstruct, Refine) method

**Dataset Source**: 
- GitHub: https://github.com/qiuhuachuan/PsyDial
- Hugging Face: https://huggingface.co/datasets/qiuhuachuan/PsyDial-D4
- Paper: https://aclanthology.org/2025.acl-long.1049/

## Quick Start

### Step 1: Process the Dataset

The dataset is available on Hugging Face, so you can download and process it directly:

```bash
uv run prepare_psydial_dataset.py
```

This will:
- Download the PsyDial-D4 dataset from Hugging Face (first time only)
- Process multi-turn conversations into training examples
- Create train/validation splits (90/10)
- Save to `psydial_processed/` directory

### Step 2: Process Different PsyDial Variants

The PsyDial dataset has multiple variants available:

```bash
# Process PsyDial-D4 (main dataset)
uv run prepare_psydial_dataset.py --dataset_name qiuhuachuan/PsyDial-D4

# Process other variants
uv run prepare_psydial_dataset.py --dataset_name qiuhuachuan/PsyDial-D0_m
uv run prepare_psydial_dataset.py --dataset_name qiuhuachuan/PsyDial-D1
uv run prepare_psydial_dataset.py --dataset_name qiuhuachuan/PsyDial-D2
uv run prepare_psydial_dataset.py --dataset_name qiuhuachuan/PsyDial-D3
uv run prepare_psydial_dataset.py --dataset_name qiuhuachuan/PsyDial-D101
```

### Step 3: Train Your Model

Update your `config.json` to use the processed dataset:

```json
{
  "dataset_path": "psydial_processed",
  ...
}
```

Then train:

```bash
uv run train_qwen_counsel.py --config config.json
```

## Dataset Statistics

After processing PsyDial-D4:
- **Total dialogues**: 2,382
- **Training samples**: 59,095
- **Validation samples**: 6,567
- **Total samples**: 65,662
- **Average turns per dialogue**: ~27-28 training examples per dialogue

## Dataset Structure

The PsyDial dataset contains multi-turn conversations with:
- **System messages**: Instructions for the counselor role
- **User messages**: Client/counselor seeker messages
- **Assistant messages**: Counselor responses

The processing script:
1. Extracts user-assistant pairs from each dialogue
2. Builds context from previous conversation turns
3. Creates instruction prompts with context
4. Formats for training compatibility

## Combining with Other Datasets

You can combine PsyDial with your other datasets for more diverse training data:

```python
from datasets import load_from_disk, concatenate_datasets, DatasetDict

# Load all datasets
counsel_chat = load_from_disk("counsel_chat_processed")
mentalchat16k = load_from_disk("mentalchat16k_processed")
kaggle_nguyen = load_from_disk("kaggle_mental_health_nguyen_processed_combined")
psydial = load_from_disk("psydial_processed")

# Combine training sets
combined_train = concatenate_datasets([
    counsel_chat["train"],
    mentalchat16k["train"],
    kaggle_nguyen["train"],
    psydial["train"]
])

# Combine validation sets
combined_val = concatenate_datasets([
    counsel_chat["validation"],
    mentalchat16k["validation"],
    kaggle_nguyen["validation"],
    psydial["validation"]
])

# Save combined dataset
combined = DatasetDict({
    "train": combined_train,
    "validation": combined_val
})
combined.save_to_disk("all_mental_health_combined")
```

## Script Options

```bash
uv run prepare_psydial_dataset.py --help
```

Options:
- `--dataset_name`: Hugging Face dataset name (default: `qiuhuachuan/PsyDial-D4`)
- `--output_dir`: Output directory (default: `psydial_processed`)
- `--max_samples`: Limit number of dialogues for testing
- `--cache_dir`: Cache directory for Hugging Face datasets
- `--split`: Dataset split to load (default: `train`)

## Notes

- **Language**: The dataset is in Chinese, so make sure your model supports Chinese or you plan to translate it
- **Multi-turn**: Each dialogue contains multiple turns, which are converted into individual training examples with context
- **Context**: The script includes conversation history as context for later turns in the dialogue
- **Filtering**: Very short or very long responses are filtered out (20-2000 characters for answers)

## Citation

If you use PsyDial in your research, please cite:

```bibtex
@inproceedings{qiu-lan-2025-psydial,
    title = "{P}sy{D}ial: A Large-scale Long-term Conversational Dataset for Mental Health Support",
    author = "Qiu, Huachuan  and Lan, Zhenzhong",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.acl-long.1049/",
    doi = "10.18653/v1/2025.acl-long.1049",
}
```

