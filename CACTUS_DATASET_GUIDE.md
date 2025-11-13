# Cactus Dataset Setup Guide

This guide explains how to download and process the Cactus dataset for training your mental health counseling model.

## About Cactus

Cactus is a psychological counseling conversations dataset using Cognitive Behavioral Theory (CBT). It contains:
- **31,577 dialogues** with full counseling conversations
- CBT techniques and plans
- Client intake forms
- Cognitive patterns identification
- Multi-turn conversations

**Dataset Source**: 
- GitHub: https://github.com/coding-groot/cactus
- Hugging Face: https://huggingface.co/datasets/cactus-camel/cactus
- Paper: https://arxiv.org/abs/2407.03103

## Quick Start

### Step 1: Process the Dataset

The dataset is available on Hugging Face:

```bash
uv run prepare_cactus_dataset.py
```

This will:
- Download the dataset from Hugging Face (first time only)
- Process multi-turn conversations into training examples
- Create train/validation splits (90/10)
- Save to `cactus_processed/` directory

**Note**: This dataset is very large (~460K training examples). Make sure you have sufficient disk space (several GB).

### Step 2: Process with Limited Samples (for testing)

If you want to test with a smaller sample first:

```bash
uv run prepare_cactus_dataset.py --max_samples 1000
```

### Step 3: Train Your Model

Update your `config.json` to use the processed dataset:

```json
{
  "dataset_path": "cactus_processed",
  ...
}
```

Then train:

```bash
uv run train_qwen_counsel.py --config config.json
```

## Dataset Statistics

After processing:
- **Total dialogues**: 31,577
- **Training samples**: ~413,736 (estimated)
- **Validation samples**: ~45,971 (estimated)
- **Average turns per dialogue**: ~14-15 turns

## Dataset Structure

The Cactus dataset contains:
- **thought**: Client's problematic thought
- **cbt_technique**: CBT technique used (e.g., Decatastrophizing, Alternative Perspective)
- **intake_form**: Client intake form with demographics and presenting problem
- **attitude**: Client attitude (positive, neutral, negative)
- **dialogue**: Full multi-turn counseling conversation
- **patterns**: List of cognitive patterns (e.g., catastrophizing, personalization)
- **cbt_plan**: Detailed CBT counseling plan

The processing script:
1. Extracts user-assistant pairs from each dialogue
2. Builds context from previous conversation turns
3. Includes CBT plan information in early turns
4. Creates instruction prompts with context
5. Formats for training compatibility

## Features

- **CBT-based**: Uses Cognitive Behavioral Therapy techniques
- **Multi-turn**: Long conversations with context
- **Rich metadata**: Includes intake forms, CBT plans, and cognitive patterns
- **Diverse**: Covers various mental health issues and client attitudes

## Combining with Other Datasets

You can combine Cactus with your other datasets:

```python
from datasets import load_from_disk, concatenate_datasets, DatasetDict

# Load all datasets
counsel_chat = load_from_disk("counsel_chat_processed")
mentalchat16k = load_from_disk("mentalchat16k_processed")
kaggle_nguyen = load_from_disk("kaggle_mental_health_nguyen_processed_combined")
esconv = load_from_disk("esconv_processed")
amod = load_from_disk("amod_processed")
cactus = load_from_disk("cactus_processed")

# Combine training sets
combined_train = concatenate_datasets([
    counsel_chat["train"],
    mentalchat16k["train"],
    kaggle_nguyen["train"],
    esconv["train"],
    amod["train"],
    cactus["train"]
])

# Combine validation sets
combined_val = concatenate_datasets([
    counsel_chat["validation"],
    mentalchat16k["validation"],
    kaggle_nguyen["validation"],
    esconv["validation"],
    amod["validation"],
    cactus["validation"]
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
uv run prepare_cactus_dataset.py --help
```

Options:
- `--dataset_name`: Hugging Face dataset name (default: `cactus-camel/cactus`)
- `--output_dir`: Output directory (default: `cactus_processed`)
- `--max_samples`: Limit number of dialogues for testing
- `--cache_dir`: Cache directory for Hugging Face datasets
- `--split`: Dataset split to load (default: `train`)

## Disk Space Requirements

**Important**: The Cactus dataset is very large. After processing, it will require:
- **Raw dataset**: ~500MB-1GB
- **Processed dataset**: ~2-5GB (depending on compression)

Make sure you have sufficient disk space before processing the full dataset.

## Troubleshooting

### Disk Space Issues
- Process with `--max_samples` to limit the dataset size
- Free up disk space before processing
- Consider processing in batches

### Processing Issues
- The script processes dialogues and extracts turns
- Some dialogues may be skipped if they don't meet length requirements
- Check the sample data file to verify the format

## Citation

If you use Cactus in your research, please cite:

```bibtex
@misc{lee2024cactus,
    title={Cactus: Towards Psychological Counseling Conversations using Cognitive Behavioral Theory}, 
    author={Suyeon Lee and Sunghwan Kim and Minju Kim and Dongjin Kang and Dongil Yang and Harim Kim and Minseok Kang and Dayi Jung and Min Hee Kim and Seungbeen Lee and Kyoung-Mee Chung and Youngjae Yu and Dongha Lee and Jinyoung Yeo},
    year={2024},
    eprint={2407.03103},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2407.03103}
}
```

