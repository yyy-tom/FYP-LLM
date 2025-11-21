# Complete Training Guide for Mental Health Counseling Model

This guide explains how to train your model with all the collected mental health datasets.

## Available Datasets

You have the following processed datasets:

1. **counsel_chat_processed/** - Counsel Chat dataset
2. **mentalchat16k_processed/** - MentalChat16K dataset (16K samples)
3. **kaggle_mental_health_nguyen_processed_combined/** - Kaggle dataset (80K samples)
4. **esconv_processed/** - ESConv empathetic conversations (9K samples)
5. **amod_processed/** - Amod mental health counseling (3K samples)
6. **psydial_processed/** - PsyDial dataset (65K samples, **Chinese**)
7. **kaggle_mental_health_processed/** - Another Kaggle dataset (if different from above)

## Training Options

### Option 1: Train on Individual Dataset

Train on a single dataset to test or focus on specific data:

```bash
# Update config.json
{
  "dataset_path": "counsel_chat_processed",
  ...
}

# Train
uv run train_qwen_counsel.py --config config.json
```

### Option 2: Train on Combined Dataset (Recommended)

Combine all datasets for maximum training data:

#### Step 1: Combine All Datasets

**English-only datasets (recommended for most cases):**

```bash
uv run combine_all_datasets.py \
    --output_dir all_mental_health_combined \
    --exclude_chinese
```

**All datasets (including Chinese):**

```bash
uv run combine_all_datasets.py \
    --output_dir all_mental_health_combined_with_chinese
```

**Custom combination:**

```bash
uv run combine_all_datasets.py \
    --output_dir custom_combined \
    --datasets counsel_chat_processed mentalchat16k_processed esconv_processed
```

#### Step 2: Update Config

Edit `config.json`:

```json
{
  "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
  "dataset_path": "all_mental_health_combined",
  "output_dir": "qwen2.5-counsel-chat-finetuned",
  "batch_size": 2,
  "num_epochs": 2,
  "learning_rate": 5e-4,
  "use_wandb": false,
  "use_4bit": false,
  "lora_r": 16,
  "lora_alpha": 32,
  "lora_dropout": 0.1,
  "max_length": 1024,
  "gradient_accumulation_steps": 2,
  "eval_batch_size": 2,
  "weight_decay": 0.01,
  "warmup_ratio": 0.1,
  "logging_steps": 10,
  "eval_steps": 100,
  "save_steps": 500,
  "save_total_limit": 3,
  "fp16": false,
  "bf16": false,
  "early_stopping": true,
  "seed": 42
}
```

#### Step 3: Train

```bash
uv run train_qwen_counsel.py --config config.json
```

### Option 3: Train with Command Line Arguments

Override config with command line arguments:

```bash
uv run train_qwen_counsel.py \
    --config config.json \
    --dataset_path all_mental_health_combined \
    --output_dir qwen2.5-counsel-chat-finetuned \
    --batch_size 4 \
    --num_epochs 3 \
    --learning_rate 2e-4
```

## Dataset Statistics

Approximate sizes (English-only, excluding PsyDial):

- **counsel_chat_processed**: ~Varies
- **mentalchat16k_processed**: ~59K training samples
- **kaggle_mental_health_nguyen_processed_combined**: ~72K training samples
- **esconv_processed**: ~8.4K training samples
- **amod_processed**: ~2.9K training samples

**Total (English-only)**: ~142K+ training samples

**With Chinese (PsyDial)**: ~207K+ training samples

## Training Configuration Tips

### For Small Models (0.5B-1.5B)

```json
{
  "batch_size": 4,
  "gradient_accumulation_steps": 4,
  "max_length": 1024,
  "num_epochs": 3,
  "learning_rate": 5e-4
}
```

### For Medium Models (3B-7B)

```json
{
  "batch_size": 2,
  "gradient_accumulation_steps": 8,
  "max_length": 2048,
  "num_epochs": 2,
  "learning_rate": 2e-4,
  "use_4bit": true
}
```

### For Large Models (14B+)

```json
{
  "batch_size": 1,
  "gradient_accumulation_steps": 16,
  "max_length": 2048,
  "num_epochs": 1,
  "learning_rate": 1e-4,
  "use_4bit": true
}
```

## Step-by-Step Training Workflow

### 1. Check Available Datasets

```bash
ls -d *_processed*
```

### 2. Combine Datasets (Optional but Recommended)

```bash
# Combine English datasets
uv run combine_all_datasets.py --exclude_chinese
```

### 3. Verify Combined Dataset

```python
from datasets import load_from_disk
ds = load_from_disk("all_mental_health_combined")
print(f"Train: {len(ds['train']):,}")
print(f"Val: {len(ds['validation']):,}")
print(f"Sample: {ds['train'][0]}")
```

### 4. Update Configuration

Edit `config.json` with your preferred settings.

### 5. Start Training

```bash
uv run train_qwen_counsel.py --config config.json
```

### 6. Monitor Training

- Check logs in the output directory
- If using wandb, monitor online
- Check saved checkpoints in `output_dir/checkpoint-*`

### 7. Evaluate Model

After training, test the model:

```bash
uv run inference.py --model_path qwen2.5-counsel-chat-finetuned --interactive
```

## Quick Start Commands

### Combine and Train (One Command)

```bash
# Combine datasets
uv run combine_all_datasets.py --exclude_chinese

# Train
uv run train_qwen_counsel.py \
    --dataset_path all_mental_health_combined \
    --output_dir qwen2.5-counsel-chat-finetuned \
    --batch_size 4 \
    --num_epochs 3
```

### Train on Single Dataset

```bash
uv run train_qwen_counsel.py \
    --dataset_path counsel_chat_processed \
    --output_dir qwen2.5-counsel-chat-only \
    --batch_size 4 \
    --num_epochs 3
```

## Troubleshooting

### Out of Memory

- Reduce `batch_size`
- Increase `gradient_accumulation_steps`
- Reduce `max_length`
- Enable `use_4bit: true` (if supported)

### Training Too Slow

- Increase `batch_size` (if memory allows)
- Reduce `max_length`
- Use smaller model
- Reduce dataset size with `--max_samples` during processing

### Poor Results

- Train for more epochs
- Adjust learning rate
- Try different model sizes
- Check data quality

## Next Steps

1. **Combine datasets**: Use `combine_all_datasets.py`
2. **Update config**: Set `dataset_path` to combined dataset
3. **Start training**: Run `train_qwen_counsel.py`
4. **Monitor progress**: Check logs and checkpoints
5. **Evaluate**: Test with `inference.py`

## Example Complete Workflow

```bash
# 1. Combine all English datasets
uv run combine_all_datasets.py --exclude_chinese

# 2. Check the combined dataset
python3 -c "from datasets import load_from_disk; ds = load_from_disk('all_mental_health_combined'); print(f'Train: {len(ds[\"train\"]):,}, Val: {len(ds[\"validation\"]):,}')"

# 3. Train the model
uv run train_qwen_counsel.py \
    --config config.json \
    --dataset_path all_mental_health_combined \
    --output_dir qwen2.5-counsel-chat-finetuned \
    --batch_size 4 \
    --num_epochs 3 \
    --learning_rate 5e-4

# 4. Test the trained model
uv run inference.py \
    --model_path qwen2.5-counsel-chat-finetuned \
    --interactive
```

Good luck with your training! 🚀

