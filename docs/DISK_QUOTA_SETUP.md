# Disk Quota Setup Guide

This guide explains how the scripts are configured to avoid disk quota errors on HPC clusters.

## Problem

HuggingFace libraries by default cache models and datasets in `~/.cache/huggingface`, which may be on a filesystem with limited quota. This can cause errors like:

```
RuntimeError: Data processing error: CAS service error : IO Error: Disk quota exceeded (os error 122)
```

## Solution

All scripts are configured to use `/research/d7/fyp25/yyyu2` for all cache and output files.

## Cache Directories

The following environment variables are set to use your quota path:

- `HF_HOME`: `/research/d7/fyp25/yyyu2/.cache/huggingface`
- `TRANSFORMERS_CACHE`: `/research/d7/fyp25/yyyu2/.cache/huggingface/transformers`
- `HF_DATASETS_CACHE`: `/research/d7/fyp25/yyyu2/.cache/huggingface/datasets`
- `HF_HUB_CACHE`: `/research/d7/fyp25/yyyu2/.cache/huggingface/hub`
- `XET_CACHE`: `/research/d7/fyp25/yyyu2/.cache/huggingface/xet`

## How It Works

### 1. Batch Jobs (`sbatch train_job.sh`)

The batch script automatically:
- Changes to `/research/d7/fyp25/yyyu2/FYP-LLM`
- Sets all HuggingFace cache environment variables
- Creates cache directories if they don't exist
- Runs training with proper cache paths

### 2. Interactive Training (`run_interactive_training.sh`)

The interactive script:
- Sets the same cache environment variables
- Changes to the correct working directory
- Ensures all downloads go to your quota path

### 3. Training Script (`train_qwen_counsel.py`)

The Python script:
- Checks for environment variables (set by shell scripts)
- Falls back to `/research/d7/fyp25/yyyu2` if not set
- Creates cache directories automatically
- Passes `cache_dir` parameter to all HuggingFace functions

## Verification

After running a job, you can verify cache locations:

```bash
# Check environment variables
echo $HF_HOME
echo $TRANSFORMERS_CACHE

# Check cache directory size
du -sh /research/d7/fyp25/yyyu2/.cache/huggingface

# List cached models
ls -lh /research/d7/fyp25/yyyu2/.cache/huggingface/transformers/
```

## Manual Setup (if needed)

If you need to set these manually in your shell:

```bash
export BASE_DIR="/research/d7/fyp25/yyyu2"
export HF_HOME="$BASE_DIR/.cache/huggingface"
export TRANSFORMERS_CACHE="$BASE_DIR/.cache/huggingface/transformers"
export HF_DATASETS_CACHE="$BASE_DIR/.cache/huggingface/datasets"
export HF_HUB_CACHE="$BASE_DIR/.cache/huggingface/hub"
export XET_CACHE="$BASE_DIR/.cache/huggingface/xet"

# Create directories
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$HF_HUB_CACHE" "$XET_CACHE"
```

## Troubleshooting

### Still getting quota errors?

1. **Check current cache location:**
   ```bash
   python -c "from transformers import file_utils; print(file_utils.default_cache_path)"
   ```

2. **Clear old cache (if needed):**
   ```bash
   # Check size first
   du -sh ~/.cache/huggingface
   
   # Remove if too large (be careful!)
   rm -rf ~/.cache/huggingface
   ```

3. **Verify environment variables are set:**
   ```bash
   env | grep HF_
   env | grep CACHE
   ```

4. **Check disk quota:**
   ```bash
   quota -s
   # or
   df -h /research/d7/fyp25/yyyu2
   ```

### Model downloads still going to wrong location?

Make sure you're using the updated scripts. The training script now explicitly passes `cache_dir` to:
- `AutoTokenizer.from_pretrained()`
- `AutoModelForCausalLM.from_pretrained()`

## Output Directories

All training outputs (models, logs) are also configured to use your quota path:

- **Models**: `models/qwen2.5-counsel-chat-finetuned` (relative to project root)
- **Logs**: `logs/train_JOBID.out` (relative to project root)
- **Datasets**: `datasets/` (relative to project root)

All paths are relative to `/research/d7/fyp25/yyyu2/FYP-LLM`.

