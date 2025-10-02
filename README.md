# FYP LLM - Qwen2.5 Counseling Chat Training

This project fine-tunes the Qwen2.5-0.5B model on the Counsel Chat dataset to create a mental health counseling assistant.

## Quick Start

### 1. Install Dependencies

**For macOS (Apple Silicon):**

```bash
uv sync
```

**For Linux/Windows with CUDA:**

```bash
uv sync --extra cuda
```

**Note:** The 0.5B model is small enough to run efficiently on macOS without quantization.

### 2. Test Setup

```bash
uv run python test_setup.py
```

### 3. Prepare Dataset (Small Test)

```bash
uv run python prepare_counsel_dataset.py --max_samples 100
```

### 4. Train Model

```bash
uv run python train_qwen_counsel.py
```

### 5. Test Inference

```bash
uv run python inference.py --interactive
```

## Files Overview

- `config.json` - Training configuration (using Qwen2.5-0.5B for testing)
- `prepare_counsel_dataset.py` - Dataset preparation script
- `train_qwen_counsel.py` - Main training script with LoRA
- `inference.py` - Inference script for testing the trained model
- `test_setup.py` - Setup verification script

## Configuration

The training uses:

- **Model**: Qwen2.5-0.5B-Instruct (smallest model for testing)
- **Method**: LoRA fine-tuning with 4-bit quantization
- **Dataset**: Counsel Chat (mental health Q&A)
- **Batch Size**: 8 (optimized for 0.5B model)
- **Max Length**: 1024 tokens
- **Epochs**: 2 (for quick testing)

## Model Sizes Available

From the [Qwen2.5 collection](https://huggingface.co/collections/Qwen/qwen25-66e81a666513e518adb90d9e):

- 0.5B (current) - Fastest, least memory
- 1.5B, 3B, 7B, 14B, 32B, 72B - Larger models for better quality

To use a larger model, update `model_name` in `config.json`.
