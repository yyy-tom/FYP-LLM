# Scripts Directory

This directory contains all scripts for the FYP-LLM project, organized by functionality.

## Directory Structure

### 📊 `data/`
Dataset preparation and combination scripts:
- `prepare_*_dataset.py` - Individual dataset preparation scripts for various sources
- `combine_all_datasets.py` - Combines multiple processed datasets
- `combine_datasets.py` - Dataset combination utilities
- `download_kaggle_dataset.py` - Downloads datasets from Kaggle

### 🎓 `training/`
Model training scripts:
- `train_qwen_counsel.py` - Main training script for single GPU
- `train_qwen_counsel_multi_gpu.py` - Multi-GPU training script

### 🖥️ `jobs/`
SLURM job submission and execution scripts:
- `train_job*.sh` - Various SLURM job configurations for training
- `run_interactive_training.sh` - Interactive training session
- `run_training_screen.sh` - Training in screen session
- `run_training_tmux.sh` - Training in tmux session
- `monitor_job.sh` - Job monitoring utilities

### 📈 `visualization/`
Visualization and diagram generation:
- `generate_architecture_diagrams.py` - Generates architecture diagrams
- `generate_model_selection_visuals.py` - Creates model selection visualizations

### 🔮 `inference/`
Model inference and comparison:
- `inference.py` - Run inference on trained models
- `compare_models.py` - Compare different model versions

### ⚙️ `setup/`
Setup and testing utilities:
- `test_setup.py` - Verify environment setup
- `main.py` - Main entry point
- `install_pytorch_cuda10.sh` - PyTorch CUDA 10 installation
- `setup_uv_cache.sh` - UV cache configuration

## Usage Examples

### Data Preparation
```bash
# Prepare a single dataset
uv run python scripts/data/prepare_counsel_dataset.py --max_samples 100

# Combine all datasets
uv run python scripts/data/combine_all_datasets.py
```

### Training
```bash
# Train on single GPU
uv run python scripts/training/train_qwen_counsel.py --config configs/config.json

# Submit SLURM job
sbatch scripts/jobs/train_job.sh
```

### Inference
```bash
# Run interactive inference
uv run python scripts/inference/inference.py --interactive
```

### Setup
```bash
# Test environment setup
uv run python scripts/setup/test_setup.py
```

## Migration Note

This directory was reorganized on 2025-11-21 to improve maintainability. All references in documentation and other scripts have been updated to reflect the new structure.

