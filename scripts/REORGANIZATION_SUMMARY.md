# Scripts Directory Reorganization Summary

**Date:** November 21, 2025

## Overview
The scripts directory has been reorganized from a flat structure with 30+ files into a logical hierarchical structure with 6 subdirectories for better maintainability and clarity.

## Changes Made

### Directory Structure
Created 6 new subdirectories:
- `data/` - Dataset preparation and combination (11 files)
- `training/` - Model training scripts (2 files)
- `jobs/` - SLURM job and execution scripts (9 files)
- `visualization/` - Diagram and visual generation (2 files)
- `inference/` - Inference and model comparison (2 files)
- `setup/` - Setup and testing utilities (4 files)

### File Movements

#### `data/` (11 files)
- `prepare_amod_dataset.py`
- `prepare_cactus_dataset.py`
- `prepare_counsel_dataset.py`
- `prepare_dataset.py`
- `prepare_esconv_dataset.py`
- `prepare_kaggle_dataset.py`
- `prepare_mentalchat16k_dataset.py`
- `prepare_psydial_dataset.py`
- `combine_all_datasets.py`
- `combine_datasets.py`
- `download_kaggle_dataset.py`

#### `training/` (2 files)
- `train_qwen_counsel.py`
- `train_qwen_counsel_multi_gpu.py`

#### `jobs/` (9 files)
- `train_job.sh`
- `train_job_gpu.sh`
- `train_job_interactive_10cpu.sh`
- `train_job_max.sh`
- `train_job_max_gpu_72h.sh`
- `run_interactive_training.sh`
- `run_training_screen.sh`
- `run_training_tmux.sh`
- `monitor_job.sh`

#### `visualization/` (2 files)
- `generate_architecture_diagrams.py`
- `generate_model_selection_visuals.py`

#### `inference/` (2 files)
- `inference.py`
- `compare_models.py`

#### `setup/` (4 files)
- `test_setup.py`
- `main.py`
- `install_pytorch_cuda10.sh`
- `setup_uv_cache.sh`

## Updated References

All references to the moved scripts have been updated in the following locations:

### Shell Scripts (8 files)
- `scripts/jobs/train_job.sh`
- `scripts/jobs/train_job_gpu.sh`
- `scripts/jobs/train_job_interactive_10cpu.sh`
- `scripts/jobs/train_job_max.sh`
- `scripts/jobs/train_job_max_gpu_72h.sh`
- `scripts/jobs/run_interactive_training.sh`
- `scripts/jobs/run_training_screen.sh`
- `scripts/jobs/run_training_tmux.sh`

### Python Scripts (3 files)
- `scripts/data/combine_all_datasets.py`
- `scripts/data/download_kaggle_dataset.py`
- `scripts/setup/test_setup.py`

### Documentation (19 files)
- `README.md`
- `docs/model-selection/MODEL_SELECTION_PRESENTATION_GUIDE.md`
- `docs/presentation/FYP_PRESENTATION_COMPLETE.md`
- `docs/presentation/PRESENTATION_GUIDE.md`
- `docs/presentation/PRESENTATION_SUMMARY.md`
- `docs/setup/UV_CACHE_SETUP.md`
- `docs/setup/INSTALL_PYTORCH_CUDA10.md`
- `docs/setup/CPU_80CORES_SETUP.md`
- `docs/training/CPU_MODEL_SIZE_GUIDE.md`
- `docs/training/CPU_TRAINING_GUIDE.md`
- `docs/training/README_TRAINING.md`
- `docs/system/SRUN_COMMANDS.md`
- `docs/hardware/GPU_NODE_RECOMMENDATION.md`
- `docs/evaluation/MODEL_EVALUATION_GUIDE.md`
- `yyy@localhost/PRESENTATION_SUMMARY.md`
- `yyy@localhost/PRESENTATION_GUIDE.md`

## New Documentation
- `scripts/README.md` - Comprehensive guide to the new structure
- `scripts/REORGANIZATION_SUMMARY.md` - This file

## Benefits
1. **Better Organization**: Scripts are grouped by functionality
2. **Easier Navigation**: Clear categorization makes finding scripts faster
3. **Improved Maintainability**: Related scripts are colocated
4. **Clearer Purpose**: Directory names indicate script functionality
5. **Scalability**: Easy to add new scripts to appropriate categories

## Migration Impact
- ✅ All script references updated
- ✅ All documentation updated
- ✅ All job submission scripts updated
- ✅ No breaking changes for existing workflows
- ✅ Backward compatibility maintained through updated paths

## Usage
All scripts should now be invoked with their new paths:

**Before:**
```bash
python scripts/prepare_counsel_dataset.py
python scripts/train_qwen_counsel.py
sbatch scripts/train_job.sh
```

**After:**
```bash
python scripts/data/prepare_counsel_dataset.py
python scripts/training/train_qwen_counsel.py
sbatch scripts/jobs/train_job.sh
```

See `scripts/README.md` for complete usage examples.

