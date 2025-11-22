# 📁 Directory Organization Summary

This document describes the recent directory cleanup and organization performed on 2025-11-22.

## Changes Made

### 1. Troubleshooting Documentation
**Location:** `docs/troubleshooting/`

Moved troubleshooting and fix documentation to a dedicated directory:
- `DISK_QUOTA_FIX.md` - Guide for fixing disk quota issues
- `OOM_FIX_APPLIED.md` - Out of Memory error fixes
- `NEW_1.5B_SETUP_SUMMARY.md` - 1.5B model setup summary

### 2. Setup Guides
**Location:** `docs/setup/`

Consolidated quick start guides:
- `QUICK_START_1.5B.md` - Quick start for 1.5B model training
- `README_1.5B_FAST_TRAINING.md` - Complete fast training setup guide

### 3. Presentation Materials
**Location:** `docs/presentation/`

Added presentation documentation:
- `PRESENTATION_GUIDE.md` - Guide for creating presentations
- `PRESENTATION_SUMMARY.md` - Summary of presentation materials

**Location:** `present_png/`

All presentation images consolidated in one directory with subdirectories:
- `architecture_explaination/` - Architecture diagrams
- `model_selection/` - Model selection charts
- Various dataset visualization images

### 4. Training Logs
**Location:** `logs/archive/`

Archived all old training logs (24 files):
- 12 `.err` files - Error logs from training jobs
- 12 `.out` files - Output logs from training jobs

### 5. Removed Directories

Cleaned up duplicate and empty directories:
- `yyy@localhost/` - Duplicate presentation files (removed)
- `present_png_dataset/` - Empty directory (removed)
- `scripts/__pycache__/` - Python cache files (removed)

## Current Directory Structure

```
FYP-LLM/
├── configs/           # Configuration files for different model sizes
├── datasets/          # All training datasets (processed and raw)
├── docs/              # All documentation (organized by category)
│   ├── datasets/      # Dataset guides
│   ├── evaluation/    # Evaluation documentation
│   ├── hardware/      # Hardware setup guides
│   ├── model-selection/ # Model selection justification
│   ├── presentation/  # Presentation materials and guides
│   ├── setup/         # Setup and quick start guides
│   ├── system/        # System commands and usage
│   ├── training/      # Training guides and documentation
│   └── troubleshooting/ # Fix guides and troubleshooting
├── evaluation/        # Evaluation scripts and documentation
├── logs/              # Training logs
│   └── archive/       # Archived old training logs
├── models/            # Trained model checkpoints
├── present_png/       # Presentation images and charts
├── samples/           # Sample data files
├── scripts/           # All executable scripts
│   ├── data/          # Data processing scripts
│   ├── inference/     # Inference scripts
│   ├── jobs/          # SLURM job scripts
│   ├── setup/         # Setup scripts
│   ├── training/      # Training scripts
│   └── visualization/ # Visualization scripts
├── README.md          # Main project README
├── pyproject.toml     # Project dependencies
└── uv.lock            # Locked dependencies
```

## Benefits

1. **Better Organization**: All documentation is now categorized and easy to find
2. **Cleaner Root**: Root directory only contains essential files
3. **Archived Logs**: Old training logs are preserved but organized
4. **No Duplicates**: Removed duplicate presentation directories
5. **Clean Cache**: Removed unnecessary Python cache files

## Finding Files

### Documentation
- **Setup guides** → `docs/setup/`
- **Troubleshooting** → `docs/troubleshooting/`
- **Dataset guides** → `docs/datasets/`
- **Training guides** → `docs/training/`
- **Presentation materials** → `docs/presentation/`

### Scripts
- **Training scripts** → `scripts/training/`
- **Job submission** → `scripts/jobs/`
- **Data processing** → `scripts/data/`

### Logs
- **Current logs** → `logs/` (empty after cleanup)
- **Archived logs** → `logs/archive/`

---

*Last updated: 2025-11-22*

