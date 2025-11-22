# Evaluation Directory Structure

## 📂 Complete Organization

```
FYP-LLM/
│
├── evaluation/                           # ← ALL EVALUATION FILES HERE
│   │
│   ├── README.md                         # Main guide - START HERE!
│   ├── INDEX.md                          # Complete index & reference
│   ├── STRUCTURE.md                      # This file - directory structure
│   │
│   ├── docs/                             # Documentation
│   │   ├── EVALUATION_QUICKREF.md        # One-page quick reference
│   │   └── EVALUATION_EMPATHETIC_GUIDE.md # Detailed step-by-step guide
│   │
│   └── scripts/                          # Evaluation scripts
│       ├── download_eval_dataset.py      # Download EmpatheticDialogues
│       ├── evaluate_on_empathetic.py     # Evaluate single model
│       ├── compare_base_vs_finetuned.py  # Compare two models
│       └── run_full_evaluation.sh        # Automated full pipeline
│
├── datasets/
│   └── empathetic_dialogues_eval/        # Downloaded by scripts
│
├── results_*.json                        # Generated evaluation results
├── comparison_report_*.json              # Generated comparison reports
│
└── MODEL_EVALUATION_GUIDE.md             # Comprehensive theory (root)
```

## 🎯 Quick Navigation

### For First-Time Users
1. **Start**: `evaluation/README.md`
2. **Quick ref**: `evaluation/docs/EVALUATION_QUICKREF.md`
3. **Run**: `./evaluation/scripts/run_full_evaluation.sh`

### For Script Usage
- **All scripts**: `evaluation/scripts/`
- **Download dataset**: `evaluation/scripts/download_eval_dataset.py`
- **Evaluate model**: `evaluation/scripts/evaluate_on_empathetic.py`
- **Compare results**: `evaluation/scripts/compare_base_vs_finetuned.py`

### For Documentation
- **Overview**: `evaluation/README.md`
- **Quick reference**: `evaluation/docs/EVALUATION_QUICKREF.md`
- **Detailed guide**: `evaluation/docs/EVALUATION_EMPATHETIC_GUIDE.md`
- **Complete index**: `evaluation/INDEX.md`
- **Theory**: `MODEL_EVALUATION_GUIDE.md` (root)

## 📋 File Descriptions

### Documentation Files

| File | Purpose | Read Time |
|------|---------|-----------|
| `evaluation/README.md` | Complete overview with all details | 10 min |
| `evaluation/docs/EVALUATION_QUICKREF.md` | One-page cheat sheet | 2 min |
| `evaluation/docs/EVALUATION_EMPATHETIC_GUIDE.md` | Step-by-step instructions | 15 min |
| `evaluation/INDEX.md` | Complete reference & workflows | 5 min |
| `evaluation/STRUCTURE.md` | Directory organization (this file) | 2 min |

### Script Files

| File | Purpose | Run Time |
|------|---------|----------|
| `evaluation/scripts/download_eval_dataset.py` | Download dataset | 2 min |
| `evaluation/scripts/evaluate_on_empathetic.py` | Evaluate one model | 15 min |
| `evaluation/scripts/compare_base_vs_finetuned.py` | Compare results | instant |
| `evaluation/scripts/run_full_evaluation.sh` | Full pipeline | 30-40 min |

## 🚀 Quick Start Commands

### From Project Root (`/Users/yyy/FYP-LLM/`)

```bash
# Automated (easiest)
./evaluation/scripts/run_full_evaluation.sh models/qwen2.5-counsel-chat-finetuned 100

# Manual
uv run evaluation/scripts/download_eval_dataset.py
uv run evaluation/scripts/evaluate_on_empathetic.py --model Qwen/Qwen2.5-7B-Instruct --output results_base.json
uv run evaluation/scripts/evaluate_on_empathetic.py --model models/qwen2.5-counsel-chat-finetuned --output results_ft.json
uv run evaluation/scripts/compare_base_vs_finetuned.py --base results_base.json --finetuned results_ft.json
```

### From Evaluation Directory (`/Users/yyy/FYP-LLM/evaluation/`)

```bash
cd evaluation

# Automated
./scripts/run_full_evaluation.sh ../models/qwen2.5-counsel-chat-finetuned 100

# Manual
uv run scripts/download_eval_dataset.py
uv run scripts/evaluate_on_empathetic.py --model Qwen/Qwen2.5-7B-Instruct --output ../results_base.json
uv run scripts/evaluate_on_empathetic.py --model ../models/qwen2.5-counsel-chat-finetuned --output ../results_ft.json
uv run scripts/compare_base_vs_finetuned.py --base ../results_base.json --finetuned ../results_ft.json
```

## 🎨 Benefits of This Organization

### ✅ Clean Separation
- All evaluation files in one place
- Easy to find and use
- No clutter in root directory

### ✅ Logical Hierarchy
- `docs/` for documentation
- `scripts/` for executable files
- Clear naming conventions

### ✅ Easy Navigation
- README at top level
- INDEX for complete reference
- STRUCTURE for overview

### ✅ Scalable
- Easy to add new evaluation datasets
- Easy to add new metrics
- Easy to add new comparison methods

## 📝 Notes

- All paths updated to reflect new structure
- Scripts work from project root
- Results saved to project root by default
- Dataset downloaded to `datasets/` as before

## 🔄 Migration Summary

**Before:**
```
FYP-LLM/
├── EVALUATION_SUMMARY.md
├── EVALUATION_QUICKREF.md
├── EVALUATION_EMPATHETIC_GUIDE.md
└── scripts/
    ├── download_eval_dataset.py
    ├── evaluate_on_empathetic.py
    ├── compare_base_vs_finetuned.py
    └── run_full_evaluation.sh
```

**After:**
```
FYP-LLM/
└── evaluation/              # ← Everything organized here!
    ├── README.md
    ├── docs/
    │   ├── EVALUATION_QUICKREF.md
    │   └── EVALUATION_EMPATHETIC_GUIDE.md
    └── scripts/
        ├── download_eval_dataset.py
        ├── evaluate_on_empathetic.py
        ├── compare_base_vs_finetuned.py
        └── run_full_evaluation.sh
```

**Changes Made:**
- ✅ Created `evaluation/` directory
- ✅ Moved all evaluation docs to `evaluation/docs/`
- ✅ Moved all evaluation scripts to `evaluation/scripts/`
- ✅ Updated all path references
- ✅ Created comprehensive index and structure docs
- ✅ Renamed EVALUATION_SUMMARY.md → evaluation/README.md

## ✨ Result

**Much cleaner project structure!** 🎉

All evaluation resources are now organized in a dedicated `evaluation/` directory with:
- Clear documentation hierarchy
- Logical script organization
- Easy-to-find resources
- Professional structure for FYP








