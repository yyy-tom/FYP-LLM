# Evaluation Directory - Complete Index

This directory contains all resources for evaluating your fine-tuned mental health counseling model using the **EmpatheticDialogues** dataset.

## 📂 Directory Structure

```
evaluation/
├── README.md                    # Main evaluation guide (START HERE!)
├── INDEX.md                     # This file - complete index
├── docs/                        # Documentation
│   ├── EVALUATION_QUICKREF.md            # One-page quick reference
│   ├── EVALUATION_EMPATHETIC_GUIDE.md    # Detailed step-by-step guide
│   └── MODEL_EVALUATION_GUIDE.md         # Comprehensive evaluation theory
└── scripts/                     # Evaluation scripts
    ├── download_eval_dataset.py          # Download EmpatheticDialogues
    ├── evaluate_on_empathetic.py         # Evaluate single model (EmpatheticDialogues)
    ├── evaluate_model.py                 # Comprehensive evaluation (all metrics)
    ├── create_expert_evaluation_dataset.py # Create expert review dataset
    ├── compare_base_vs_finetuned.py      # Compare two models
    └── run_full_evaluation.sh            # Automated full pipeline
```

## 📖 Documentation Guide

### 🚀 Quick Start

1. **Start here**: [`README.md`](README.md) - Complete overview with examples
2. **Quick reference**: [`docs/EVALUATION_QUICKREF.md`](docs/EVALUATION_QUICKREF.md) - One-page cheat sheet
3. **Detailed guide**: [`docs/EVALUATION_EMPATHETIC_GUIDE.md`](docs/EVALUATION_EMPATHETIC_GUIDE.md) - Step-by-step instructions
4. **Comprehensive theory**: [`docs/MODEL_EVALUATION_GUIDE.md`](docs/MODEL_EVALUATION_GUIDE.md) - All evaluation methods

### 📚 Additional Resources

- **Training datasets**: `../DATASETS_FROM_PAPER_GUIDE.md` (in project root)

## 🛠️ Scripts Guide

### Core Scripts

#### 1. `download_eval_dataset.py`

Downloads the EmpatheticDialogues dataset from HuggingFace.

**Usage:**

```bash
uv run evaluation/scripts/download_eval_dataset.py
```

**Time**: ~2 minutes  
**Output**: `datasets/empathetic_dialogues_eval/`

---

#### 2. `evaluate_on_empathetic.py`

Evaluates a single model on the EmpatheticDialogues dataset.

**Usage:**

```bash
uv run evaluation/scripts/evaluate_on_empathetic.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --max_samples 100 \
    --output results.json
```

**Arguments:**

- `--model`: Path to model (HuggingFace or local)
- `--max_samples`: Number of samples to evaluate (default: 100)
- `--output`: Output JSON file for results
- `--device`: cuda or cpu (default: cuda if available)
- `--dataset`: Path to dataset (default: datasets/empathetic_dialogues_eval)

**Time**: 10-20 minutes (GPU), 1-2 hours (CPU)  
**Output**: JSON file with metrics and examples

**Metrics Computed:**

- Perplexity
- BLEU score
- ROUGE-1, ROUGE-2, ROUGE-L
- Empathy score (custom metric)

---

#### 2b. `evaluate_model.py`

Comprehensive evaluation script with all metrics (perplexity, BLEU, ROUGE, domain quality, safety).

**Usage:**

```bash
uv run evaluation/scripts/evaluate_model.py \
    --model_path models/qwen2.5-counsel-chat-finetuned \
    --base_model Qwen/Qwen2.5-7B-Instruct \
    --test_dataset datasets/all_mental_health_combined \
    --output evaluation_results.json \
    --max_samples 100
```

**Arguments:**

- `--model_path`: Path to fine-tuned model (LoRA weights)
- `--base_model`: Base model name
- `--test_dataset`: Path to test dataset
- `--output`: Output JSON file
- `--max_samples`: Number of samples (default: 100)
- `--device`: cuda or cpu (default: cuda)

**Time**: 20-30 minutes (GPU)  
**Output**: Comprehensive JSON with all metrics

**Metrics Computed:**

- Perplexity
- BLEU, ROUGE-1, ROUGE-2, ROUGE-L
- Domain quality (empathy, active listening, evidence-based, boundaries, crisis detection)
- Safety evaluation
- Response properties (length, coherence)

---

#### 2c. `create_expert_evaluation_dataset.py`

Creates dataset for expert evaluation by mental health professionals.

**Usage:**

```bash
uv run evaluation/scripts/create_expert_evaluation_dataset.py \
    --model_path models/qwen2.5-counsel-chat-finetuned \
    --base_model Qwen/Qwen2.5-7B-Instruct \
    --test_dataset datasets/all_mental_health_combined \
    --output expert_evaluation.jsonl \
    --num_samples 50
```

**Arguments:**

- `--model_path`: Path to fine-tuned model
- `--base_model`: Base model name
- `--test_dataset`: Path to test dataset
- `--output`: Output JSONL file
- `--num_samples`: Number of samples (default: 50)
- `--device`: cuda or cpu (default: cuda)

**Time**: 10-15 minutes  
**Output**: JSONL file with samples for expert review

**Use Case**: Share with mental health professionals for qualitative evaluation

---

#### 3. `compare_base_vs_finetuned.py`

Compares evaluation results between two models.

**Usage:**

```bash
uv run evaluation/scripts/compare_base_vs_finetuned.py \
    --base results_base.json \
    --finetuned results_finetuned.json \
    --num_examples 5
```

**Arguments:**

- `--base`: Path to base model results JSON
- `--finetuned`: Path to fine-tuned model results JSON
- `--output`: Output comparison report JSON (default: comparison_report.json)
- `--num_examples`: Number of example comparisons to show (default: 3)

**Time**: Instant  
**Output**:

- Console: Comparison table and examples
- File: JSON comparison report

---

#### 4. `run_full_evaluation.sh`

Automated pipeline that runs the entire evaluation process.

**Usage:**

```bash
./evaluation/scripts/run_full_evaluation.sh [model-path] [num-samples]
```

**Example:**

```bash
./evaluation/scripts/run_full_evaluation.sh models/qwen2.5-counsel-chat-finetuned 100
```

**What it does:**

1. Downloads dataset (if needed)
2. Evaluates base model
3. Evaluates fine-tuned model
4. Generates comparison report

**Time**: 30-40 minutes total  
**Output**: Three JSON files (base results, fine-tuned results, comparison)

---

## 🎯 Common Workflows

### Workflow 1: Complete Evaluation (Automated)

```bash
# Run everything in one command
./evaluation/scripts/run_full_evaluation.sh models/qwen2.5-counsel-chat-finetuned 100
```

### Workflow 2: Manual Step-by-Step

```bash
# Step 1: Download dataset
uv run evaluation/scripts/download_eval_dataset.py

# Step 2: Evaluate base model
uv run evaluation/scripts/evaluate_on_empathetic.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --output results_base.json

# Step 3: Evaluate fine-tuned model
uv run evaluation/scripts/evaluate_on_empathetic.py \
    --model models/qwen2.5-counsel-chat-finetuned \
    --output results_finetuned.json

# Step 4: Compare
uv run evaluation/scripts/compare_base_vs_finetuned.py \
    --base results_base.json \
    --finetuned results_finetuned.json
```

### Workflow 3: Evaluate Single Model Only

```bash
# Just evaluate your fine-tuned model
uv run evaluation/scripts/evaluate_on_empathetic.py \
    --model models/qwen2.5-counsel-chat-finetuned \
    --max_samples 200 \
    --output my_model_results.json
```

### Workflow 4: Compare 7B vs 14B Models

```bash
# Evaluate both
uv run evaluation/scripts/evaluate_on_empathetic.py \
    --model models/qwen2.5-7b-finetuned \
    --output results_7b.json

uv run evaluation/scripts/evaluate_on_empathetic.py \
    --model models/qwen2.5-14b-finetuned \
    --output results_14b.json

# Compare
uv run evaluation/scripts/compare_base_vs_finetuned.py \
    --base results_7b.json \
    --finetuned results_14b.json
```

---

## 📊 Understanding Results

### Metrics Explanation

| Metric            | Range               | Interpretation            | Target               |
| ----------------- | ------------------- | ------------------------- | -------------------- |
| **Perplexity**    | 1-∞ (lower better)  | Language modeling quality | ↓ 30-40% vs base     |
| **BLEU**          | 0-1 (higher better) | Word overlap accuracy     | ↑ 50-100% vs base    |
| **ROUGE-1**       | 0-1 (higher better) | Unigram overlap           | ↑ 30-50% vs base     |
| **ROUGE-L**       | 0-1 (higher better) | Longest sequence match    | ↑ 30-50% vs base     |
| **Empathy Score** | 0-1 (higher better) | Empathetic language use   | ↑ 70-100% vs base ⭐ |

**Most Important**: Empathy Score - Shows counseling-specific improvement!

### Expected Results

**Base Model (Qwen 2.5-7B):**

- Perplexity: 18-25
- Empathy: 0.30-0.50
- BLEU: 0.05-0.15

**Fine-Tuned Model (Target):**

- Perplexity: 10-15 (⬇️ 40% better)
- Empathy: 0.65-0.85 (⬆️ 100% better) ⭐
- BLEU: 0.15-0.25 (⬆️ 80% better)

---

## 🐛 Troubleshooting

### Common Issues

**Issue: CUDA out of memory**

```bash
# Solution: Use CPU or reduce samples
uv run evaluation/scripts/evaluate_on_empathetic.py \
    --model models/qwen2.5-counsel-chat-finetuned \
    --device cpu \
    --max_samples 50 \
    --output results.json
```

**Issue: Missing NLTK data**

```bash
# Solution: Download NLTK resources
python -c "import nltk; nltk.download('punkt')"
```

**Issue: rouge-score not installed**

```bash
# Solution: Install dependencies
uv pip install rouge-score nltk
```

**Issue: Dataset not found**

```bash
# Solution: Re-download dataset
uv run evaluation/scripts/download_eval_dataset.py
```

---

## ✅ Quality Checklist

Before presenting your results:

- [ ] Dataset downloaded successfully
- [ ] Base model evaluated (results look reasonable)
- [ ] Fine-tuned model evaluated
- [ ] Comparison report generated
- [ ] Empathy score improved (most important!)
- [ ] Sample outputs reviewed for quality
- [ ] Results documented for presentation
- [ ] Visualizations created (optional)

---

## 🎓 For Your FYP

### What to Present

1. **Methodology**

   - "Evaluated on EmpatheticDialogues dataset"
   - "25K conversations, not used in training"
   - "Independent evaluation of generalization"

2. **Quantitative Results**

   - Show comparison table
   - Highlight empathy improvement
   - Explain perplexity reduction

3. **Qualitative Examples**

   - 3-5 side-by-side comparisons
   - Show how fine-tuned is more empathetic
   - Highlight professional language

4. **Conclusion**
   - "Fine-tuning successfully adapted model to counseling domain"
   - "X% improvement in empathy score"
   - "Y% reduction in perplexity"

---

## 📞 Quick Reference Links

- **Main Guide**: [`README.md`](README.md)
- **Quick Start**: [`docs/EVALUATION_QUICKREF.md`](docs/EVALUATION_QUICKREF.md)
- **Detailed Guide**: [`docs/EVALUATION_EMPATHETIC_GUIDE.md`](docs/EVALUATION_EMPATHETIC_GUIDE.md)
- **Comprehensive Theory**: [`docs/MODEL_EVALUATION_GUIDE.md`](docs/MODEL_EVALUATION_GUIDE.md)

---

## 🎉 Summary

This evaluation framework provides:

- ✅ Independent dataset (EmpatheticDialogues)
- ✅ Automated scripts (4 ready-to-use tools)
- ✅ Comprehensive documentation (3 guides)
- ✅ Quantitative metrics (5 metrics)
- ✅ Qualitative examples (side-by-side comparisons)
- ✅ Presentation-ready results

**Total time**: 30-60 minutes  
**Result**: Strong evidence your fine-tuning works! 🎓
