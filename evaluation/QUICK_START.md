# Quick Start: Model Evaluation

## ⚠️ Important Note

**Most HuggingFace datasets use deprecated formats** that don't work with newer versions of the datasets library.

**✅ SOLUTION: Use your existing validation split** - it's actually the BEST option!

---

## ✅ RECOMMENDED: Use Your Existing Validation Split

**No download needed!** Your validation split is already available and ready to use:

### Evaluate Base Model (While Fine-Tuning is Running)

Get baseline metrics first:

```bash
uv run evaluation/scripts/evaluate_model.py \
    --base_model Qwen/Qwen2.5-7B-Instruct \
    --test_dataset datasets/all_mental_health_combined \
    --output base_model_results.json \
    --max_samples 100
```

**Note:** No `--model_path` needed! The script will use the base model only.

### Evaluate Fine-Tuned Model (After Training Completes)

```bash
uv run evaluation/scripts/evaluate_model.py \
    --model_path models/qwen2.5-counsel-chat-finetuned \
    --base_model Qwen/Qwen2.5-7B-Instruct \
    --test_dataset datasets/all_mental_health_combined \
    --output finetuned_model_results.json \
    --max_samples 100
```

**Why this is best:**

- ✅ Already downloaded and processed
- ✅ Matches your training data format
- ✅ No compatibility issues
- ✅ Tests on similar data distribution
- ✅ Ready to use immediately

**Time**: 20-30 minutes

---

## Alternative: Download External Dataset (May Not Work)

⚠️ **Warning**: Most external datasets will fail to download due to deprecated formats.

If you still want to try:

```bash
# 1. See available options (with warnings)
uv run evaluation/scripts/download_eval_dataset.py --dataset list

# 2. Try downloading (will likely fail)
uv run evaluation/scripts/download_eval_dataset.py --dataset dailydialog
```

**If download fails** (which is likely), the script will show you how to use your validation split instead.

**Why external datasets fail:**

- HuggingFace datasets use old script formats
- Newer `datasets` library doesn't support them
- Would require downgrading: `pip install 'datasets==2.14.0'`

**Better solution:** Just use your validation split - it's more appropriate anyway!

---

## What You'll Get

The evaluation script computes:

- **Perplexity** - Language modeling quality
- **BLEU/ROUGE** - Text overlap metrics
- **Domain Quality** - Empathy, active listening, evidence-based techniques
- **Safety** - Harmful content detection
- **Response Properties** - Length, coherence

Results saved to JSON file for analysis and presentation.

---

## For FYP Presentation

Use the validation split evaluation - it's the most appropriate for demonstrating your model's performance on the data it was trained on.
