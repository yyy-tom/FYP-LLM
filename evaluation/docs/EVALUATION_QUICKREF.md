# Evaluation Quick Reference Card

## 🎯 Recommended: EmpatheticDialogues Dataset

**Best choice for evaluating your mental health counseling model**

### Why?

- ✅ Not in training data (independent evaluation)
- ✅ Tests empathy (critical for counseling)
- ✅ 25,000 high-quality conversations
- ✅ Easy access (HuggingFace)
- ✅ 32 emotion categories

---

## ⚡ 3-Step Evaluation

### Step 1: Download Dataset (2 min)

```bash
uv run evaluation/scripts/download_eval_dataset.py
```

### Step 2: Evaluate Models (30 min total)

```bash
# Base model
uv run evaluation/scripts/evaluate_on_empathetic.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --output results_base.json

# Fine-tuned model
uv run evaluation/scripts/evaluate_on_empathetic.py \
    --model models/qwen2.5-counsel-chat-finetuned \
    --output results_finetuned.json
```

### Step 3: Compare (instant)

```bash
uv run evaluation/scripts/compare_base_vs_finetuned.py \
    --base results_base.json \
    --finetuned results_finetuned.json
```

---

## 📊 Key Metrics

| Metric            | What It Measures            | Target                    |
| ----------------- | --------------------------- | ------------------------- |
| **Perplexity**    | Language modeling quality   | ⬇️ 30-40% lower than base |
| **BLEU**          | Word overlap with reference | ⬆️ 50-100% higher         |
| **ROUGE**         | Content coverage            | ⬆️ 30-50% higher          |
| **Empathy Score** | Empathetic language use     | ⬆️ 70-100% higher ⭐      |

**Most Important**: Empathy Score (shows counseling improvement)

---

## 📈 Expected Results

### Base Model (Qwen 2.5-7B):

```
Perplexity:    18-25
Empathy:       0.30-0.50
BLEU:          0.05-0.15
```

### Fine-Tuned Model (Target):

```
Perplexity:    10-15  (⬇️ 40% better)
Empathy:       0.65-0.85  (⬆️ 100% better) ⭐
BLEU:          0.15-0.25  (⬆️ 80% better)
```

---

## 📝 For Your Presentation

Show these 4 things:

1. **Comparison Table** (from comparison script)
2. **3-5 Example Conversations** (side-by-side)
3. **Empathy Improvement** (highlight this!)
4. **Independent Dataset** (mention EmpatheticDialogues)

---

## 🐛 Quick Troubleshooting

**Out of memory?**

```bash
--max_samples 50 --device cpu
```

**Missing NLTK?**

```bash
python -c "import nltk; nltk.download('punkt')"
uv pip install rouge-score nltk
```

---

## 📚 Full Documentation

- **Detailed Guide**: `evaluation/docs/EVALUATION_EMPATHETIC_GUIDE.md`
- **All Metrics**: `MODEL_EVALUATION_GUIDE.md` (root)
- **Training Datasets**: `DATASETS_FROM_PAPER_GUIDE.md` (root)

---

## ✅ Checklist

- [ ] Downloaded EmpatheticDialogues dataset
- [ ] Evaluated base model
- [ ] Evaluated fine-tuned model
- [ ] Generated comparison report
- [ ] Reviewed sample outputs
- [ ] Documented results for presentation
- [ ] Created comparison visualizations

---

**Total Time**: ~1 hour
**Result**: Quantitative proof your fine-tuning works! 🎉
