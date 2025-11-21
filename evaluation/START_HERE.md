# 🎯 START HERE - Evaluation Quick Start

## Welcome to the Evaluation Directory!

This directory contains everything you need to evaluate your fine-tuned mental health counseling model using the **EmpatheticDialogues** dataset.

---

## ⚡ Quick Start (3 Steps)

### 1️⃣ Read the Overview (2 minutes)
```bash
# Open the main guide
cat evaluation/README.md
# or just open it in your editor
```

### 2️⃣ Run Automated Evaluation (30 minutes)
```bash
# From project root
./evaluation/scripts/run_full_evaluation.sh models/qwen2.5-counsel-chat-finetuned 100
```

### 3️⃣ Review Results
- Check console output for comparison table
- Review generated JSON files
- Look at example conversations

**Done! You now have quantitative proof your model works! 🎉**

---

## 📚 Documentation Hierarchy

```
📖 START_HERE.md              ← You are here!
    ↓
📖 README.md                  ← Main guide (read first)
    ↓
📖 docs/EVALUATION_QUICKREF.md    ← Quick reference
📖 docs/EVALUATION_EMPATHETIC_GUIDE.md  ← Detailed guide
    ↓
📖 INDEX.md                   ← Complete reference
📖 STRUCTURE.md               ← Directory organization
```

**Recommendation:** Read in this order!

---

## 🛠️ What's Available

### Documentation (in `docs/`)
- **EVALUATION_QUICKREF.md** - One-page cheat sheet (2 min read)
- **EVALUATION_EMPATHETIC_GUIDE.md** - Comprehensive guide (15 min read)

### Scripts (in `scripts/`)
- **run_full_evaluation.sh** - Automated pipeline ⭐ (easiest!)
- **download_eval_dataset.py** - Download EmpatheticDialogues
- **evaluate_on_empathetic.py** - Evaluate single model
- **compare_base_vs_finetuned.py** - Compare two models

---

## 🎯 Two Ways to Evaluate

### Option A: Automated (Recommended)
```bash
./evaluation/scripts/run_full_evaluation.sh models/qwen2.5-counsel-chat-finetuned 100
```
**Time:** 30-40 minutes  
**What it does:** Everything automatically!

### Option B: Manual (More Control)
```bash
# 1. Download dataset (2 min)
uv run evaluation/scripts/download_eval_dataset.py

# 2. Evaluate base model (15 min)
uv run evaluation/scripts/evaluate_on_empathetic.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --output results_base.json

# 3. Evaluate fine-tuned model (15 min)
uv run evaluation/scripts/evaluate_on_empathetic.py \
    --model models/qwen2.5-counsel-chat-finetuned \
    --output results_finetuned.json

# 4. Compare (instant)
uv run evaluation/scripts/compare_base_vs_finetuned.py \
    --base results_base.json \
    --finetuned results_finetuned.json
```

---

## 📊 What You'll Get

### Quantitative Metrics
- ✅ Perplexity (language quality)
- ✅ BLEU & ROUGE (response quality)
- ✅ **Empathy Score** (most important!)

### Qualitative Examples
- ✅ Side-by-side comparisons
- ✅ Shows improvement in empathy
- ✅ Ready for presentation

### Comparison Report
- ✅ Base vs Fine-tuned table
- ✅ Improvement percentages
- ✅ JSON file for analysis

---

## 🎓 For Your FYP

This evaluation provides:
1. **Independent dataset** (EmpatheticDialogues)
2. **Quantitative proof** (metrics showing improvement)
3. **Qualitative evidence** (example conversations)
4. **Professional presentation** (tables & comparisons)

Perfect for demonstrating your model works! ✨

---

## 💡 Need Help?

### Quick Questions?
- Check: `docs/EVALUATION_QUICKREF.md`

### Detailed Guide?
- Read: `docs/EVALUATION_EMPATHETIC_GUIDE.md`

### Understanding Metrics?
- See: `../MODEL_EVALUATION_GUIDE.md` (in project root)

### Complete Reference?
- Check: `INDEX.md`

---

## ✅ Next Step

**Run your first evaluation right now:**

```bash
./evaluation/scripts/run_full_evaluation.sh models/qwen2.5-counsel-chat-finetuned 100
```

**Time needed:** 30-40 minutes  
**Difficulty:** Easy (fully automated)  
**Result:** Quantitative proof your fine-tuning works!

---

## 🎉 You're All Set!

Everything is ready to go:
- ✅ Scripts are executable
- ✅ Documentation is complete  
- ✅ Paths are all updated
- ✅ Structure is organized

**Just run the evaluation and you're done!** 🚀

---

**Questions?** Check `README.md` or `INDEX.md` for more details!

