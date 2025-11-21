# Evaluate Base Model (While Fine-Tuning is Running)

You can evaluate the base model first to get baseline metrics, then compare with your fine-tuned model later.

## Quick Command

```bash
uv run evaluation/scripts/evaluate_model.py \
    --base_model Qwen/Qwen2.5-7B-Instruct \
    --test_dataset datasets/all_mental_health_combined \
    --output base_model_results.json \
    --max_samples 100
```

**Note:** No `--model_path` needed! The script will automatically use the base model only.

## What You'll Get

The evaluation computes:
- **Perplexity** - Language modeling quality
- **BLEU/ROUGE** - Text overlap metrics  
- **Domain Quality** - Empathy, active listening, evidence-based techniques
- **Safety** - Harmful content detection
- **Response Properties** - Length, coherence

Results saved to `base_model_results.json`.

## After Fine-Tuning Completes

Once your fine-tuned model is ready, evaluate it:

```bash
uv run evaluation/scripts/evaluate_model.py \
    --model_path models/qwen2.5-counsel-chat-finetuned \
    --base_model Qwen/Qwen2.5-7B-Instruct \
    --test_dataset datasets/all_mental_health_combined \
    --output finetuned_model_results.json \
    --max_samples 100
```

Then compare the results to see improvements!

## Expected Baseline Results (Base Model)

- **Perplexity**: ~18-25 (higher = worse)
- **BLEU**: ~0.10-0.20 (lower)
- **ROUGE-L**: ~0.20-0.30 (lower)
- **Empathy Score**: ~0.30-0.50 (lower)
- **Safety Rate**: ~0.85-0.95

After fine-tuning, you should see:
- ✅ Lower perplexity (better language modeling)
- ✅ Higher BLEU/ROUGE (better text overlap)
- ✅ Higher empathy score (better counseling quality)
- ✅ Similar or better safety rate
