# Diagnosing Low BLEU/ROUGE Scores

If you're seeing BLEU and ROUGE scores near 0, this guide will help you diagnose the issue.

## Quick Diagnosis Steps

### Step 1: Re-run Evaluation with Saved Responses

First, re-run your evaluation with the `--save_responses` flag to save individual responses:

```bash
python evaluation/scripts/evaluate_model.py \
    --model_path models/qwen2.5-1.5b-blazing-fast \
    --base_model Qwen/Qwen2.5-1.5B-Instruct \
    --test_dataset datasets/all_mental_health_combined \
    --save_responses \
    --output evaluation_results_with_responses.json \
    --max_samples 100
```

### Step 2: Use the Diagnostic Script

Run the diagnostic script to see what's actually being generated:

```bash
python evaluation/scripts/diagnose_low_scores.py \
    --results_file evaluation_results_with_responses.json \
    --num_samples 10
```

This will show you:
- The actual generated responses vs reference responses
- Individual BLEU/ROUGE scores for each sample
- Potential issues (e.g., responses too short, no common words, etc.)

### Step 3: Compare Models Side-by-Side

Use the comparison script to see base vs trained model outputs:

```bash
python evaluation/scripts/compare_models.py \
    --trained_model models/qwen2.5-1.5b-blazing-fast \
    --base_model Qwen/Qwen2.5-1.5B-Instruct \
    --test_file evaluation/scripts/sample\ for\ testing\ models.md \
    --output comparison_results.json
```

## Common Issues and Solutions

### Issue 1: Generated Responses Are Empty or Very Short

**Symptoms:**
- Generated responses are < 10 characters
- BLEU/ROUGE scores are 0

**Possible Causes:**
- Response extraction is cutting off too early
- Model is not generating properly
- Prompt format doesn't match training format

**Solutions:**
- Check the `generate_response()` function in `evaluate_model.py`
- Verify prompt format matches training format
- Check if `max_new_tokens` is too low
- Inspect raw model outputs before extraction

### Issue 2: Generated Responses Don't Match Reference Format

**Symptoms:**
- Responses are generated but completely different from references
- No common words between generated and reference

**Possible Causes:**
- Model is generating in a different style/format
- Training data format doesn't match evaluation format
- Model needs more training

**Solutions:**
- Compare a few samples manually
- Check training data format
- Verify model was trained correctly

### Issue 3: Response Extraction Issues

**Symptoms:**
- Responses seem cut off
- Stop patterns are triggering too early

**Solutions:**
- Check stop patterns in `generate_response()`
- Verify the prompt extraction logic
- Try generating without stop patterns to see full output

## Manual Inspection

To manually inspect what's happening:

1. **Check a single response:**
   ```python
   from evaluation.scripts.evaluate_model import load_model, generate_response
   
   model, tokenizer = load_model("models/qwen2.5-1.5b-blazing-fast", ...)
   response = generate_response(model, tokenizer, "I'm feeling sad. Can you help?")
   print(response)
   ```

2. **Compare with reference:**
   - Load your test dataset
   - Generate response for a specific sample
   - Compare manually with the reference

3. **Check prompt format:**
   - Print the actual prompt being sent to the model
   - Verify it matches the training format

## Next Steps

After diagnosing:
1. Fix any response extraction issues
2. Adjust prompt format if needed
3. Re-train model if responses are completely off
4. Re-run evaluation and check if scores improve

