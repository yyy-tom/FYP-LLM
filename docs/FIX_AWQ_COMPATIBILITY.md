# Fixing AWQ Compatibility Issues

If you're getting `ImportError: cannot import name 'PytorchGELUTanh'`, this is a version compatibility issue between `autoawq` and `transformers`.

## Solution 1: Upgrade autoawq (Recommended)

Upgrade to the latest version of `autoawq` which is compatible with newer `transformers`:

```bash
# On your HPC cluster
cd /research/d7/fyp25/yyyu2/FYP-LLM

# Upgrade autoawq to latest version
uv pip install --upgrade autoawq

# Or reinstall with latest version
uv pip uninstall autoawq
uv pip install "autoawq>=0.2.3"
```

## Solution 2: Use BitsAndBytes Instead (Alternative)

If upgrading doesn't work, switch to using BitsAndBytes quantization with a regular (non-AWQ) model. This is often more compatible and easier to set up.

### Step 1: Update config.json

Change your model from AWQ to regular:

```json
{
  "model_name": "Qwen/Qwen2.5-7B-Instruct",  // Remove -AWQ
  "use_4bit": true,  // This will use BitsAndBytes
  ...
}
```

### Step 2: Ensure BitsAndBytes is installed

```bash
uv pip install bitsandbytes
```

### Step 3: Run training

The training script will automatically use BitsAndBytes for 4-bit quantization.

## Why This Happens

- `autoawq` older versions (< 0.2.3) expect `PytorchGELUTanh` from transformers
- Newer `transformers` versions (4.36+) removed/renamed this class
- Latest `autoawq` (0.2.3+) has fixed this compatibility issue

## Verify Installation

After upgrading, verify it works:

```bash
python -c "from awq.models.auto import AutoAWQForCausalLM; print('AWQ OK')"
```

## Recommendation

**For production training**: Use **Solution 2 (BitsAndBytes)** - it's more stable and widely supported.

**For AWQ-specific features**: Use **Solution 1** - upgrade to latest autoawq.

