# 💾 Disk Quota Exceeded - Fix Guide

## 🚨 Problem

Your training failed with:
```
RuntimeError: Disk quota exceeded (os error 122)
```

The HuggingFace cache is trying to download **Qwen2.5-1.5B-Instruct** (~3 GB) but you've exceeded your disk quota.

---

## 🔍 Step 1: Check Your Disk Usage

Run this script on your server:

```bash
bash scripts/setup/check_disk_space.sh
```

This will show:
- Your current disk quota/usage
- HuggingFace cache size
- Largest directories
- Large model files

---

## 🧹 Step 2: Free Up Space (Choose One)

### Option A: Clean XET Cache (Safest) ⭐ RECOMMENDED

XET cache is often large and safe to delete:

```bash
bash scripts/setup/clean_hf_cache.sh
# Choose option 1
```

**Space saved:** Usually 5-20 GB  
**Impact:** None - will re-download if needed

### Option B: Remove Old Models

If you have old model downloads you don't need:

```bash
# Manual cleanup
rm -rf /research/d7/fyp25/yyyu2/.cache/huggingface/hub/models--Qwen--Qwen2.5-*

# Or use the script
bash scripts/setup/clean_hf_cache.sh
# Choose option 2
```

**Check what models you have:**
```bash
ls -lh /research/d7/fyp25/yyyu2/.cache/huggingface/hub/
```

**Keep:** Current models you're using  
**Delete:** Old versions, unused models (Qwen2.5-7B, 14B if not needed)

### Option C: Remove Dataset Cache

If you have large cached datasets:

```bash
bash scripts/setup/clean_hf_cache.sh
# Choose option 3
```

### Option D: Nuclear Option (Last Resort)

Remove everything and start fresh:

```bash
bash scripts/setup/clean_hf_cache.sh
# Choose option 4 (type 'DELETE ALL')
```

**⚠️ Warning:** You'll need to re-download everything!

---

## 🗂️ Step 3: Move Cache to Larger Storage (Alternative)

If you have access to a larger disk/scratch space, move your cache there:

### Find Available Storage

```bash
# Check all mounted filesystems
df -h

# Look for locations with more space, e.g.:
# /scratch/
# /tmp/
# /project/
```

### Move Cache to New Location

```bash
# Example: Move to scratch space
NEW_CACHE="/scratch/yyyu2/.cache/huggingface"

# Create new directory
mkdir -p "$NEW_CACHE"

# Copy existing cache (if you want to keep it)
rsync -av /research/d7/fyp25/yyyu2/.cache/huggingface/ "$NEW_CACHE/"

# Or start fresh (skip the rsync)
```

### Update Job Script

Edit `scripts/jobs/train_1.5b_fast.sh` and change the cache directory:

```bash
# Find this section:
export HF_HOME="$BASE_DIR/.cache/huggingface"
export TRANSFORMERS_CACHE="$BASE_DIR/.cache/huggingface/transformers"
# ...

# Replace with:
export HF_HOME="/scratch/yyyu2/.cache/huggingface"
export TRANSFORMERS_CACHE="/scratch/yyyu2/.cache/huggingface/transformers"
export HF_DATASETS_CACHE="/scratch/yyyu2/.cache/huggingface/datasets"
export HF_HUB_CACHE="/scratch/yyyu2/.cache/huggingface/hub"
export XET_CACHE="/scratch/yyyu2/.cache/huggingface/xet"
```

---

## 📊 Model Size Reference

Space needed for Qwen models:

| Model | Download Size | Disk Space After | 4-bit Quantized |
|-------|--------------|------------------|-----------------|
| Qwen2.5-1.5B | ~3 GB | ~6 GB | ~1.5 GB in memory |
| Qwen2.5-3B | ~6 GB | ~12 GB | ~3 GB in memory |
| Qwen2.5-7B | ~14 GB | ~28 GB | ~7 GB in memory |
| Qwen2.5-14B | ~28 GB | ~56 GB | ~14 GB in memory |

**Your current model (1.5B):** Needs ~6 GB disk space

---

## ✅ Step 4: Verify You Have Space

After cleaning up:

```bash
# Check disk space
quota -s
# Or
df -h /research/d7/fyp25/yyyu2

# You need at least 10 GB free for comfortable operation
```

**Minimum requirements:**
- Model download: 3 GB
- Cache/temp files: 3 GB
- Working space: 2 GB
- **Total needed:** ~10 GB free

---

## 🚀 Step 5: Try Training Again

Once you have enough space:

```bash
sbatch scripts/jobs/train_1.5b_fast.sh
```

**Monitor download progress:**
```bash
tail -f logs/train_1.5b_fast_*.out
```

You should see:
```
Downloading model from Qwen/Qwen2.5-1.5B-Instruct...
Downloading: 100% [████████████████] 3.2GB/3.2GB
Model loaded successfully!
```

---

## 🛠️ Quick Fix Commands

### Fast Cleanup (Run on Server)

```bash
# 1. Check space
quota -s

# 2. Quick clean - remove XET cache
rm -rf /research/d7/fyp25/yyyu2/.cache/huggingface/xet

# 3. Check space again
quota -s

# 4. If still not enough, remove old models
# List models first
ls -lh /research/d7/fyp25/yyyu2/.cache/huggingface/hub/

# Remove specific old model (example)
rm -rf /research/d7/fyp25/yyyu2/.cache/huggingface/hub/models--Qwen--Qwen2.5-14B-Instruct
rm -rf /research/d7/fyp25/yyyu2/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct

# 5. Try training again
sbatch scripts/jobs/train_1.5b_fast.sh
```

---

## 📝 Prevention Tips

### 1. Regular Cleanup

Set up a monthly cleanup:

```bash
# Add to your crontab or run manually
bash scripts/setup/clean_hf_cache.sh
```

### 2. Monitor Disk Usage

Check regularly:

```bash
quota -s
du -sh ~/.cache/huggingface
```

### 3. Use Scratch Space

For temporary work, use scratch/tmp:

```bash
export HF_HOME="/scratch/yyyu2/.cache/huggingface"
```

### 4. Download Only What You Need

Don't keep multiple model sizes if you only use one:
- Using 1.5B? → Delete 7B, 14B downloads
- Using 7B? → Delete 1.5B, 3B, 14B downloads

---

## 🔍 Common Issues

### "XET cache is huge!"

XET is HuggingFace's caching system. Safe to delete:

```bash
rm -rf /research/d7/fyp25/yyyu2/.cache/huggingface/xet
```

### "Multiple model versions downloaded"

HuggingFace keeps old versions. Clean them:

```bash
# List all cached models
ls /research/d7/fyp25/yyyu2/.cache/huggingface/hub/

# Remove all Qwen models (careful!)
rm -rf /research/d7/fyp25/yyyu2/.cache/huggingface/hub/models--Qwen*

# They'll re-download when needed
```

### "Datasets taking up space"

Tokenized datasets can be large:

```bash
# Check dataset cache
du -sh /research/d7/fyp25/yyyu2/.cache/huggingface/datasets/

# Remove if needed
rm -rf /research/d7/fyp25/yyyu2/.cache/huggingface/datasets/*
```

---

## 📊 Expected Space Usage

After cleanup and with 1.5B model:

```
Total space needed:
├─ Qwen2.5-1.5B model: ~6 GB
├─ Dataset cache: ~2 GB
├─ Tokenized dataset: ~1 GB
├─ Working space: ~2 GB
└─ Buffer: ~2 GB
    └─ TOTAL: ~13 GB

Recommended free space: 15-20 GB
```

---

## ✅ Checklist

Before retrying training:

- [ ] Checked disk quota: `quota -s`
- [ ] Cleaned up cache: `bash scripts/setup/clean_hf_cache.sh`
- [ ] Have at least 10 GB free
- [ ] Model will download ~3 GB (verify space)
- [ ] Ready to retry: `sbatch scripts/jobs/train_1.5b_fast.sh`

---

## 🆘 Still Having Issues?

### If quota is too strict:

1. **Contact system admin** to increase quota
2. **Use scratch space** (usually larger quota)
3. **Use smaller model** - but 1.5B is already smallest!
4. **Work on different server** with more space

### If no other options:

Consider using a different storage location:
```bash
# Check available space on all mounts
df -h

# Find locations like:
# /scratch/, /tmp/, /project/, /data/
```

---

## 🎯 Summary

**Problem:** Disk quota exceeded  
**Cause:** HuggingFace cache full  
**Solution:** Clean cache or move to larger storage

**Quick fix:**
1. Run: `quota -s` (check space)
2. Run: `bash scripts/setup/clean_hf_cache.sh` (clean cache)
3. Choose option 1 (remove XET cache)
4. Run: `sbatch scripts/jobs/train_1.5b_fast.sh` (retry)

**You need:** ~10 GB free space minimum

---

Good luck! 🚀

