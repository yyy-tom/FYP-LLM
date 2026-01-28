#!/bin/bash
# Check disk space and HuggingFace cache usage

echo "=========================================="
echo "Disk Space Check"
echo "=========================================="
echo ""

# User's home directory
BASE_DIR="${HF_BASE_DIR:-/research/d7/fyp25/yyyu2}"

echo "1. Checking disk quota for $BASE_DIR"
echo "---"
quota -s 2>/dev/null || df -h "$BASE_DIR" | tail -1
echo ""

echo "2. Checking HuggingFace cache size"
echo "---"
HF_CACHE="$BASE_DIR/.cache/huggingface"
if [ -d "$HF_CACHE" ]; then
    echo "Cache directory: $HF_CACHE"
    du -sh "$HF_CACHE" 2>/dev/null || echo "Cannot calculate cache size"
    echo ""
    
    echo "Breakdown:"
    du -sh "$HF_CACHE"/* 2>/dev/null | sort -hr | head -10
else
    echo "Cache directory not found: $HF_CACHE"
fi
echo ""

echo "3. Largest directories in home"
echo "---"
du -sh "$BASE_DIR"/* 2>/dev/null | sort -hr | head -10
echo ""

echo "4. Checking for large model files"
echo "---"
find "$HF_CACHE" -type f -size +1G 2>/dev/null | while read file; do
    size=$(du -h "$file" | cut -f1)
    echo "$size - $file"
done | head -10
echo ""

echo "=========================================="
echo "Summary"
echo "=========================================="
quota -s 2>/dev/null || df -h "$BASE_DIR" | tail -1





