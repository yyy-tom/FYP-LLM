#!/bin/bash
# Clean up HuggingFace cache to free disk space

set -e

BASE_DIR="${HF_BASE_DIR:-/research/d7/fyp25/yyyu2}"
HF_CACHE="$BASE_DIR/.cache/huggingface"

echo "=========================================="
echo "HuggingFace Cache Cleanup"
echo "=========================================="
echo ""

if [ ! -d "$HF_CACHE" ]; then
    echo "Cache directory not found: $HF_CACHE"
    exit 1
fi

echo "Current cache size:"
du -sh "$HF_CACHE"
echo ""

# Function to safely remove a directory
safe_remove() {
    local dir="$1"
    local name="$2"
    
    if [ -d "$dir" ]; then
        size=$(du -sh "$dir" | cut -f1)
        echo "Removing $name ($size)..."
        rm -rf "$dir"
        echo "✓ Removed"
    else
        echo "⊘ $name not found, skipping"
    fi
}

echo "What would you like to clean?"
echo ""
echo "1. Remove XET cache (usually safe to delete)"
echo "2. Remove old model downloads (transformers cache)"
echo "3. Remove dataset cache"
echo "4. Remove ALL caches (nuclear option)"
echo "5. Show detailed breakdown only"
echo ""
read -p "Enter choice (1-5): " choice

case $choice in
    1)
        echo ""
        echo "Removing XET cache..."
        safe_remove "$BASE_DIR/.cache/huggingface/xet" "XET cache"
        ;;
    2)
        echo ""
        echo "WARNING: This will remove all downloaded models!"
        echo "You'll need to re-download them for training."
        read -p "Are you sure? (yes/no): " confirm
        if [ "$confirm" == "yes" ]; then
            safe_remove "$BASE_DIR/.cache/huggingface/transformers" "Transformers cache"
            safe_remove "$BASE_DIR/.cache/huggingface/hub" "Hub cache"
        else
            echo "Cancelled."
        fi
        ;;
    3)
        echo ""
        echo "Removing dataset cache..."
        safe_remove "$BASE_DIR/.cache/huggingface/datasets" "Datasets cache"
        ;;
    4)
        echo ""
        echo "⚠️  NUCLEAR OPTION: This will remove EVERYTHING!"
        echo "You'll need to re-download all models and datasets."
        read -p "Are you ABSOLUTELY sure? (type 'DELETE ALL'): " confirm
        if [ "$confirm" == "DELETE ALL" ]; then
            safe_remove "$HF_CACHE" "Entire HuggingFace cache"
            mkdir -p "$HF_CACHE"
            echo "✓ Cache directory recreated"
        else
            echo "Cancelled (smart choice!)."
        fi
        ;;
    5)
        echo ""
        echo "Detailed breakdown:"
        du -sh "$HF_CACHE"/* 2>/dev/null | sort -hr
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "New cache size:"
du -sh "$HF_CACHE" 2>/dev/null || echo "0B (cache removed)"
echo ""
echo "Disk space:"
quota -s 2>/dev/null || df -h "$BASE_DIR" | tail -1
echo ""
echo "Done!"





