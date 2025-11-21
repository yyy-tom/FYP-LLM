#!/bin/bash
#
# Full Evaluation Pipeline for Mental Health Counseling Model
# Evaluates base and fine-tuned models on EmpatheticDialogues dataset
#
# Usage:
#   ./evaluation/scripts/run_full_evaluation.sh [fine-tuned-model-path] [num-samples]
#
# Example:
#   ./evaluation/scripts/run_full_evaluation.sh models/qwen2.5-counsel-chat-finetuned 100
#

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
FINETUNED_MODEL=${1:-"models/qwen2.5-counsel-chat-finetuned"}
NUM_SAMPLES=${2:-100}
BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"
DATASET_DIR="datasets/empathetic_dialogues_eval"

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  Mental Health Model Evaluation Pipeline ${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""
echo "Configuration:"
echo "  Base Model: $BASE_MODEL"
echo "  Fine-Tuned Model: $FINETUNED_MODEL"
echo "  Evaluation Samples: $NUM_SAMPLES"
echo "  Dataset: EmpatheticDialogues"
echo ""

# Step 1: Check if dataset exists, download if not
echo -e "${GREEN}[1/4] Checking dataset...${NC}"
if [ ! -d "$DATASET_DIR" ]; then
    echo "Dataset not found. Downloading EmpatheticDialogues..."
    uv run evaluation/scripts/download_eval_dataset.py
else
    echo "✓ Dataset already exists at $DATASET_DIR"
fi
echo ""

# Step 2: Evaluate base model
echo -e "${GREEN}[2/4] Evaluating base model...${NC}"
echo "This may take 10-20 minutes..."
uv run evaluation/scripts/evaluate_on_empathetic.py \
    --model "$BASE_MODEL" \
    --max_samples "$NUM_SAMPLES" \
    --output "results_base_$(date +%Y%m%d_%H%M%S).json"

BASE_RESULTS=$(ls -t results_base_*.json | head -1)
echo "✓ Base model evaluation complete: $BASE_RESULTS"
echo ""

# Step 3: Evaluate fine-tuned model
echo -e "${GREEN}[3/4] Evaluating fine-tuned model...${NC}"
echo "This may take 10-20 minutes..."
if [ ! -d "$FINETUNED_MODEL" ]; then
    echo -e "${YELLOW}Warning: Fine-tuned model not found at $FINETUNED_MODEL${NC}"
    echo "Skipping fine-tuned evaluation. Please train your model first."
    exit 1
fi

uv run evaluation/scripts/evaluate_on_empathetic.py \
    --model "$FINETUNED_MODEL" \
    --max_samples "$NUM_SAMPLES" \
    --output "results_finetuned_$(date +%Y%m%d_%H%M%S).json"

FINETUNED_RESULTS=$(ls -t results_finetuned_*.json | head -1)
echo "✓ Fine-tuned model evaluation complete: $FINETUNED_RESULTS"
echo ""

# Step 4: Compare results
echo -e "${GREEN}[4/4] Generating comparison report...${NC}"
uv run evaluation/scripts/compare_base_vs_finetuned.py \
    --base "$BASE_RESULTS" \
    --finetuned "$FINETUNED_RESULTS" \
    --output "comparison_report_$(date +%Y%m%d_%H%M%S).json" \
    --num_examples 5

COMPARISON_REPORT=$(ls -t comparison_report_*.json | head -1)
echo ""
echo -e "${BLUE}============================================${NC}"
echo -e "${GREEN}✓ Evaluation Complete!${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""
echo "Results saved to:"
echo "  • Base model: $BASE_RESULTS"
echo "  • Fine-tuned model: $FINETUNED_RESULTS"
echo "  • Comparison: $COMPARISON_REPORT"
echo ""
echo "Next steps:"
echo "  1. Review the comparison output above"
echo "  2. Check the JSON files for detailed metrics"
echo "  3. Use these results in your FYP presentation"
echo ""
echo -e "${YELLOW}Tip: See evaluation/docs/EVALUATION_EMPATHETIC_GUIDE.md for interpretation${NC}"

