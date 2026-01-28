#!/usr/bin/env python3
"""
Add crisis handling examples to training data.

This is the MOST IMPORTANT training improvement for a mental health LLM.
"""

import json
from pathlib import Path
import argparse


def main():
    parser = argparse.ArgumentParser(description="Add crisis examples to training data")
    parser.add_argument(
        "--base_dataset",
        type=str,
        default="datasets/counsel_chat/processed/train.json",
        help="Base training dataset"
    )
    parser.add_argument(
        "--crisis_examples",
        type=str,
        default="datasets/crisis_training_examples.json",
        help="Crisis training examples"
    )
    parser.add_argument(
        "--empathy_examples",
        type=str,
        default="datasets/empathy_examples.json",
        help="Empathy training examples (optional)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="datasets/counsel_chat/processed/train_enhanced.json",
        help="Output path for enhanced dataset"
    )
    parser.add_argument(
        "--crisis_weight",
        type=int,
        default=5,
        help="How many times to repeat crisis examples (default: 5)"
    )
    parser.add_argument(
        "--empathy_weight",
        type=int,
        default=2,
        help="How many times to repeat empathy examples (default: 2)"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("ADDING CRISIS & EMPATHY EXAMPLES TO TRAINING DATA")
    print("=" * 70)

    # Load base dataset
    print(f"\nLoading base dataset: {args.base_dataset}")
    base_path = Path(args.base_dataset)
    if not base_path.exists():
        print(f"❌ Error: Base dataset not found at {args.base_dataset}")
        print(f"   Available datasets:")
        datasets_dir = Path("datasets")
        if datasets_dir.exists():
            for dataset in datasets_dir.rglob("train.json"):
                print(f"   - {dataset}")
        return

    with open(base_path) as f:
        base_data = json.load(f)
    print(f"✓ Loaded {len(base_data)} base examples")

    # Load crisis examples
    print(f"\nLoading crisis examples: {args.crisis_examples}")
    crisis_path = Path(args.crisis_examples)
    if not crisis_path.exists():
        print(f"❌ Error: Crisis examples not found at {args.crisis_examples}")
        return

    with open(crisis_path) as f:
        crisis_data = json.load(f)
    print(f"✓ Loaded {len(crisis_data)} crisis examples")

    # Load empathy examples (optional)
    empathy_data = []
    empathy_path = Path(args.empathy_examples)
    if empathy_path.exists():
        print(f"\nLoading empathy examples: {args.empathy_examples}")
        with open(empathy_path) as f:
            empathy_data = json.load(f)
        print(f"✓ Loaded {len(empathy_data)} empathy examples")
    else:
        print(f"\nℹ️  No empathy examples found at {args.empathy_examples} (optional)")

    # Combine datasets
    print(f"\nCombining datasets:")
    print(f"  Base: {len(base_data)} examples (1x)")
    print(f"  Crisis: {len(crisis_data)} examples ({args.crisis_weight}x)")
    if empathy_data:
        print(f"  Empathy: {len(empathy_data)} examples ({args.empathy_weight}x)")

    combined = base_data + (crisis_data * args.crisis_weight)
    if empathy_data:
        combined += (empathy_data * args.empathy_weight)

    print(f"\n✓ Total: {len(combined)} examples")

    # Calculate percentages
    crisis_count = len(crisis_data) * args.crisis_weight
    empathy_count = len(empathy_data) * args.empathy_weight
    crisis_pct = (crisis_count / len(combined)) * 100
    empathy_pct = (empathy_count / len(combined)) * 100

    print(f"\nDataset composition:")
    print(f"  Base: {len(base_data)} ({100 - crisis_pct - empathy_pct:.1f}%)")
    print(f"  Crisis: {crisis_count} ({crisis_pct:.1f}%)")
    if empathy_data:
        print(f"  Empathy: {empathy_count} ({empathy_pct:.1f}%)")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving to: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(combined, f, indent=2)

    print(f"✓ Saved enhanced dataset")

    # Print next steps
    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("\n1. Train model with enhanced dataset:")
    print(f"   uv run python scripts/training/train_qwen_counsel.py \\")
    print(f"       --config configs/config_1.5b_fast.json \\")
    print(f"       --dataset {output_path} \\")
    print(f"       --output_dir models/qwen2.5-1.5b-crisis-aware")

    print("\n2. Test the trained model:")
    print(f"   uv run python scripts/inference/safe_inference.py \\")
    print(f"       --interactive \\")
    print(f"       --model_path models/qwen2.5-1.5b-crisis-aware")

    print("\n3. Test with crisis inputs:")
    print('   - "I\'m thinking about suicide"')
    print('   - "I want to end my life"')
    print('   - "Everyone would be better off without me"')

    print("\n" + "=" * 70)
    print("✅ DONE - Enhanced dataset ready for training!")
    print("=" * 70)


if __name__ == "__main__":
    main()
