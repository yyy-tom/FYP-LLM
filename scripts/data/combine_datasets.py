#!/usr/bin/env python3
"""
Script to combine multiple processed datasets into one.
"""

import argparse
import os
from datasets import load_from_disk, concatenate_datasets, DatasetDict
from pathlib import Path


def combine_datasets(input_dirs, output_dir):
    """Combine multiple datasets into one."""
    print(f"Loading datasets from: {input_dirs}")
    
    datasets = []
    for input_dir in input_dirs:
        # Check if path exists
        if not os.path.exists(input_dir):
            print(f"Warning: {input_dir} not found, skipping...")
            continue
        
        print(f"\nLoading: {input_dir}")
        try:
        dataset = load_from_disk(input_dir)
            train_size = len(dataset['train'])
            val_size = len(dataset['validation'])
            print(f"  Train: {train_size:,}, Val: {val_size:,}")
        datasets.append(dataset)
        except Exception as e:
            print(f"  Error loading {input_dir}: {e}")
            continue
    
    if not datasets:
        print("No datasets loaded. Exiting.")
        return
    
    # Combine training sets
    print("\nCombining training sets...")
    combined_train = concatenate_datasets([ds["train"] for ds in datasets])
    print(f"Combined training samples: {len(combined_train):,}")
    
    # Combine validation sets
    print("Combining validation sets...")
    combined_val = concatenate_datasets([ds["validation"] for ds in datasets])
    print(f"Combined validation samples: {len(combined_val):,}")
    
    # Create combined dataset
    combined = DatasetDict({
        "train": combined_train,
        "validation": combined_val
    })
    
    # Save combined dataset
    print(f"\nSaving combined dataset to {output_dir}...")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    combined.save_to_disk(output_dir)
    
    print(f"\nCombined dataset saved successfully!")
    print(f"Total training samples: {len(combined_train):,}")
    print(f"Total validation samples: {len(combined_val):,}")


def main():
    parser = argparse.ArgumentParser(description="Combine multiple processed datasets")
    parser.add_argument(
        "--input_dirs",
        type=str,
        nargs="+",
        required=True,
        help="List of input dataset directories to combine"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for combined dataset"
    )
    
    args = parser.parse_args()
    
    combine_datasets(args.input_dirs, args.output_dir)


if __name__ == "__main__":
    main()

