#!/usr/bin/env python3
"""
Script to combine all processed mental health datasets into one large training dataset.
"""

import argparse
from datasets import load_from_disk, concatenate_datasets, DatasetDict
from pathlib import Path
import os


def combine_all_datasets(dataset_dirs, output_dir, exclude_chinese=False):
    """Combine multiple datasets into one."""
    print("="*60)
    print("Combining Mental Health Datasets")
    print("="*60)
    
    datasets = []
    dataset_names = []
    
    # Define which datasets are Chinese (to optionally exclude)
    chinese_datasets = ['psydial_processed']
    
    for dataset_dir in dataset_dirs:
        if not os.path.exists(dataset_dir):
            print(f"Warning: {dataset_dir} not found, skipping...")
            continue
        
        # Skip Chinese datasets if requested
        if exclude_chinese and any(chinese in dataset_dir for chinese in chinese_datasets):
            print(f"Skipping {dataset_dir} (Chinese dataset)")
            continue
        
        print(f"\nLoading: {dataset_dir}")
        try:
            dataset = load_from_disk(dataset_dir)
            train_size = len(dataset['train'])
            val_size = len(dataset['validation'])
            print(f"  Train: {train_size:,}, Val: {val_size:,}")
            datasets.append(dataset)
            dataset_names.append(os.path.basename(dataset_dir))
        except Exception as e:
            print(f"  Error loading {dataset_dir}: {e}")
            continue
    
    if not datasets:
        print("No datasets loaded. Exiting.")
        return
    
    # Combine training sets
    print(f"\n{'='*60}")
    print("Combining training sets...")
    print(f"{'='*60}")
    combined_train = concatenate_datasets([ds["train"] for ds in datasets])
    print(f"Combined training samples: {len(combined_train):,}")
    
    # Combine validation sets
    print(f"\nCombining validation sets...")
    combined_val = concatenate_datasets([ds["validation"] for ds in datasets])
    print(f"Combined validation samples: {len(combined_val):,}")
    
    # Create combined dataset
    combined = DatasetDict({
        "train": combined_train,
        "validation": combined_val
    })
    
    # Save combined dataset
    print(f"\n{'='*60}")
    print(f"Saving combined dataset to {output_dir}...")
    print(f"{'='*60}")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    combined.save_to_disk(output_dir)
    
    print(f"\n{'='*60}")
    print("Combined dataset saved successfully!")
    print(f"{'='*60}")
    print(f"Total training samples: {len(combined_train):,}")
    print(f"Total validation samples: {len(combined_val):,}")
    print(f"\nDatasets included:")
    for name in dataset_names:
        print(f"  - {name}")
    print(f"\nOutput directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Combine all processed mental health datasets"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="all_mental_health_combined",
        help="Output directory for combined dataset"
    )
    parser.add_argument(
        "--exclude_chinese",
        action="store_true",
        help="Exclude Chinese datasets (e.g., PsyDial)"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=None,
        help="Specific datasets to combine (default: all found)"
    )
    
    args = parser.parse_args()
    
    # Default dataset directories
    if args.datasets:
        dataset_dirs = args.datasets
    else:
        dataset_dirs = [
            "counsel_chat_processed",
            "mentalchat16k_processed",
            "kaggle_mental_health_nguyen_processed_combined",
            "esconv_processed",
            "amod_processed",
            "psydial_processed",  # Chinese dataset - included by default
        ]
    
    combine_all_datasets(
        dataset_dirs=dataset_dirs,
        output_dir=args.output_dir,
        exclude_chinese=args.exclude_chinese
    )


if __name__ == "__main__":
    main()

