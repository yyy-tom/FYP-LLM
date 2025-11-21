#!/usr/bin/env python3
"""
Script to combine all processed mental health datasets into one large training dataset.

Usage:
    # Combine all datasets (English + Chinese)
    python scripts/data/combine_all_datasets.py
    
    # Combine only English datasets (exclude Chinese)
    python scripts/data/combine_all_datasets.py --exclude_chinese
    
    # Combine specific datasets
    python scripts/data/combine_all_datasets.py --datasets datasets/counsel_chat_processed datasets/esconv_processed
    
    # Custom output directory
    python scripts/data/combine_all_datasets.py --output_dir datasets/my_combined_dataset

Note: Run this script from the project root directory.
"""

import argparse
from datasets import load_from_disk, concatenate_datasets, DatasetDict
from pathlib import Path
import os
import sys

# Get the project root directory (parent of scripts/)
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent


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
        # Resolve relative paths relative to project root
        if not os.path.isabs(dataset_dir):
            dataset_dir = os.path.join(PROJECT_ROOT, dataset_dir)
        
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
            
            # Check if dataset has required splits
            if 'train' not in dataset:
                print(f"  ✗ Error: {dataset_dir} does not have a 'train' split, skipping...")
                continue
            
            train_size = len(dataset['train'])
            
            # Check if validation exists
            if 'validation' in dataset and len(dataset['validation']) > 0:
                val_size = len(dataset['validation'])
                print(f"  ✓ Train: {train_size:,}, Val: {val_size:,}")
            else:
                val_size = 0
                if 'validation' not in dataset:
                    print(f"  ⚠ Train: {train_size:,}, Val: missing (will be excluded from validation combination)")
                else:
                    print(f"  ⚠ Train: {train_size:,}, Val: empty (will be excluded from validation combination)")
            
            # Normalize question_id to string type for all datasets
            # This ensures compatibility when concatenating
            if 'question_id' in dataset['train'].features:
                from datasets import Features, Value
                # Always cast question_id to string for consistency
                def cast_question_id(example):
                    example['question_id'] = str(example.get('question_id', ''))
                    return example
                dataset = dataset.map(cast_question_id, desc=f"Normalizing question_id for {os.path.basename(dataset_dir)}")
                # Update features to reflect string type
                new_features = dataset['train'].features.copy()
                new_features['question_id'] = Value('string')
                dataset = dataset.cast(new_features)
            
            datasets.append(dataset)
            dataset_names.append(os.path.basename(dataset_dir))
        except FileNotFoundError as e:
            print(f"  ✗ Error: Dataset directory incomplete or corrupted: {os.path.basename(dataset_dir)}")
            print(f"    (This dataset will be skipped)")
            continue
        except Exception as e:
            print(f"  ✗ Error loading {dataset_dir}: {e}")
            print(f"    (This dataset will be skipped)")
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
    
    # Combine validation sets (only include datasets that have validation data)
    print(f"\nCombining validation sets...")
    validation_datasets = []
    for ds in datasets:
        if "validation" in ds:
            val_split = ds["validation"]
            if len(val_split) > 0:
                validation_datasets.append(val_split)
    if validation_datasets:
        combined_val = concatenate_datasets(validation_datasets)
        print(f"Combined validation samples: {len(combined_val):,}")
        print(f"  (Excluded {len(datasets) - len(validation_datasets)} dataset(s) without validation data)")
    else:
        print("⚠ Warning: No validation data found in any dataset!")
        print("  Creating empty validation set with same schema as training set...")
        # Create minimal validation set with same columns as train
        from datasets import Dataset
        if len(combined_train) > 0:
            # Get column names from first dataset
            sample_cols = combined_train.column_names
            combined_val = Dataset.from_dict({col: [] for col in sample_cols})
        else:
            combined_val = Dataset.from_dict({'text': []})
    
    # Create combined dataset
    combined = DatasetDict({
        "train": combined_train,
        "validation": combined_val
    })
    
    # Save combined dataset (resolve output path relative to project root)
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(PROJECT_ROOT, output_dir)
    
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
        description="Combine all processed mental health datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Combine all datasets (including Chinese)
  python scripts/combine_all_datasets.py
  
  # Combine only English datasets
  python scripts/combine_all_datasets.py --exclude_chinese
  
  # Combine specific datasets
  python scripts/combine_all_datasets.py --datasets datasets/counsel_chat_processed datasets/esconv_processed
  
  # Custom output location
  python scripts/combine_all_datasets.py --output_dir datasets/custom_combined
        """
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="datasets/all_mental_health_combined",
        help="Output directory for combined dataset (default: datasets/all_mental_health_combined)"
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
    
    # Default dataset directories (all in datasets/ directory)
    if args.datasets:
        # If user provides datasets, check if they're already in datasets/ or add prefix
        dataset_dirs = []
        for ds in args.datasets:
            # Try absolute path first
            if os.path.isabs(ds) and os.path.exists(ds):
                dataset_dirs.append(ds)
            # Try relative to project root
            elif os.path.exists(os.path.join(PROJECT_ROOT, ds)):
                dataset_dirs.append(ds)
            # Try with datasets/ prefix
            elif os.path.exists(os.path.join(PROJECT_ROOT, "datasets", ds)):
                dataset_dirs.append(f"datasets/{ds}")
            else:
                print(f"Warning: Dataset not found: {ds}")
                dataset_dirs.append(ds)  # Try anyway, will fail gracefully
    else:
        # Default dataset directories (all in datasets/ directory relative to project root)
        # Note: Using kaggle_mental_health_nguyen_processed_combined (already combined)
        dataset_dirs = [
            "datasets/counsel_chat_processed",
            "datasets/mentalchat16k_processed",
            "datasets/kaggle_mental_health_nguyen_processed_combined",
            "datasets/esconv_processed",
            "datasets/amod_processed",
            "datasets/psydial_processed",  # Chinese dataset - included by default
            "datasets/cactus_processed",  # Optional: Very large dataset
        ]
    
    combine_all_datasets(
        dataset_dirs=dataset_dirs,
        output_dir=args.output_dir,
        exclude_chinese=args.exclude_chinese
    )


if __name__ == "__main__":
    main()

