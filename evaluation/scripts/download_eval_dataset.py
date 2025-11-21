#!/usr/bin/env python3
"""
Download and prepare evaluation datasets for mental health counseling model evaluation.

Supports multiple datasets:
1. DailyDialog - General conversations (widely available, reliable)
2. PersonaChat - Conversational dataset (good for dialogue quality)
3. BlendedSkillTalk - Multi-skill conversations
4. Your existing validation split (RECOMMENDED - already available)
"""

import argparse
from datasets import load_dataset
from pathlib import Path


def download_dailydialog(output_dir="datasets/dailydialog_eval"):
    """
    Download DailyDialog dataset - general conversations, widely available.
    
    Dataset info:
    - 13,118 multi-turn dialogues
    - 10 topics (ordinary life, school life, culture, etc.)
    - Clean, high-quality conversations
    """
    print("=" * 80)
    print("Downloading DailyDialog Dataset")
    print("=" * 80)
    
    # Try multiple methods
    dataset = None
    methods = [
        ("daily_dialog", "trust_remote_code"),
        ("daily_dialog", "standard"),
        ("parquet", "from_hub"),  # Try loading from parquet files
    ]
    
    for method_name, method_type in methods:
        try:
            print(f"\n1. Trying method: {method_type}...")
            if method_type == "trust_remote_code":
                dataset = load_dataset(method_name, trust_remote_code=True)
            elif method_type == "standard":
                dataset = load_dataset(method_name)
            elif method_type == "from_hub":
                # Try loading from parquet files on HuggingFace Hub
                dataset = load_dataset("parquet", data_files={
                    "train": "https://huggingface.co/datasets/daily_dialog/resolve/main/train-*.parquet",
                    "validation": "https://huggingface.co/datasets/daily_dialog/resolve/main/validation-*.parquet",
                    "test": "https://huggingface.co/datasets/daily_dialog/resolve/main/test-*.parquet"
                })
            print(f"   ✓ Successfully loaded using {method_type}")
            break
        except Exception as e:
            print(f"   ✗ Failed: {str(e)[:100]}")
            continue
    
    if dataset is None:
        raise RuntimeError("Could not download DailyDialog. All methods failed.")
    
    try:
        print(f"\n2. Dataset loaded successfully!")
        print(f"   - Train: {len(dataset['train'])} examples")
        if 'validation' in dataset:
            print(f"   - Validation: {len(dataset['validation'])} examples")
        if 'test' in dataset:
            print(f"   - Test: {len(dataset['test'])} examples")
        
        # Show sample
        print("\n3. Sample conversation:")
        test_key = 'test' if 'test' in dataset else ('validation' if 'validation' in dataset else 'train')
        sample = dataset[test_key][0]
        print(f"   Keys: {list(sample.keys())}")
        
        # Save dataset
        print(f"\n4. Saving dataset to {output_dir}...")
        Path(output_dir).parent.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(output_dir)
        
        print("\n✅ Dataset downloaded and saved successfully!")
        print(f"   Location: {output_dir}")
        
        return dataset
        
    except Exception as e:
        print(f"\n❌ Error processing DailyDialog: {e}")
        raise


def download_personachat(output_dir="datasets/personachat_eval"):
    """
    Download PersonaChat dataset - conversational dataset with personas.
    
    Dataset info:
    - 164,356 utterances
    - Conversations with consistent personas
    - Good for testing conversational quality
    """
    print("=" * 80)
    print("Downloading PersonaChat Dataset")
    print("=" * 80)
    
    try:
        print("\n1. Loading dataset from HuggingFace...")
        dataset = load_dataset("bavard/personachat_truecased")
        
        print(f"\n2. Dataset loaded successfully!")
        print(f"   - Train: {len(dataset['train'])} examples")
        if 'validation' in dataset:
            print(f"   - Validation: {len(dataset['validation'])} examples")
        if 'test' in dataset:
            print(f"   - Test: {len(dataset['test'])} examples")
        
        # Save dataset
        print(f"\n3. Saving dataset to {output_dir}...")
        Path(output_dir).parent.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(output_dir)
        
        print("\n✅ Dataset downloaded and saved successfully!")
        print(f"   Location: {output_dir}")
        
        return dataset
        
    except Exception as e:
        print(f"\n❌ Error downloading PersonaChat: {e}")
        raise


def download_blended_skill_talk(output_dir="datasets/blended_skill_talk_eval"):
    """
    Download BlendedSkillTalk dataset - multi-skill conversations.
    
    Dataset info:
    - Conversations combining persona, knowledge, and empathy
    - Good for testing multi-faceted conversational abilities
    """
    print("=" * 80)
    print("Downloading BlendedSkillTalk Dataset")
    print("=" * 80)
    
    try:
        print("\n1. Loading dataset from HuggingFace...")
        dataset = load_dataset("blended_skill_talk")
        
        print(f"\n2. Dataset loaded successfully!")
        print(f"   - Train: {len(dataset['train'])} examples")
        if 'validation' in dataset:
            print(f"   - Validation: {len(dataset['validation'])} examples")
        if 'test' in dataset:
            print(f"   - Test: {len(dataset['test'])} examples")
        
        # Save dataset
        print(f"\n3. Saving dataset to {output_dir}...")
        Path(output_dir).parent.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(output_dir)
        
        print("\n✅ Dataset downloaded and saved successfully!")
        print(f"   Location: {output_dir}")
        
        return dataset
        
    except Exception as e:
        print(f"\n❌ Error downloading BlendedSkillTalk: {e}")
        raise


def download_smile(output_dir="datasets/smile_eval"):
    """
    Download SMILE dataset - mental health specific conversations.
    
    Dataset info:
    - Mental health counseling conversations
    - High relevance for counseling evaluation
    """
    print("=" * 80)
    print("Downloading SMILE Dataset")
    print("=" * 80)
    
    try:
        print("\n1. Loading dataset from HuggingFace...")
        # Try different possible names
        dataset = None
        for name in ["qiuhuachuan/SMILE", "SMILE"]:
            try:
                dataset = load_dataset(name)
                print(f"   ✓ Loaded from {name}")
                break
            except:
                continue
        
        if dataset is None:
            raise RuntimeError("Could not find SMILE dataset on HuggingFace")
        
        print(f"\n2. Dataset loaded successfully!")
        print(f"   - Train: {len(dataset['train'])} examples")
        if 'validation' in dataset:
            print(f"   - Validation: {len(dataset['validation'])} examples")
        if 'test' in dataset:
            print(f"   - Test: {len(dataset['test'])} examples")
        
        # Save dataset
        print(f"\n3. Saving dataset to {output_dir}...")
        Path(output_dir).parent.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(output_dir)
        
        print("\n✅ Dataset downloaded and saved successfully!")
        print(f"   Location: {output_dir}")
        
        return dataset
        
    except Exception as e:
        print(f"\n❌ Error downloading SMILE: {e}")
        raise


def list_available_datasets():
    """List all available evaluation datasets."""
    print("=" * 80)
    print("Available Evaluation Datasets")
    print("=" * 80)
    print("\n⚠️  NOTE: Many HuggingFace datasets use deprecated script formats.")
    print("   The datasets library no longer supports these.")
    print("\n" + "="*80)
    print("✅ BEST OPTION: Use Your Existing Validation Split")
    print("="*80)
    print("   - Already downloaded: datasets/all_mental_health_combined")
    print("   - Matches your training data format")
    print("   - No download needed!")
    print("   - No compatibility issues!")
    print("   - Command:")
    print("     uv run evaluation/scripts/evaluate_model.py \\")
    print("         --model_path models/qwen2.5-counsel-chat-finetuned \\")
    print("         --test_dataset datasets/all_mental_health_combined")
    print("\n" + "="*80)
    print("Alternative Datasets (May Have Download Issues)")
    print("="*80)
    print("\n1. DailyDialog")
    print("   - General conversations")
    print("   - 13K+ dialogues, 10 topics")
    print("   - Command: --dataset dailydialog")
    print("   - ⚠️  May fail due to deprecated format")
    print("\n2. PersonaChat")
    print("   - Conversational dataset with personas")
    print("   - 164K+ utterances")
    print("   - Command: --dataset personachat")
    print("   - ⚠️  May fail due to deprecated format")
    print("\n3. BlendedSkillTalk")
    print("   - Multi-skill conversations")
    print("   - Command: --dataset blended_skill_talk")
    print("   - ⚠️  May fail due to deprecated format")
    print("\n4. SMILE (Mental Health Specific)")
    print("   - Mental health counseling conversations")
    print("   - Command: --dataset smile")
    print("   - ⚠️  May fail due to deprecated format")
    print("\n" + "="*80)
    print("RECOMMENDATION: Use your validation split - it's the best option!")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Download evaluation datasets for mental health counseling model"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["dailydialog", "personachat", "blended_skill_talk", "smile", "list"],
        default="dailydialog",
        help="Dataset to download (default: dailydialog). Use 'list' to see all options."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory to save the dataset (default: datasets/{dataset_name}_eval)"
    )
    
    args = parser.parse_args()
    
    # List available datasets
    if args.dataset == "list":
        list_available_datasets()
        return
    
    # Set default output directory
    if args.output_dir is None:
        args.output_dir = f"datasets/{args.dataset}_eval"
    
    # Download selected dataset
    dataset = None
    try:
        if args.dataset == "dailydialog":
            dataset = download_dailydialog(args.output_dir)
        elif args.dataset == "personachat":
            dataset = download_personachat(args.output_dir)
        elif args.dataset == "blended_skill_talk":
            dataset = download_blended_skill_talk(args.output_dir)
        elif args.dataset == "smile":
            dataset = download_smile(args.output_dir)
    except Exception as e:
        print("\n" + "="*80)
        print("DOWNLOAD FAILED - This is Expected!")
        print("="*80)
        print("\nMost HuggingFace datasets use deprecated script formats that")
        print("are no longer supported by newer versions of the datasets library.")
        print("\n" + "="*80)
        print("✅ RECOMMENDED SOLUTION: Use Your Validation Split")
        print("="*80)
        print("\nThis is actually the BEST option for evaluation:")
        print("\n  uv run evaluation/scripts/evaluate_model.py \\")
        print("      --model_path models/qwen2.5-counsel-chat-finetuned \\")
        print("      --base_model Qwen/Qwen2.5-7B-Instruct \\")
        print("      --test_dataset datasets/all_mental_health_combined \\")
        print("      --output evaluation_results.json \\")
        print("      --max_samples 100")
        print("\nWhy this is better:")
        print("  ✓ Already downloaded and processed")
        print("  ✓ Matches your training data format")
        print("  ✓ No compatibility issues")
        print("  ✓ Tests on similar data distribution")
        print("  ✓ Ready to use immediately")
        print("  ✓ Appropriate for FYP evaluation")
        print("\n" + "="*80)
        print("Alternative: Install Older Datasets Version")
        print("="*80)
        print("\nIf you really need external datasets, downgrade:")
        print("  pip install 'datasets==2.14.0'")
        print("\nBut we strongly recommend using your validation split instead!")
        print("="*80)
        raise
    
    # Print next steps
    print("\n" + "=" * 80)
    print("Next Steps:")
    print("=" * 80)
    print(f"\n1. Evaluate your model on this dataset:")
    print(f"   uv run evaluation/scripts/evaluate_model.py \\")
    print(f"       --model_path models/qwen2.5-counsel-chat-finetuned \\")
    print(f"       --base_model Qwen/Qwen2.5-7B-Instruct \\")
    print(f"       --test_dataset {args.output_dir} \\")
    print(f"       --output evaluation_results.json")
    print()
    print(f"\n2. Or use the EmpatheticDialogues evaluation script:")
    print(f"   uv run evaluation/scripts/evaluate_on_empathetic.py \\")
    print(f"       --model models/qwen2.5-counsel-chat-finetuned \\")
    print(f"       --dataset {args.output_dir} \\")
    print(f"       --output results.json")
    print()


if __name__ == "__main__":
    main()
