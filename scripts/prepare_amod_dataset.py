#!/usr/bin/env python3
"""
Dataset preparation script for Amod Mental Health Counseling Conversations dataset.
This script loads the dataset from Hugging Face and formats it for instruction tuning.
Dataset: https://huggingface.co/datasets/Amod/mental_health_counseling_conversations
"""

import json
import re
from typing import List, Dict, Any, Optional
from datasets import Dataset, DatasetDict, load_dataset
import argparse
from pathlib import Path


def clean_text(text: str) -> str:
    """Clean and normalize text content."""
    if text is None:
        return ""
    
    text = str(text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def create_instruction_prompt(question: str, context: str = "") -> str:
    """Create an instruction prompt for the counseling question."""
    if context:
        return f"""You are a compassionate and professional mental health counselor. Please provide helpful, empathetic, and evidence-based advice for the following situation.

Context: {context}

Question: {question}

Please provide a thoughtful and supportive response that:
1. Acknowledges the person's feelings
2. Offers practical advice
3. Suggests professional resources if appropriate
4. Maintains a warm, non-judgmental tone

Response:"""
    else:
        return f"""You are a compassionate and professional mental health counselor. Please provide helpful, empathetic, and evidence-based advice for the following question.

Question: {question}

Please provide a thoughtful and supportive response that:
1. Acknowledges the person's feelings
2. Offers practical advice
3. Suggests professional resources if appropriate
4. Maintains a warm, non-judgmental tone

Response:"""


def process_amod_data(
    dataset_name: str = "Amod/mental_health_counseling_conversations",
    output_path: str = "amod_processed",
    max_samples: Optional[int] = None,
    cache_dir: Optional[str] = None,
    split: str = "train"
) -> None:
    """Process Amod Mental Health Counseling dataset and create training dataset."""
    
    print(f"Loading dataset from Hugging Face: {dataset_name}")
    
    try:
        dataset = load_dataset(dataset_name, split=split, cache_dir=cache_dir)
        print(f"Loaded {len(dataset)} samples")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise
    
    # Inspect structure
    print(f"\nDataset features: {dataset.features}")
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\nSample entry:")
        for key, value in sample.items():
            if isinstance(value, str) and len(value) > 100:
                print(f"  {key}: {value[:100]}...")
            else:
                print(f"  {key}: {value}")
    
    # Process dataset
    print(f"\nProcessing dataset...")
    training_examples = []
    
    dataset_to_process = dataset
    if max_samples and len(dataset) > max_samples:
        dataset_to_process = dataset.select(range(max_samples))
        print(f"Limited to {max_samples} samples")
    
    for idx, example in enumerate(dataset_to_process):
        # Extract context and response
        context = example.get('Context', '') or example.get('context', '') or example.get('question', '')
        response = example.get('Response', '') or example.get('response', '') or example.get('answer', '')
        
        # Clean text
        context = clean_text(str(context)) if context else ""
        response = clean_text(str(response)) if response else ""
        
        # Skip if empty
        if not context or not response:
            continue
        
        # Filter by length
        if len(response) < 20 or len(response) > 2000:
            continue
        if len(context) < 10:
            continue
        
        # Create instruction prompt
        instruction = create_instruction_prompt(context, "")
        
        training_example = {
            "instruction": instruction,
            "input": "",
            "output": response,
            "topic": "",
            "upvotes": 0,
            "question_id": str(idx),
        }
        
        training_examples.append(training_example)
        
        if (idx + 1) % 500 == 0:
            print(f"  Processed {idx + 1}/{len(dataset_to_process)} samples...")
    
    if not training_examples:
        raise ValueError("No valid training examples created")
    
    print(f"\nCreated {len(training_examples)} valid training examples")
    
    # Split into train/validation (90/10)
    train_size = int(0.9 * len(training_examples))
    train_examples = training_examples[:train_size]
    val_examples = training_examples[train_size:]
    
    # Create datasets
    train_dataset = Dataset.from_list(train_examples)
    val_dataset = Dataset.from_list(val_examples)
    
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset
    })
    
    # Save
    print(f"\nSaving dataset to {output_path}...")
    Path(output_path).mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(output_path)
    
    print(f"\nDataset saved successfully!")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Save sample
    sample_path = Path("samples") / "amod_sample_data.json"
    with open(sample_path, 'w', encoding='utf-8') as f:
        json.dump(train_examples[:3], f, indent=2, ensure_ascii=False)
    
    print(f"Sample data saved to {sample_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare Amod Mental Health Counseling dataset for training")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="Amod/mental_health_counseling_conversations",
        help="Hugging Face dataset name"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="amod_processed",
        help="Output directory for processed dataset"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Cache directory for Hugging Face datasets"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to load"
    )
    
    args = parser.parse_args()
    
    try:
        process_amod_data(
            dataset_name=args.dataset_name,
            output_path=args.output_dir,
            max_samples=args.max_samples,
            cache_dir=args.cache_dir,
            split=args.split
        )
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

