#!/usr/bin/env python3
"""
Dataset preparation script for MentalChat16K dataset.
This script loads the dataset from Hugging Face and formats it for instruction tuning with Qwen2.5.
Dataset: https://huggingface.co/datasets/ShenLab/MentalChat16K
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
    
    # Convert to string
    text = str(text)
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove extra quotes and escape characters
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")
    text = text.replace('&#34;', '"').replace('&#39;', "'")
    
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


def process_mentalchat16k_data(
    dataset_name: str = "ShenLab/MentalChat16K",
    output_path: str = "mentalchat16k_processed",
    max_samples: Optional[int] = None,
    cache_dir: Optional[str] = None,
    split: str = "train"
) -> None:
    """Process MentalChat16K dataset and create training dataset."""
    
    print(f"Loading dataset from Hugging Face: {dataset_name}")
    print(f"This may take a few minutes for the first download...")
    
    try:
        # Load dataset from Hugging Face
        dataset = load_dataset(
            dataset_name,
            split=split,
            cache_dir=cache_dir
        )
        print(f"Loaded {len(dataset)} samples")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you have internet connection")
        print("2. Check that the dataset name is correct")
        print("3. Try accessing the dataset page: https://huggingface.co/datasets/ShenLab/MentalChat16K")
        raise
    
    # Inspect the dataset structure
    print(f"\nDataset features: {dataset.features}")
    if len(dataset) > 0:
        print(f"\nSample entry:")
        sample = dataset[0]
        for key, value in sample.items():
            if isinstance(value, str) and len(value) > 100:
                print(f"  {key}: {value[:100]}...")
            else:
                print(f"  {key}: {value}")
    
    # Detect column names
    columns = dataset.column_names
    print(f"\nAvailable columns: {columns}")
    
    # Common column name patterns for MentalChat16K
    question_col = None
    answer_col = None
    topic_col = None
    
    # Try to find question/answer columns
    # For MentalChat16K, the structure is: instruction (system prompt), input (question), output (answer)
    question_patterns = ['input', 'question', 'user', 'query', 'prompt']
    answer_patterns = ['output', 'answer', 'response', 'assistant', 'reply']
    topic_patterns = ['topic', 'category', 'label', 'type', 'domain']
    
    for col in columns:
        col_lower = col.lower()
        if not question_col:
            for pattern in question_patterns:
                if pattern in col_lower:
                    question_col = col
                    break
        if not answer_col:
            for pattern in answer_patterns:
                if pattern in col_lower:
                    answer_col = col
                    break
        if not topic_col:
            for pattern in topic_patterns:
                if pattern in col_lower:
                    topic_col = col
                    break
    
    # If not found, try common names (prioritize 'input' for MentalChat16K)
    if not question_col:
        if 'input' in columns:
            question_col = 'input'
        elif 'question' in columns:
            question_col = 'question'
        elif len(columns) >= 2:
            question_col = columns[1]  # Usually input is second column
    
    if not answer_col:
        if 'output' in columns:
            answer_col = 'output'
        elif 'answer' in columns:
            answer_col = 'answer'
        elif 'response' in columns:
            answer_col = 'response'
        elif len(columns) >= 3:
            answer_col = columns[2]  # Usually output is third column
    
    print(f"\nDetected column mapping:")
    print(f"  Question: {question_col}")
    print(f"  Answer: {answer_col}")
    print(f"  Topic: {topic_col}")
    
    if not question_col or not answer_col:
        raise ValueError(f"Could not detect question/answer columns. Available columns: {columns}")
    
    # Process dataset
    print(f"\nProcessing dataset...")
    training_examples = []
    
    # Limit samples if specified
    dataset_to_process = dataset
    if max_samples and len(dataset) > max_samples:
        dataset_to_process = dataset.select(range(max_samples))
        print(f"Limited to {max_samples} samples")
    
    for idx, example in enumerate(dataset_to_process):
        # Extract question and answer
        question = example.get(question_col, "")
        answer = example.get(answer_col, "")
        
        # Get topic if available
        topic = ""
        if topic_col and topic_col in example:
            topic_val = example.get(topic_col, "")
            topic = clean_text(str(topic_val)) if topic_val else ""
        
        # Clean text
        question = clean_text(str(question)) if question else ""
        answer = clean_text(str(answer)) if answer else ""
        
        # Skip if empty
        if not question or not answer:
            continue
        
        # Filter by length
        if len(answer) < 50 or len(answer) > 2000:
            continue
        if len(question) < 20:
            continue
        
        # Create instruction prompt
        instruction = create_instruction_prompt(question, topic if topic else "")
        
        training_example = {
            "instruction": instruction,
            "input": "",
            "output": answer,
            "topic": topic,
            "upvotes": 0,  # MentalChat16K may not have upvotes
            "question_id": str(idx),
        }
        
        training_examples.append(training_example)
        
        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1}/{len(dataset_to_process)} samples...")
    
    if not training_examples:
        raise ValueError("No valid training examples created from the dataset")
    
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
    
    # Save the dataset
    print(f"\nSaving dataset to {output_path}...")
    Path(output_path).mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(output_path)
    
    print(f"\nDataset saved successfully!")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Save a sample for inspection
    sample_path = Path("samples") / "mentalchat16k_sample_data.json"
    with open(sample_path, 'w', encoding='utf-8') as f:
        json.dump(train_examples[:3], f, indent=2, ensure_ascii=False)
    
    print(f"Sample data saved to {sample_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare MentalChat16K dataset for training"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="ShenLab/MentalChat16K",
        help="Hugging Face dataset name"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="mentalchat16k_processed",
        help="Output directory for processed dataset"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (for testing)"
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
        help="Dataset split to load (default: train)"
    )
    
    args = parser.parse_args()
    
    try:
        process_mentalchat16k_data(
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

