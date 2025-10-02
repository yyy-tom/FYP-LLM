#!/usr/bin/env python3
"""
Dataset preparation script for Counsel Chat dataset to be used with Qwen2.5 training.
This script processes the Counsel Chat CSV data and formats it for instruction tuning.
"""

import pandas as pd
import json
import re
from typing import List, Dict, Any
from datasets import Dataset, DatasetDict
import argparse
from pathlib import Path


def clean_text(text: str) -> str:
    """Clean and normalize text content."""
    if pd.isna(text) or text is None:
        return ""
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', str(text))
    # Before: "Hello <div>world</div>!"
    # After:  "Hello !"
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove extra quotes and escape characters
    # 左引號右引號 -> 引號
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")
    text = text.replace('&#34;', '"').replace('&#39;', "'")
    
    return text.strip()
    # Removes leading and trailing whitespace


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


def process_counsel_chat_data(csv_path: str, output_path: str, max_samples: int = None) -> None:
    """Process Counsel Chat CSV data and create training dataset."""
    
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    print(f"Loaded {len(df)} samples")
    
    # Filter out rows with missing essential data
    df = df.dropna(subset=['questionText', 'answerText'])
    
    # Clean the text data
    df['questionText'] = df['questionText'].apply(clean_text)
    df['answerText'] = df['answerText'].apply(clean_text)
    
    # Remove very short or very long responses
    df = df[df['answerText'].str.len() > 50]  # At least 50 characters
    df = df[df['answerText'].str.len() < 2000]  # Less than 2000 characters
    
    # Remove very short questions
    df = df[df['questionText'].str.len() > 20]  # At least 20 characters
    
    print(f"After filtering: {len(df)} samples")
    
    # Limit samples if specified
    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)
        print(f"Limited to {max_samples} samples")
    
    # Create training examples
    training_examples = []
    
    for _, row in df.iterrows():
        question = row['questionText']
        answer = row['answerText']
        topic = row.get('topic', '') or row.get('topics', '')
        
        # Create instruction prompt
        instruction = create_instruction_prompt(question, topic)
        
        training_example = {
            "instruction": instruction,
            "input": "",
            "output": answer,
            "topic": topic,
            "upvotes": row.get('upvotes', 0),
            "question_id": row.get('questionID', ''),
        }
        
        training_examples.append(training_example)
    
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
    print(f"Saving dataset to {output_path}...")
    dataset_dict.save_to_disk(output_path)
    
    print(f"Dataset saved successfully!")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Save a sample for inspection
    sample_path = Path(output_path).parent / "sample_data.json"
    with open(sample_path, 'w', encoding='utf-8') as f:
        json.dump(train_examples[:3], f, indent=2, ensure_ascii=False)
    
    print(f"Sample data saved to {sample_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare Counsel Chat dataset for training")
    parser.add_argument(
        "--input_csv", 
        type=str, 
        default="counsel-chat/data/20200325_counsel_chat.csv",
        help="Path to the input CSV file"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="counsel_chat_processed",
        help="Output directory for processed dataset"
    )
    parser.add_argument(
        "--max_samples", 
        type=int, 
        default=None,
        help="Maximum number of samples to process (for testing)"
    )
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    process_counsel_chat_data(
        csv_path=args.input_csv,
        output_path=args.output_dir,
        max_samples=args.max_samples
    )


if __name__ == "__main__":
    main()
