#!/usr/bin/env python3
"""
Dataset preparation script for Kaggle Mental Health Counseling Conversations dataset.
This script processes the Kaggle dataset and formats it for instruction tuning with Qwen2.5.
"""

import pandas as pd
import json
import re
import os
from typing import List, Dict, Any, Optional
from datasets import Dataset, DatasetDict
import argparse
from pathlib import Path
import glob


def clean_text(text: str) -> str:
    """Clean and normalize text content."""
    if pd.isna(text) or text is None:
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


def find_csv_files(input_dir: str) -> List[str]:
    """Find all CSV files in the input directory."""
    csv_files = []
    
    # Check if input_dir is a file
    if os.path.isfile(input_dir):
        if input_dir.endswith('.csv'):
            return [input_dir]
        else:
            raise ValueError(f"Input file {input_dir} is not a CSV file")
    
    # Search for CSV files
    patterns = [
        os.path.join(input_dir, "*.csv"),
        os.path.join(input_dir, "**", "*.csv"),
    ]
    
    for pattern in patterns:
        csv_files.extend(glob.glob(pattern, recursive=True))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")
    
    return csv_files


def detect_column_mapping(df: pd.DataFrame) -> Dict[str, str]:
    """Detect column names for questions and answers."""
    # Common column name patterns
    question_patterns = [
        'question', 'query', 'user', 'client', 'patient', 'input', 
        'text', 'message', 'prompt', 'utterance'
    ]
    answer_patterns = [
        'answer', 'response', 'reply', 'counselor', 'therapist', 
        'output', 'assistant', 'counseling'
    ]
    topic_patterns = ['topic', 'category', 'theme', 'subject', 'label']
    
    mapping = {
        'question': None,
        'answer': None,
        'topic': None
    }
    
    columns_lower = {col.lower(): col for col in df.columns}
    
    # Find question column
    for pattern in question_patterns:
        for col_lower, col in columns_lower.items():
            if pattern in col_lower:
                mapping['question'] = col
                break
        if mapping['question']:
            break
    
    # Find answer column
    for pattern in answer_patterns:
        for col_lower, col in columns_lower.items():
            if pattern in col_lower:
                mapping['answer'] = col
                break
        if mapping['answer']:
            break
    
    # Find topic column (optional)
    for pattern in topic_patterns:
        for col_lower, col in columns_lower.items():
            if pattern in col_lower:
                mapping['topic'] = col
                break
        if mapping['topic']:
            break
    
    return mapping


def process_kaggle_data(
    csv_path: str, 
    output_path: str, 
    max_samples: Optional[int] = None,
    question_col: Optional[str] = None,
    answer_col: Optional[str] = None,
    topic_col: Optional[str] = None
) -> None:
    """Process Kaggle Mental Health dataset and create training dataset."""
    
    print(f"Loading data from {csv_path}...")
    try:
        # Try different encodings
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        df = None
        for encoding in encodings:
            try:
                df = pd.read_csv(csv_path, encoding=encoding)
                print(f"Successfully loaded with {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue
        
        if df is None:
            raise ValueError("Could not read CSV file with any encoding")
            
    except Exception as e:
        print(f"Error loading CSV: {e}")
        raise
    
    print(f"Loaded {len(df)} samples")
    print(f"Columns: {list(df.columns)}")
    
    # Detect column mapping if not provided
    if not question_col or not answer_col:
        mapping = detect_column_mapping(df)
        question_col = question_col or mapping['question']
        answer_col = answer_col or mapping['answer']
        topic_col = topic_col or mapping['topic']
        
        print(f"\nDetected column mapping:")
        print(f"  Question: {question_col}")
        print(f"  Answer: {answer_col}")
        print(f"  Topic: {topic_col}")
    
    # Validate columns exist
    if not question_col or question_col not in df.columns:
        raise ValueError(f"Question column '{question_col}' not found. Available columns: {list(df.columns)}")
    if not answer_col or answer_col not in df.columns:
        raise ValueError(f"Answer column '{answer_col}' not found. Available columns: {list(df.columns)}")
    
    # Filter out rows with missing essential data
    initial_count = len(df)
    df = df.dropna(subset=[question_col, answer_col])
    print(f"After removing rows with missing questions/answers: {len(df)} samples (removed {initial_count - len(df)})")
    
    # Clean the text data
    df[question_col] = df[question_col].apply(clean_text)
    df[answer_col] = df[answer_col].apply(clean_text)
    
    # Remove empty strings after cleaning
    df = df[df[question_col].str.len() > 0]
    df = df[df[answer_col].str.len() > 0]
    
    # Remove very short or very long responses
    df = df[df[answer_col].str.len() > 50]  # At least 50 characters
    df = df[df[answer_col].str.len() < 2000]  # Less than 2000 characters
    
    # Remove very short questions
    df = df[df[question_col].str.len() > 20]  # At least 20 characters
    
    print(f"After filtering by length: {len(df)} samples")
    
    # Limit samples if specified
    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)
        print(f"Limited to {max_samples} samples")
    
    # Create training examples
    training_examples = []
    
    for idx, row in df.iterrows():
        question = row[question_col]
        answer = row[answer_col]
        topic = ""
        
        if topic_col and topic_col in df.columns:
            topic_val = row.get(topic_col, '')
            topic = clean_text(str(topic_val)) if pd.notna(topic_val) else ""
        
        # Create instruction prompt
        instruction = create_instruction_prompt(question, topic if topic else "")
        
        training_example = {
            "instruction": instruction,
            "input": "",
            "output": answer,
            "topic": topic,
            "upvotes": 0,  # Kaggle dataset may not have upvotes
            "question_id": str(idx),  # Use row index as ID
        }
        
        training_examples.append(training_example)
    
    if not training_examples:
        raise ValueError("No valid training examples created from the dataset")
    
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
    sample_path = Path("samples") / "kaggle_sample_data.json"
    with open(sample_path, 'w', encoding='utf-8') as f:
        json.dump(train_examples[:3], f, indent=2, ensure_ascii=False)
    
    print(f"Sample data saved to {sample_path}")


def main():
    import os
    
    parser = argparse.ArgumentParser(
        description="Prepare Kaggle Mental Health Counseling dataset for training"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to the input directory or CSV file from Kaggle dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="kaggle_mental_health_processed",
        help="Output directory for processed dataset"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (for testing)"
    )
    parser.add_argument(
        "--question_col",
        type=str,
        default=None,
        help="Name of the question column (auto-detected if not provided)"
    )
    parser.add_argument(
        "--answer_col",
        type=str,
        default=None,
        help="Name of the answer column (auto-detected if not provided)"
    )
    parser.add_argument(
        "--topic_col",
        type=str,
        default=None,
        help="Name of the topic column (optional, auto-detected if not provided)"
    )
    
    args = parser.parse_args()
    
    # Find CSV files
    try:
        csv_files = find_csv_files(args.input_dir)
        print(f"Found {len(csv_files)} CSV file(s):")
        for f in csv_files:
            print(f"  - {f}")
    except Exception as e:
        print(f"Error finding CSV files: {e}")
        return
    
    # Process each CSV file
    for csv_file in csv_files:
        print(f"\n{'='*60}")
        print(f"Processing: {csv_file}")
        print(f"{'='*60}")
        
        # Create output path based on CSV filename
        csv_name = Path(csv_file).stem
        if len(csv_files) == 1:
            output_path = args.output_dir
        else:
            output_path = os.path.join(args.output_dir, csv_name)
        
        try:
            process_kaggle_data(
                csv_path=csv_file,
                output_path=output_path,
                max_samples=args.max_samples,
                question_col=args.question_col,
                answer_col=args.answer_col,
                topic_col=args.topic_col
            )
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            import traceback
            traceback.print_exc()
            continue


if __name__ == "__main__":
    main()

