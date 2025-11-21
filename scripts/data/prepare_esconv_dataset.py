#!/usr/bin/env python3
"""
Dataset preparation script for ESConv (Empathetic Conversations) dataset.
This script loads the dataset from Hugging Face and formats it for instruction tuning.
Dataset: https://huggingface.co/datasets/thu-coai/esconv
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


def create_instruction_prompt(question: str, context: str = "", strategy: str = "") -> str:
    """Create an instruction prompt for the counseling question."""
    prompt = """You are a compassionate and professional mental health counselor. Please provide helpful, empathetic, and evidence-based advice for the following question."""
    
    if context:
        prompt += f"\n\nContext: {context}"
    
    if strategy:
        prompt += f"\n\nCounseling Strategy: {strategy}"
    
    prompt += f"\n\nQuestion: {question}\n\nPlease provide a thoughtful and supportive response that:\n1. Acknowledges the person's feelings\n2. Offers practical advice\n3. Suggests professional resources if appropriate\n4. Maintains a warm, non-judgmental tone\n\nResponse:"
    
    return prompt


def process_esconv_data(
    dataset_name: str = "thu-coai/esconv",
    output_path: str = "esconv_processed",
    max_samples: Optional[int] = None,
    cache_dir: Optional[str] = None
) -> None:
    """Process ESConv dataset and create training dataset."""
    
    print(f"Loading dataset from Hugging Face: {dataset_name}")
    
    try:
        dataset = load_dataset(dataset_name, cache_dir=cache_dir)
        
        # Try to get train split, otherwise use the first available split
        if 'train' in dataset:
            train_data = dataset['train']
        elif len(dataset) > 0:
            # Use first available split
            first_split = list(dataset.keys())[0]
            train_data = dataset[first_split]
            print(f"Note: Using '{first_split}' split (train not available)")
        else:
            raise ValueError("No data found in dataset")
        
        print(f"Loaded {len(train_data)} samples")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise
    
    # Inspect structure
    if len(train_data) > 0:
        sample = train_data[0]
        print(f"\nSample structure:")
        print(f"  Keys: {list(sample.keys())}")
        if 'text' in sample:
            text_sample = sample['text'][:200] if len(sample['text']) > 200 else sample['text']
            print(f"  Text preview: {text_sample}...")
    
    # Process dataset
    print(f"\nProcessing dataset...")
    training_examples = []
    
    dataset_to_process = train_data
    if max_samples and len(train_data) > max_samples:
        dataset_to_process = train_data.select(range(max_samples))
        print(f"Limited to {max_samples} samples")
    
    for idx, example in enumerate(dataset_to_process):
        try:
            # ESConv stores data as JSON string in 'text' field
            if 'text' in example:
                data = json.loads(example['text'])
            else:
                data = example
            
            # Extract dialogue
            dialog = data.get('dialog', [])
            if not dialog:
                continue
            
            # Build conversation context
            context_messages = []
            current_user_msg = None
            
            for turn in dialog:
                speaker = turn.get('speaker', '')
                text = clean_text(str(turn.get('text', '')))
                strategy = turn.get('strategy', '')
                
                if not text:
                    continue
                
                # User message
                if speaker == 'usr':
                    current_user_msg = text
                
                # System/assistant message
                elif speaker == 'sys' and current_user_msg:
                    if len(current_user_msg) > 5 and len(text) > 10:
                        # Build context
                        context = ""
                        if context_messages:
                            recent = context_messages[-4:] if len(context_messages) > 4 else context_messages
                            context_parts = []
                            for m in recent:
                                context_parts.append(f"User: {m['user']}")
                                context_parts.append(f"Counselor: {m['assistant']}")
                            context = " ".join(context_parts)
                        
                        # Create instruction
                        instruction = create_instruction_prompt(
                            current_user_msg, 
                            context if context else "",
                            strategy if strategy else ""
                        )
                        
                        training_example = {
                            "instruction": instruction,
                            "input": "",
                            "output": text,
                            "topic": data.get('problem_type', ''),
                            "upvotes": 0,
                            "question_id": f"{idx}_turn_{len(training_examples)}",
                        }
                        
                        training_examples.append(training_example)
                        
                        # Add to context
                        context_messages.append({
                            "user": current_user_msg,
                            "assistant": text
                        })
                    
                    current_user_msg = None
        
        except Exception as e:
            print(f"Warning: Error processing example {idx}: {e}")
            continue
        
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(dataset_to_process)} dialogues, created {len(training_examples)} training examples...")
    
    if not training_examples:
        raise ValueError("No valid training examples created")
    
    print(f"\nCreated {len(training_examples)} valid training examples")
    
    # Filter by length
    filtered = [ex for ex in training_examples if 20 <= len(ex['output']) <= 2000]
    print(f"After filtering by length: {len(filtered)} examples")
    
    # Split into train/validation (90/10)
    train_size = int(0.9 * len(filtered))
    train_examples = filtered[:train_size]
    val_examples = filtered[train_size:]
    
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
    sample_path = Path("samples") / "esconv_sample_data.json"
    with open(sample_path, 'w', encoding='utf-8') as f:
        json.dump(train_examples[:3], f, indent=2, ensure_ascii=False)
    
    print(f"Sample data saved to {sample_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare ESConv dataset for training")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="thu-coai/esconv",
        help="Hugging Face dataset name"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="esconv_processed",
        help="Output directory for processed dataset"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of dialogues to process"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Cache directory for Hugging Face datasets"
    )
    
    args = parser.parse_args()
    
    try:
        process_esconv_data(
            dataset_name=args.dataset_name,
            output_path=args.output_dir,
            max_samples=args.max_samples,
            cache_dir=args.cache_dir
        )
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

