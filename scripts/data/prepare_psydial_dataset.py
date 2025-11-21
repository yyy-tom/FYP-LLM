#!/usr/bin/env python3
"""
Dataset preparation script for PsyDial dataset.
This script loads the PsyDial datasets from Hugging Face and formats them for instruction tuning.
Dataset: https://huggingface.co/datasets/qiuhuachuan/PsyDial-D4
Paper: https://aclanthology.org/2025.acl-long.1049/
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


def process_multi_turn_conversation(messages: List[Dict], dialogue_id: str) -> List[Dict]:
    """Process a multi-turn conversation into training examples.
    
    Args:
        messages: List of message dicts with 'role' and 'content' fields
        dialogue_id: Unique identifier for this dialogue
    """
    training_examples = []
    
    # Build context as we go through the conversation
    context_messages = []
    current_user_msg = None
    
    for i, msg in enumerate(messages):
        role = msg.get('role', '').lower()
        content = clean_text(str(msg.get('content', '')))
        
        # Skip system messages (they're instructions, not part of the conversation)
        if role == 'system':
            continue
        
        # If it's a user message, store it
        if role in ['user', 'client']:
            current_user_msg = content
        
        # If it's an assistant message and we have a user message, create a training example
        elif role in ['assistant', 'counselor'] and current_user_msg:
            # Only create example if both messages have sufficient content
            if len(current_user_msg) > 5 and len(content) > 10:
                # Build context from previous messages (last 2-3 exchanges)
                context = ""
                if context_messages:
                    # Use last 2-3 exchanges as context
                    recent_context = context_messages[-4:] if len(context_messages) > 4 else context_messages
                    context_parts = []
                    for m in recent_context:
                        context_parts.append(f"User: {m['user']}")
                        context_parts.append(f"Counselor: {m['assistant']}")
                    context = " ".join(context_parts)
                
                # Create instruction prompt
                instruction = create_instruction_prompt(current_user_msg, context if context else "")
                
                training_example = {
                    "instruction": instruction,
                    "input": "",
                    "output": content,
                    "topic": "",
                    "upvotes": 0,
                    "question_id": f"{dialogue_id}_turn_{len(training_examples)}",
                }
                
                training_examples.append(training_example)
                
                # Add to context for next turns
                context_messages.append({
                    "user": current_user_msg,
                    "assistant": content
                })
            
            # Reset current user message after processing
            current_user_msg = None
    
    return training_examples


def process_psydial_data(
    dataset_name: str = "qiuhuachuan/PsyDial-D4",
    output_path: str = "psydial_processed",
    max_samples: Optional[int] = None,
    cache_dir: Optional[str] = None,
    split: str = "train"
) -> None:
    """Process PsyDial dataset and create training dataset."""
    
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
        print("3. Try accessing the dataset page: https://huggingface.co/datasets/qiuhuachuan/PsyDial-D4")
        raise
    
    # Inspect the dataset structure
    print(f"\nDataset features: {dataset.features}")
    if len(dataset) > 0:
        print(f"\nSample entry:")
        sample = dataset[0]
        for key, value in sample.items():
            if isinstance(value, (list, dict)):
                print(f"  {key}: {type(value).__name__} with {len(value) if hasattr(value, '__len__') else 'N/A'} items")
                if isinstance(value, list) and len(value) > 0:
                    print(f"    First item: {value[0]}")
            elif isinstance(value, str) and len(value) > 100:
                print(f"  {key}: {value[:100]}...")
            else:
                print(f"  {key}: {value}")
    
    # Process dataset
    print(f"\nProcessing dataset...")
    all_training_examples = []
    
    # Limit samples if specified
    dataset_to_process = dataset
    if max_samples and len(dataset) > max_samples:
        dataset_to_process = dataset.select(range(max_samples))
        print(f"Limited to {max_samples} samples")
    
    for idx, example in enumerate(dataset_to_process):
        dialogue_id = str(example.get('id', idx))
        
        # Extract messages - PsyDial uses 'messages' field
        messages = None
        
        if 'messages' in example:
            messages = example['messages']
        elif 'conversation' in example:
            messages = example['conversation']
        elif 'dialogues' in example:
            messages = example['dialogues']
        elif 'turns' in example:
            messages = example['turns']
        elif 'dialogue' in example:
            messages = example['dialogue']
        elif isinstance(example, list):
            messages = example
        else:
            # Try to find any list field
            for key, value in example.items():
                if isinstance(value, list) and len(value) > 0:
                    messages = value
                    break
        
        if messages is None or not isinstance(messages, list) or len(messages) == 0:
            print(f"Warning: Could not find valid messages in example {idx}")
            continue
        
        # Process the conversation
        try:
            training_examples = process_multi_turn_conversation(messages, dialogue_id)
            all_training_examples.extend(training_examples)
        except Exception as e:
            print(f"Warning: Error processing conversation {idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(dataset_to_process)} dialogues, created {len(all_training_examples)} training examples...")
    
    if not all_training_examples:
        raise ValueError("No valid training examples created from the dataset")
    
    print(f"\nCreated {len(all_training_examples)} valid training examples from {len(dataset_to_process)} dialogues")
    
    # Filter by length
    filtered_examples = []
    for ex in all_training_examples:
        if len(ex['output']) >= 20 and len(ex['output']) <= 2000:
            filtered_examples.append(ex)
    
    print(f"After filtering by length: {len(filtered_examples)} examples")
    
    # Split into train/validation (90/10)
    train_size = int(0.9 * len(filtered_examples))
    train_examples = filtered_examples[:train_size]
    val_examples = filtered_examples[train_size:]
    
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
    sample_path = Path("samples") / "psydial_sample_data.json"
    with open(sample_path, 'w', encoding='utf-8') as f:
        json.dump(train_examples[:3], f, indent=2, ensure_ascii=False)
    
    print(f"Sample data saved to {sample_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare PsyDial dataset for training"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="qiuhuachuan/PsyDial-D4",
        help="Hugging Face dataset name (e.g., qiuhuachuan/PsyDial-D4)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="psydial_processed",
        help="Output directory for processed dataset"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of dialogues to process (for testing)"
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
        process_psydial_data(
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

