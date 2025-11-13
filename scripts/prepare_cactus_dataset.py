#!/usr/bin/env python3
"""
Dataset preparation script for Cactus dataset.
This script loads the Cactus dataset from Hugging Face and formats it for instruction tuning.
Dataset: https://huggingface.co/datasets/cactus-camel/cactus
Paper: https://arxiv.org/abs/2407.03103
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


def create_instruction_prompt(question: str, context: str = "", cbt_plan: str = "") -> str:
    """Create an instruction prompt for the counseling question."""
    prompt = """You are a compassionate and professional mental health counselor using Cognitive Behavioral Therapy (CBT) techniques. Please provide helpful, empathetic, and evidence-based advice for the following question."""
    
    if context:
        prompt += f"\n\nContext: {context}"
    
    if cbt_plan:
        prompt += f"\n\nCBT Plan: {cbt_plan}"
    
    prompt += f"\n\nQuestion: {question}\n\nPlease provide a thoughtful and supportive response that:\n1. Acknowledges the person's feelings\n2. Uses appropriate CBT techniques\n3. Offers practical advice\n4. Suggests professional resources if appropriate\n5. Maintains a warm, non-judgmental tone\n\nResponse:"
    
    return prompt


def extract_dialogue_turns(dialogue_text: str) -> List[Dict[str, str]]:
    """Extract turns from dialogue text."""
    turns = []
    lines = dialogue_text.split('\n')
    
    current_speaker = None
    current_text = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Check if line starts with "Counselor:" or "Client:"
        if line.startswith('Counselor:'):
            # Save previous turn if exists
            if current_speaker and current_text:
                turns.append({
                    'role': current_speaker,
                    'text': ' '.join(current_text)
                })
            current_speaker = 'assistant'
            current_text = [line.replace('Counselor:', '').strip()]
        elif line.startswith('Client:'):
            # Save previous turn if exists
            if current_speaker and current_text:
                turns.append({
                    'role': current_speaker,
                    'text': ' '.join(current_text)
                })
            current_speaker = 'user'
            current_text = [line.replace('Client:', '').strip()]
        else:
            # Continue current turn
            if current_text:
                current_text.append(line)
    
    # Add last turn
    if current_speaker and current_text:
        turns.append({
            'role': current_speaker,
            'text': ' '.join(current_text)
        })
    
    return turns


def process_cactus_data(
    dataset_name: str = "cactus-camel/cactus",
    output_path: str = "cactus_processed",
    max_samples: Optional[int] = None,
    cache_dir: Optional[str] = None,
    split: str = "train"
) -> None:
    """Process Cactus dataset and create training dataset."""
    
    print(f"Loading dataset from Hugging Face: {dataset_name}")
    print(f"This may take a few minutes for the first download...")
    
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
        print(f"\nSample entry keys: {list(sample.keys())}")
        print(f"Dialogue length: {len(sample.get('dialogue', ''))} chars")
    
    # Process dataset
    print(f"\nProcessing dataset...")
    training_examples = []
    
    dataset_to_process = dataset
    if max_samples and len(dataset) > max_samples:
        dataset_to_process = dataset.select(range(max_samples))
        print(f"Limited to {max_samples} samples")
    
    for idx, example in enumerate(dataset_to_process):
        try:
            # Extract dialogue
            dialogue_text = example.get('dialogue', '')
            if not dialogue_text:
                continue
            
            # Extract dialogue turns
            turns = extract_dialogue_turns(dialogue_text)
            if len(turns) < 2:
                continue
            
            # Extract other information
            thought = clean_text(str(example.get('thought', '')))
            cbt_technique = example.get('cbt_technique', '')
            cbt_plan = clean_text(str(example.get('cbt_plan', '')))
            patterns = example.get('patterns', [])
            intake_form = example.get('intake_form', '')
            
            # Process dialogue into training examples
            context_messages = []
            current_user_msg = None
            
            for turn in turns:
                role = turn.get('role', '')
                text = clean_text(turn.get('text', ''))
                
                if not text:
                    continue
                
                # User message
                if role == 'user':
                    current_user_msg = text
                
                # Assistant message
                elif role == 'assistant' and current_user_msg:
                    if len(current_user_msg) > 5 and len(text) > 10:
                        # Build context from previous messages
                        context = ""
                        if context_messages:
                            recent = context_messages[-4:] if len(context_messages) > 4 else context_messages
                            context_parts = []
                            for m in recent:
                                context_parts.append(f"User: {m['user']}")
                                context_parts.append(f"Counselor: {m['assistant']}")
                            context = " ".join(context_parts)
                        
                        # Add intake form info to context if available and it's early in conversation
                        if len(context_messages) < 2 and intake_form:
                            intake_summary = intake_form.split('\n\n')[0] if '\n\n' in intake_form else intake_form[:200]
                            if context:
                                context = f"Client Info: {intake_summary}\n\n{context}"
                            else:
                                context = f"Client Info: {intake_summary}"
                        
                        # Create instruction prompt
                        instruction = create_instruction_prompt(
                            current_user_msg,
                            context if context else "",
                            cbt_plan if len(context_messages) == 0 else ""  # Include CBT plan in first turn
                        )
                        
                        training_example = {
                            "instruction": instruction,
                            "input": "",
                            "output": text,
                            "topic": cbt_technique if cbt_technique else "",
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
            import traceback
            traceback.print_exc()
            continue
        
        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1}/{len(dataset_to_process)} dialogues, created {len(training_examples)} training examples...")
    
    if not training_examples:
        raise ValueError("No valid training examples created from the dataset")
    
    print(f"\nCreated {len(training_examples)} valid training examples from {len(dataset_to_process)} dialogues")
    
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
    sample_path = Path("samples") / "cactus_sample_data.json"
    with open(sample_path, 'w', encoding='utf-8') as f:
        json.dump(train_examples[:3], f, indent=2, ensure_ascii=False)
    
    print(f"Sample data saved to {sample_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare Cactus dataset for training")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="cactus-camel/cactus",
        help="Hugging Face dataset name"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="cactus_processed",
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
        process_cactus_data(
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

