#!/usr/bin/env python3
"""
Create dataset for expert evaluation by mental health professionals.

This script generates a JSONL file with samples for expert review,
including user inputs, reference responses, and generated responses.
"""

import argparse
import json
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk
from peft import PeftModel

from evaluate_model import load_model, generate_response, format_prompt


def create_expert_evaluation_dataset(
    model_path: str,
    base_model_name: str,
    test_dataset_path: str,
    output_file: str = "expert_evaluation.jsonl",
    num_samples: int = 50,
    device: str = "cuda"
):
    """Create dataset for expert evaluation."""
    
    print("=" * 60)
    print("Creating Expert Evaluation Dataset")
    print("=" * 60)
    
    # Load model
    print(f"\nLoading model from: {model_path}")
    model, tokenizer = load_model(model_path, base_model_name, device)
    print("✓ Model loaded")
    
    # Load test dataset
    print(f"\nLoading test dataset from: {test_dataset_path}")
    test_data = load_from_disk(test_dataset_path)
    if isinstance(test_data, dict):
        test_data = test_data.get("validation", test_data.get("test", list(test_data.values())[0]))
    print(f"✓ Loaded {len(test_data)} test samples")
    
    # Create evaluation samples
    print(f"\nGenerating {num_samples} samples for expert evaluation...")
    evaluation_samples = []
    
    for i, example in enumerate(test_data):
        if i >= num_samples:
            break
        
        try:
            user_input = example.get("input", example.get("instruction", ""))
            reference = example.get("output", "")
            topic = example.get("topic", "unknown")
            
            # Generate response
            generated = generate_response(model, tokenizer, user_input, device)
            
            sample = {
                'id': i + 1,
                'user_input': user_input,
                'reference_response': reference,
                'generated_response': generated,
                'topic': topic,
                'expert_ratings': {
                    'empathy': None,  # 1-5 scale
                    'professionalism': None,  # 1-5 scale
                    'helpfulness': None,  # 1-5 scale
                    'safety': None,  # 1-5 scale
                    'cultural_appropriateness': None,  # 1-5 scale
                    'evidence_based': None  # 1-5 scale
                },
                'expert_comments': "",
                'overall_rating': None  # 1-5 scale
            }
            
            evaluation_samples.append(sample)
            
            if (i + 1) % 10 == 0:
                print(f"  Generated {i + 1}/{num_samples} samples...")
        
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue
    
    # Save to JSONL
    print(f"\nSaving to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in evaluation_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"✓ Created {len(evaluation_samples)} samples for expert evaluation")
    print(f"\nNext steps:")
    print(f"1. Share {output_file} with mental health professionals")
    print(f"2. Ask them to fill in 'expert_ratings' and 'expert_comments'")
    print(f"3. Collect the completed file for analysis")
    
    return evaluation_samples


def main():
    parser = argparse.ArgumentParser(
        description="Create expert evaluation dataset"
    )
    parser.add_argument(
        "--model_path",
        required=True,
        help="Path to fine-tuned model (LoRA weights)"
    )
    parser.add_argument(
        "--base_model",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Base model name"
    )
    parser.add_argument(
        "--test_dataset",
        required=True,
        help="Path to test dataset"
    )
    parser.add_argument(
        "--output",
        default="expert_evaluation.jsonl",
        help="Output JSONL file"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=50,
        help="Number of samples to generate"
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to use (cuda or cpu)"
    )
    
    args = parser.parse_args()
    
    create_expert_evaluation_dataset(
        args.model_path,
        args.base_model,
        args.test_dataset,
        args.output,
        args.num_samples,
        args.device
    )


if __name__ == "__main__":
    main()

