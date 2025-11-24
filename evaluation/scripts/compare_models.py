#!/usr/bin/env python3
"""
Compare outputs from base model and trained model side-by-side.

This script loads both models and generates responses for the same inputs,
displaying them side-by-side for easy comparison.
"""

import argparse
import json
import re
import os
import torch
from pathlib import Path
from typing import List, Dict, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Set HuggingFace cache directory to use large disk space
# This should be set before any model loading
LARGE_DISK_PATH = "/research/d7/fyp25/yyyu2"
if os.path.exists(LARGE_DISK_PATH):
    # Set HuggingFace cache directories to use the large disk
    cache_base = f"{LARGE_DISK_PATH}/.cache/huggingface"
    os.environ['HF_HOME'] = cache_base
    os.environ['TRANSFORMERS_CACHE'] = f"{cache_base}/transformers"
    os.environ['HF_DATASETS_CACHE'] = f"{cache_base}/datasets"
    os.environ['HF_HUB_CACHE'] = f"{cache_base}/hub"
    os.environ['XET_CACHE'] = f"{cache_base}/xet"
    
    # CRITICAL: Set TMPDIR to large disk to avoid quota issues during download
    # HuggingFace uses temp directories during download, which can hit quota limits
    tmp_dir = f"{LARGE_DISK_PATH}/.cache/tmp"
    os.environ['TMPDIR'] = tmp_dir
    os.environ['TMP'] = tmp_dir
    os.environ['TEMP'] = tmp_dir
    
    # Create cache directories if they don't exist
    for cache_dir in [
        os.environ['HF_HOME'],
        os.environ['TRANSFORMERS_CACHE'],
        os.environ['HF_DATASETS_CACHE'],
        os.environ['HF_HUB_CACHE'],
        os.environ['XET_CACHE'],
        tmp_dir
    ]:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"✓ Using large disk cache: {cache_base}")
    print(f"✓ Temporary files directory: {tmp_dir}")

# Set CUDA environment variables BEFORE any CUDA operations
if 'CUDA_VISIBLE_DEVICES' in os.environ:
    pass


def load_model(
    model_path: Optional[str] = None,
    base_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    device: str = "cuda",
    local_files_only: bool = False
):
    """Load model - base model only or fine-tuned with LoRA."""
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU")
        device = "cpu"
    
    print(f"Loading base model: {base_model_name}")
    if local_files_only:
        print("⚠️  Using local files only (no download)")
    print(f"Using device: {device}")
    
    # Get cache directory from environment
    cache_dir = os.environ.get('HF_HUB_CACHE', os.environ.get('HF_HOME', None))
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
            trust_remote_code=True
        )
    except Exception as e:
        if "Disk quota exceeded" in str(e) or "quota" in str(e).lower():
            print(f"⚠️  Disk quota error loading tokenizer. Trying local files only...")
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    base_model_name,
                    local_files_only=True,
                    cache_dir=cache_dir,
                    trust_remote_code=True
                )
            except Exception as e2:
                raise OSError(f"Failed to load tokenizer. Error: {e2}")
        else:
            raise
    
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        if device == "cpu":
            base_model = base_model.to("cpu")
    except Exception as e:
        if "Disk quota exceeded" in str(e) or "quota" in str(e).lower():
            print(f"⚠️  Disk quota error. Trying to load from local cache only...")
            if not local_files_only:
                try:
                    base_model = AutoModelForCausalLM.from_pretrained(
                        base_model_name,
                        torch_dtype=dtype,
                        device_map="auto" if device == "cuda" else None,
                        local_files_only=True,
                        cache_dir=cache_dir,
                        trust_remote_code=True
                    )
                    if device == "cpu":
                        base_model = base_model.to("cpu")
                    print("✓ Loaded model from local cache")
                except Exception as e2:
                    raise OSError(f"Failed to load model. Error: {e2}")
            else:
                raise OSError(f"Failed to load model from local cache. Error: {e}")
        else:
            raise
    
    # Check if LoRA weights exist
    if model_path and Path(model_path).exists() and any(Path(model_path).iterdir()):
        print(f"Loading LoRA weights from: {model_path}")
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        if model_path:
            print(f"Note: LoRA path '{model_path}' not found, using base model only")
        else:
            print("Using base model only (no fine-tuned weights)")
        model = base_model
    
    model.eval()
    return model, tokenizer


def format_prompt(input_text: str, tokenizer) -> str:
    """Format input into prompt using chat template if available."""
    if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
        messages = [
            {"role": "user", "content": input_text}
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        # Fallback format
        return f"User: {input_text}\nCounselor:"


def generate_response(
    model,
    tokenizer,
    input_text: str,
    device: str = "cuda",
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9
) -> str:
    """Generate response from model."""
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    
    # Handle DataParallel wrapper
    if isinstance(model, torch.nn.DataParallel):
        actual_model = model.module
        device = "cuda"
        input_device = "cuda:0"
    else:
        actual_model = model
        if hasattr(model, 'device'):
            device = str(model.device).split(':')[0] if ':' in str(model.device) else str(model.device)
        elif next(model.parameters()).device.type == 'cpu':
            device = "cpu"
        input_device = device
    
    # Format prompt
    prompt = format_prompt(input_text, tokenizer)
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(input_device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = actual_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the generated part
    response = response[len(prompt):].strip()
    
    # Clean up response
    stop_patterns = [
        "\n\nUser:", "\n\nHuman:", "\n\nQuestion:",
        "[End]", "\n\nBased on", "\n\nThis response"
    ]
    for pattern in stop_patterns:
        if pattern in response:
            response = response.split(pattern)[0].strip()
            break
    
    return response


def parse_test_cases_from_markdown(markdown_file: str) -> List[Dict[str, str]]:
    """Parse test cases from the markdown file with patient backgrounds."""
    test_cases = []
    
    with open(markdown_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by patient sections
    sections = re.split(r'^# \*\*Background', content, flags=re.MULTILINE)
    
    for section in sections[1:]:  # Skip first empty section
        # Extract patient name
        name_match = re.search(r'### \*\*Patient Name: ([^\n]+)', section)
        patient_name = name_match.group(1) if name_match else "Unknown"
        
        # Extract overall situation
        situation_match = re.search(r'### \*\*Overall Situation\*\*\s*\n\n(.*?)(?=\n###)', section, re.DOTALL)
        situation = situation_match.group(1).strip() if situation_match else ""
        
        # Create test inputs based on the patient background
        # Generate a few different types of questions
        test_inputs = [
            f"I'm {patient_name}. {situation[:200]}... I don't know what to do. Can you help me?",
            f"Hi, I'm struggling with {situation[:100]}... How should I cope with this?",
            f"I need help. {situation[:150]}... What would you suggest?"
        ]
        
        for i, test_input in enumerate(test_inputs):
            test_cases.append({
                'patient': patient_name,
                'input': test_input,
                'context': situation[:300] + "..." if len(situation) > 300 else situation
            })
    
    return test_cases


def print_comparison(
    test_case: Dict,
    base_response: str,
    trained_response: str,
    width: int = 80
):
    """Print side-by-side comparison of responses."""
    print("\n" + "=" * (width * 2 + 3))
    print(f"Test Case: {test_case.get('patient', 'Unknown')}")
    print("=" * (width * 2 + 3))
    
    print(f"\n📝 Input:")
    print(f"   {test_case['input']}")
    
    if 'context' in test_case:
        print(f"\n📋 Context:")
        print(f"   {test_case['context']}")
    
    print(f"\n{'─' * (width * 2 + 3)}")
    print(f"{'BASE MODEL':<{width}} | {'TRAINED MODEL':<{width}}")
    print("─" * (width * 2 + 3))
    
    # Split responses into lines for side-by-side display
    base_lines = base_response.split('\n')
    trained_lines = trained_response.split('\n')
    max_lines = max(len(base_lines), len(trained_lines))
    
    for i in range(max_lines):
        base_line = base_lines[i] if i < len(base_lines) else ""
        trained_line = trained_lines[i] if i < len(trained_lines) else ""
        
        # Wrap long lines
        base_wrapped = []
        trained_wrapped = []
        
        while len(base_line) > width:
            # Try to break at word boundary
            break_point = base_line.rfind(' ', 0, width)
            if break_point == -1:
                break_point = width
            base_wrapped.append(base_line[:break_point])
            base_line = base_line[break_point:].lstrip()
        base_wrapped.append(base_line)
        
        while len(trained_line) > width:
            break_point = trained_line.rfind(' ', 0, width)
            if break_point == -1:
                break_point = width
            trained_wrapped.append(trained_line[:break_point])
            trained_line = trained_line[break_point:].lstrip()
        trained_wrapped.append(trained_line)
        
        # Print side by side
        max_wrapped = max(len(base_wrapped), len(trained_wrapped))
        for j in range(max_wrapped):
            base_part = base_wrapped[j] if j < len(base_wrapped) else ""
            trained_part = trained_wrapped[j] if j < len(trained_wrapped) else ""
            print(f"{base_part:<{width}} | {trained_part:<{width}}")
    
    print("─" * (width * 2 + 3))
    print()


def compare_models(
    base_model_name: str,
    trained_model_path: Optional[str],
    test_inputs: List[str],
    device: str = "cuda",
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    local_files_only: bool = False,
    output_file: Optional[str] = None
):
    """Compare base and trained models on given test inputs."""
    
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA not available, using CPU instead")
        device = "cpu"
    elif device == "cuda":
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
    
    print("=" * 80)
    print("Model Comparison: Base vs Trained")
    print("=" * 80)
    print(f"Device: {device.upper()}")
    print()
    
    # Load base model
    print("[1/3] Loading base model...")
    base_model, base_tokenizer = load_model(
        None, base_model_name, device, local_files_only
    )
    print("✓ Base model loaded\n")
    
    # Load trained model
    print("[2/3] Loading trained model...")
    if trained_model_path:
        trained_model, trained_tokenizer = load_model(
            trained_model_path, base_model_name, device, local_files_only
        )
    else:
        print("⚠️  No trained model path provided, using base model for both")
        trained_model, trained_tokenizer = base_model, base_tokenizer
    print("✓ Trained model loaded\n")
    
    # Generate responses
    print(f"[3/3] Generating responses for {len(test_inputs)} test cases...")
    print()
    
    results = []
    
    for i, test_input in enumerate(test_inputs, 1):
        print(f"Processing test case {i}/{len(test_inputs)}...")
        
        # Prepare test case dict
        test_case = {
            'id': i,
            'input': test_input,
            'patient': f"Test {i}"
        }
        
        try:
            # Generate from base model
            base_response = generate_response(
                base_model, base_tokenizer, test_input, device,
                max_new_tokens, temperature, top_p
            )
            
            # Generate from trained model
            trained_response = generate_response(
                trained_model, trained_tokenizer, test_input, device,
                max_new_tokens, temperature, top_p
            )
            
            # Store results
            result = {
                'test_case': test_case,
                'base_response': base_response,
                'trained_response': trained_response
            }
            results.append(result)
            
            # Print comparison
            print_comparison(test_case, base_response, trained_response)
            
        except Exception as e:
            print(f"⚠️  Error processing test case {i}: {e}")
            continue
    
    # Save results if output file specified
    if output_file:
        print(f"\n💾 Saving results to {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print("✓ Results saved")
    
    print("\n" + "=" * 80)
    print("Comparison complete!")
    print("=" * 80)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compare outputs from base model and trained model"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Base model name (default: Qwen/Qwen2.5-1.5B-Instruct)"
    )
    parser.add_argument(
        "--trained_model",
        type=str,
        default=None,
        help="Path to trained model (LoRA weights). If not provided, uses base model for both."
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="evaluation/scripts/sample for testing models.md",
        help="Path to markdown file with test cases (default: evaluation/scripts/sample for testing models.md)"
    )
    parser.add_argument(
        "--input",
        type=str,
        action="append",
        help="Test input text (can be used multiple times). If provided, overrides test_file."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use (default: auto)"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum number of new tokens to generate (default: 256)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p sampling parameter (default: 0.9)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file to save results (optional)"
    )
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="Only use local cached files, don't download models"
    )
    
    args = parser.parse_args()
    
    # Auto-detect device
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Auto-detected device: {args.device}")
    
    # Get test inputs
    test_inputs = []
    
    if args.input:
        # Use command-line inputs
        test_inputs = args.input
        print(f"Using {len(test_inputs)} test input(s) from command line")
    else:
        # Parse from markdown file
        if os.path.exists(args.test_file):
            print(f"Parsing test cases from: {args.test_file}")
            test_cases = parse_test_cases_from_markdown(args.test_file)
            test_inputs = [tc['input'] for tc in test_cases]
            print(f"Found {len(test_inputs)} test case(s)")
        else:
            print(f"⚠️  Test file not found: {args.test_file}")
            print("Using default test inputs...")
            test_inputs = [
                "I've been feeling really down lately. Nothing seems to help. What should I do?",
                "I'm stressed about my exams and can't sleep. How can I manage this?",
                "I just went through a breakup and I'm struggling to move on. Can you help?"
            ]
    
    if not test_inputs:
        print("⚠️  No test inputs found. Exiting.")
        return
    
    # Run comparison
    compare_models(
        args.base_model,
        args.trained_model,
        test_inputs,
        args.device,
        args.max_new_tokens,
        args.temperature,
        args.top_p,
        args.local_files_only,
        args.output
    )


if __name__ == "__main__":
    main()

