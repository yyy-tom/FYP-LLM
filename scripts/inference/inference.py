#!/usr/bin/env python3
"""
Inference script for the fine-tuned Qwen2.5 model on Counsel Chat dataset.
"""

import os
from pathlib import Path

_LARGE_DISK_PATH = Path(os.environ.get("HF_LARGE_DISK_PATH", "/research/d7/fyp25/yyyu2"))


def _configure_large_disk_cache() -> None:
    """Relocate HuggingFace cache directories onto a large disk (if available)."""
    if not _LARGE_DISK_PATH.exists():
        return

    cache_base = _LARGE_DISK_PATH / ".cache" / "huggingface"
    tmp_dir = _LARGE_DISK_PATH / ".cache" / "tmp"
    env_dirs = {
        "HF_HOME": cache_base,
        "TRANSFORMERS_CACHE": cache_base / "transformers",
        "HF_DATASETS_CACHE": cache_base / "datasets",
        "HF_HUB_CACHE": cache_base / "hub",
        "XET_CACHE": cache_base / "xet",
        "TMPDIR": tmp_dir,
        "TMP": tmp_dir,
        "TEMP": tmp_dir,
    }

    for directory in env_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    for env_var, directory in env_dirs.items():
        os.environ[env_var] = str(directory)

    print(f"✓ Using large disk cache: {cache_base}")
    print(f"✓ Temporary files directory: {tmp_dir}")


_configure_large_disk_cache()

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import argparse


def load_model_and_tokenizer(model_path: str, base_model: str = "Qwen/Qwen2.5-0.5B-Instruct"):
    """Load the fine-tuned model and tokenizer."""
    print(f"Loading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load LoRA weights if they exist
    try:
        model = PeftModel.from_pretrained(model, model_path)
        print(f"Loaded LoRA weights from {model_path}")
    except:
        print(f"No LoRA weights found at {model_path}, using base model")
    
    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    question: str,
    history: list[tuple[str, str]] | None = None,
    max_length: int = 512,
):
    """Generate a counseling response for the given question."""
    
    history_block = ""
    if history:
        history_lines = []
        for user_turn, counselor_turn in history:
            history_lines.append(f"User: {user_turn}")
            history_lines.append(f"Counselor: {counselor_turn}")
        history_block = "\n".join(history_lines).strip()
        if history_block:
            history_block += "\n\n"

    # Create the prompt
    prompt = f"""{history_block}You are a compassionate and professional mental health counselor. Please provide helpful, empathetic, and evidence-based advice for the following question.

Question: {question}

Please provide a thoughtful and supportive response that:
1. Acknowledges the person's feelings
2. Offers practical advice
3. Suggests professional resources if appropriate
4. Maintains a warm, non-judgmental tone

Response:"""
    
    # Tokenize and move tensors to the model device
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=min(max_length, 256),  # Limit to reasonable length
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=3,
        )
    
    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the generated part
    response = response[len(prompt):].strip()
    
    # Clean up the response - stop at common ending patterns
    stop_patterns = [
        "\n\nQuestion:", "\n\nHuman:", "\n\nUser:", 
        "[End]", "\n\nBased on", "\n\nThis response",
        "\n\nYou need to", "\n\nHuman Resources"
    ]
    
    for pattern in stop_patterns:
        if pattern in response:
            response = response.split(pattern)[0].strip()
            break
    
    # If response is too long, truncate at sentence boundary
    if len(response) > 500:
        sentences = response.split('. ')
        response = '. '.join(sentences[:3]) + '.'
    
    return response


def main():
    parser = argparse.ArgumentParser(description="Inference with fine-tuned Qwen2.5 model")
    parser.add_argument(
        "--model_path",
        type=str,
        default="qwen2.5-counsel-chat-finetuned",
        help="Path to the fine-tuned model"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Base model name"
    )
    parser.add_argument(
        "--question",
        type=str,
        help="Question to ask the model"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    
    args = parser.parse_args()
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.base_model)
    
    if args.interactive:
        print("Interactive mode. Type 'quit' to exit.")
        print("=" * 50)
        conversation_history: list[tuple[str, str]] = []
        history_limit = 6
        
        while True:
            question = input("\nYour question: ").strip()
            if question.lower() in ['quit', 'exit', 'q']:
                break
            
            if question:
                print("\nGenerating response...")
                response = generate_response(
                    model, tokenizer, question, conversation_history
                )
                print(f"\nCounselor: {response}")
                print("-" * 50)
                conversation_history.append((question, response))
                if len(conversation_history) > history_limit:
                    conversation_history.pop(0)
    
    elif args.question:
        print(f"Question: {args.question}")
        print("\nGenerating response...")
        response = generate_response(model, tokenizer, args.question)
        print(f"\nCounselor: {response}")
    
    else:
        # Default test questions
        test_questions = [
            "I'm feeling really anxious about my job interview tomorrow. How can I calm my nerves?",
            "I've been having trouble sleeping lately. What can I do to improve my sleep?",
            "I feel like I'm not good enough and everyone is better than me. How can I build my self-confidence?"
        ]
        
        print("Testing with sample questions:")
        print("=" * 50)
        
        for i, question in enumerate(test_questions, 1):
            print(f"\nTest {i}: {question}")
            print("\nGenerating response...")
            response = generate_response(model, tokenizer, question)
            print(f"\nCounselor: {response}")
            print("-" * 50)


if __name__ == "__main__":
    main()
