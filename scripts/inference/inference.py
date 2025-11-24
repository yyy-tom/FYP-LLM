#!/usr/bin/env python3
"""
Inference script for the fine-tuned Qwen2.5 model on Counsel Chat dataset.
"""

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


def generate_response(model, tokenizer, question: str, max_length: int = 512, conversation_history: list = None):
    """Generate a counseling response for the given question.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        question: Current user question
        max_length: Maximum response length
        conversation_history: List of previous conversation turns in format 
                             [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    """
    
    # Use chat template if available (better for multi-turn conversations)
    if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
        messages = []
        
        # Add conversation history if available
        if conversation_history:
            messages.extend(conversation_history)
        
        # Add system message and current user question
        messages.append({"role": "user", "content": question})
        
        prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
    else:
        # Fallback: Build prompt with conversation history manually
        prompt_parts = [
            "You are a compassionate and professional mental health counselor. Please provide helpful, empathetic, and evidence-based advice."
        ]
        
        # Add conversation history if available
        if conversation_history:
            prompt_parts.append("\n\nPrevious conversation:")
            for turn in conversation_history:
                role = turn.get("role", "")
                content = turn.get("content", "")
                if role == "user":
                    prompt_parts.append(f"User: {content}")
                elif role == "assistant":
                    prompt_parts.append(f"Counselor: {content}")
        
        # Add current question
        prompt_parts.append(f"\n\nQuestion: {question}")
        prompt_parts.append("\nPlease provide a thoughtful and supportive response that:")
        prompt_parts.append("1. Acknowledges the person's feelings")
        prompt_parts.append("2. Offers practical advice")
        prompt_parts.append("3. Suggests professional resources if appropriate")
        prompt_parts.append("4. Maintains a warm, non-judgmental tone")
        prompt_parts.append("\nResponse:")
        
        prompt = "\n".join(prompt_parts)
    
    # Use larger max_length for multi-turn conversations to preserve history
    max_input_length = 2048 if conversation_history else 1024
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_length)
    
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
            early_stopping=True,
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
        print("Interactive mode. Type 'quit' to exit, 'clear' to clear conversation history.")
        print("=" * 50)
        
        # Maintain conversation history across turns
        conversation_history = []
        
        while True:
            question = input("\nYour question: ").strip()
            if question.lower() in ['quit', 'exit', 'q']:
                break
            
            # Clear conversation history
            if question.lower() in ['clear', 'reset']:
                conversation_history = []
                print("Conversation history cleared.")
                continue
            
            if question:
                print("\nGenerating response...")
                response = generate_response(model, tokenizer, question, conversation_history=conversation_history)
                print(f"\nCounselor: {response}")
                print("-" * 50)
                
                # Update conversation history
                conversation_history.append({"role": "user", "content": question})
                conversation_history.append({"role": "assistant", "content": response})
    
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
