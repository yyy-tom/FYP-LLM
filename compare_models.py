#!/usr/bin/env python3
"""
Script to compare base model vs fine-tuned model responses.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import argparse

def load_model_and_tokenizer(model_path: str, base_model: str = "Qwen/Qwen2.5-0.5B-Instruct"):
    """Load the model and tokenizer."""
    print(f"Loading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load LoRA weights if they exist
    if model_path:
        try:
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Loaded LoRA weights from {model_path}")
        except:
            print(f"No LoRA weights found at {model_path}, using base model")
    
    return model, tokenizer

def generate_response(model, tokenizer, question: str):
    """Generate a counseling response."""
    prompt = f"""You are a compassionate and professional mental health counselor. Please provide helpful, empathetic, and evidence-based advice for the following question.

Question: {question}

Please provide a thoughtful and supportive response that:
1. Acknowledges the person's feelings
2. Offers practical advice
3. Suggests professional resources if appropriate
4. Maintains a warm, non-judgmental tone

Response:"""
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=3,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response[len(prompt):].strip()
    
    # Clean up response
    stop_patterns = [
        "\n\nQuestion:", "\n\nHuman:", "\n\nUser:", 
        "[End]", "\n\nBased on", "\n\nThis response",
        "\n\nYou need to", "\n\nHuman Resources"
    ]
    
    for pattern in stop_patterns:
        if pattern in response:
            response = response.split(pattern)[0].strip()
            break
    
    if len(response) > 500:
        sentences = response.split('. ')
        response = '. '.join(sentences[:3]) + '.'
    
    return response

def main():
    parser = argparse.ArgumentParser(description="Compare base vs fine-tuned model")
    parser.add_argument("--model_path", type=str, default="qwen2.5-counsel-chat-finetuned")
    parser.add_argument("--question", type=str, required=True)
    
    args = parser.parse_args()
    
    # Test questions
    test_questions = [
        "I'm feeling really anxious about my job interview tomorrow. How can I calm my nerves?",
        "I've been having trouble sleeping lately. What can I do to improve my sleep?",
        "I feel like I'm not good enough and everyone is better than me. How can I build my self-confidence?",
        "I'm struggling with depression and don't know where to turn. What should I do?",
        "My relationship with my partner is falling apart. How can we fix it?"
    ]
    
    questions = [args.question] if args.question else test_questions
    
    print("=" * 80)
    print("MODEL COMPARISON: Base vs Fine-tuned Qwen2.5")
    print("=" * 80)
    
    for i, question in enumerate(questions, 1):
        print(f"\n{'='*20} QUESTION {i} {'='*20}")
        print(f"Question: {question}")
        print("\n" + "-"*60)
        
        # Base model
        print("BASE MODEL RESPONSE:")
        base_model, tokenizer = load_model_and_tokenizer("")
        base_response = generate_response(base_model, tokenizer, question)
        print(base_response)
        
        print("\n" + "-"*60)
        
        # Fine-tuned model
        print("FINE-TUNED MODEL RESPONSE:")
        finetuned_model, tokenizer = load_model_and_tokenizer(args.model_path)
        finetuned_response = generate_response(finetuned_model, tokenizer, question)
        print(finetuned_response)
        
        print("\n" + "="*80)
        
        # Clear memory
        del base_model, finetuned_model, tokenizer
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
