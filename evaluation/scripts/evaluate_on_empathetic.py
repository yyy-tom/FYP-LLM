#!/usr/bin/env python3
"""
Evaluate model on EmpatheticDialogues dataset.
Computes automatic metrics and generates qualitative examples.
"""

import argparse
import json
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk
import numpy as np
from datetime import datetime


def load_model(model_path, device="cuda"):
    """Load model and tokenizer."""
    print(f"\n📥 Loading model from {model_path}...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    
    print(f"✅ Model loaded on {device}")
    return model, tokenizer


def format_prompt(example):
    """
    Format EmpatheticDialogues example into prompt.
    
    EmpatheticDialogues format:
    - context: emotion label (e.g., "joyful", "anxious")
    - prompt: user's utterance
    - utterance: expected empathetic response
    """
    emotion = example.get('context', '')
    user_input = example.get('prompt', '')
    
    # Format similar to your training data
    prompt = f"User: {user_input}\nCounselor:"
    
    return prompt


def generate_response(model, tokenizer, prompt, device="cuda", max_new_tokens=150):
    """Generate response from model."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove the prompt from response
    if "Counselor:" in response:
        response = response.split("Counselor:")[-1].strip()
    
    return response


def calculate_perplexity(model, tokenizer, dataset, device="cuda", max_samples=100):
    """Calculate perplexity on dataset."""
    print(f"\n📊 Calculating perplexity (on {max_samples} samples)...")
    
    total_loss = 0
    total_count = 0
    
    for i, example in enumerate(tqdm(dataset[:max_samples], desc="Perplexity")):
        prompt = format_prompt(example)
        reference = example.get('utterance', '')
        full_text = f"{prompt} {reference}"
        
        inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            total_loss += outputs.loss.item()
            total_count += 1
    
    avg_loss = total_loss / total_count
    perplexity = np.exp(avg_loss)
    
    return perplexity


def calculate_bleu_rouge(reference, candidate):
    """Calculate BLEU and ROUGE scores."""
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        from rouge_score import rouge_scorer
        from nltk.tokenize import word_tokenize
        
        # BLEU
        ref_tokens = word_tokenize(reference.lower())
        cand_tokens = word_tokenize(candidate.lower())
        smoothing = SmoothingFunction().method1
        bleu = sentence_bleu([ref_tokens], cand_tokens, smoothing_function=smoothing)
        
        # ROUGE
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge_scores = scorer.score(reference, candidate)
        
        return {
            'bleu': bleu,
            'rouge1': rouge_scores['rouge1'].fmeasure,
            'rouge2': rouge_scores['rouge2'].fmeasure,
            'rougeL': rouge_scores['rougeL'].fmeasure
        }
    except ImportError:
        print("⚠️  NLTK or rouge-score not installed. Skipping BLEU/ROUGE.")
        return {'bleu': 0, 'rouge1': 0, 'rouge2': 0, 'rougeL': 0}


def evaluate_empathy(response):
    """
    Evaluate empathy indicators in response.
    Returns score between 0 and 1.
    """
    empathy_keywords = [
        "understand", "feel", "difficult", "challenging", "support",
        "acknowledge", "hear", "sounds like", "that must be", "i can see"
    ]
    
    response_lower = response.lower()
    score = sum(1 for kw in empathy_keywords if kw in response_lower)
    # Normalize to 0-1
    return min(score / 3, 1.0)


def run_evaluation(model, tokenizer, dataset, device="cuda", max_samples=100):
    """Run complete evaluation."""
    print("\n" + "=" * 80)
    print("Running Evaluation on EmpatheticDialogues")
    print("=" * 80)
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'model_path': str(model.config._name_or_path),
        'dataset': 'empathetic_dialogues',
        'num_samples': max_samples,
        'metrics': {},
        'examples': []
    }
    
    # 1. Perplexity
    perplexity = calculate_perplexity(model, tokenizer, dataset, device, max_samples)
    results['metrics']['perplexity'] = float(perplexity)
    print(f"\n✅ Perplexity: {perplexity:.2f}")
    
    # 2. Generation metrics
    print(f"\n📝 Generating responses and computing metrics...")
    bleu_scores = []
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    empathy_scores = []
    
    for i, example in enumerate(tqdm(dataset[:max_samples], desc="Generation")):
        prompt = format_prompt(example)
        reference = example.get('utterance', '')
        generated = generate_response(model, tokenizer, prompt, device)
        
        # Calculate metrics
        scores = calculate_bleu_rouge(reference, generated)
        bleu_scores.append(scores['bleu'])
        rouge1_scores.append(scores['rouge1'])
        rouge2_scores.append(scores['rouge2'])
        rougeL_scores.append(scores['rougeL'])
        
        # Empathy score
        empathy = evaluate_empathy(generated)
        empathy_scores.append(empathy)
        
        # Save first 10 examples
        if i < 10:
            results['examples'].append({
                'emotion': example.get('context', ''),
                'input': example.get('prompt', ''),
                'reference': reference,
                'generated': generated,
                'empathy_score': empathy,
                'bleu': scores['bleu'],
                'rouge1': scores['rouge1']
            })
    
    # Aggregate metrics
    results['metrics']['bleu'] = float(np.mean(bleu_scores))
    results['metrics']['rouge1'] = float(np.mean(rouge1_scores))
    results['metrics']['rouge2'] = float(np.mean(rouge2_scores))
    results['metrics']['rougeL'] = float(np.mean(rougeL_scores))
    results['metrics']['empathy'] = float(np.mean(empathy_scores))
    
    print("\n" + "=" * 80)
    print("📊 Evaluation Results")
    print("=" * 80)
    print(f"Perplexity:     {results['metrics']['perplexity']:.2f}")
    print(f"BLEU Score:     {results['metrics']['bleu']:.4f}")
    print(f"ROUGE-1:        {results['metrics']['rouge1']:.4f}")
    print(f"ROUGE-2:        {results['metrics']['rouge2']:.4f}")
    print(f"ROUGE-L:        {results['metrics']['rougeL']:.4f}")
    print(f"Empathy Score:  {results['metrics']['empathy']:.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on EmpatheticDialogues")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model (e.g., Qwen/Qwen2.5-7B-Instruct or models/finetuned)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="datasets/empathetic_dialogues_eval",
        help="Path to EmpatheticDialogues dataset"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=100,
        help="Maximum number of samples to evaluate (default: 100)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results.json",
        help="Output file for results"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (cuda/cpu)"
    )
    
    args = parser.parse_args()
    
    # Load dataset
    print(f"\n📂 Loading dataset from {args.dataset}...")
    dataset = load_from_disk(args.dataset)
    test_data = dataset['test']
    print(f"✅ Loaded {len(test_data)} test examples")
    
    # Load model
    model, tokenizer = load_model(args.model, args.device)
    
    # Run evaluation
    results = run_evaluation(
        model, tokenizer, test_data,
        device=args.device,
        max_samples=args.max_samples
    )
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to {args.output}")
    
    # Show sample outputs
    print("\n" + "=" * 80)
    print("📝 Sample Outputs (first 3)")
    print("=" * 80)
    for i, example in enumerate(results['examples'][:3]):
        print(f"\n--- Example {i+1} ---")
        print(f"Emotion: {example['emotion']}")
        print(f"Input: {example['input']}")
        print(f"Reference: {example['reference']}")
        print(f"Generated: {example['generated']}")
        print(f"Empathy: {example['empathy_score']:.2f}, BLEU: {example['bleu']:.3f}")


if __name__ == "__main__":
    main()

