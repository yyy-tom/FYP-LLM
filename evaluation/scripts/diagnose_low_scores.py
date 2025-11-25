#!/usr/bin/env python3
"""
Diagnostic script to investigate why BLEU/ROUGE scores are low.

This script loads evaluation results and displays actual generated responses
vs reference responses to help identify issues.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    print("Warning: rouge-score not installed. Install with: pip install rouge-score")

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.tokenize import word_tokenize
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    BLEU_AVAILABLE = True
except ImportError:
    BLEU_AVAILABLE = False
    print("Warning: nltk not installed. Install with: pip install nltk")


def calculate_bleu(reference: str, candidate: str) -> float:
    """Calculate BLEU score between reference and candidate."""
    if not BLEU_AVAILABLE:
        return 0.0
    
    try:
        ref_tokens = word_tokenize(reference.lower())
        cand_tokens = word_tokenize(candidate.lower())
        smoothing = SmoothingFunction().method1
        return sentence_bleu([ref_tokens], cand_tokens, smoothing_function=smoothing)
    except:
        return 0.0


def calculate_rouge(reference: str, candidate: str, scorer) -> Dict[str, float]:
    """Calculate ROUGE scores."""
    if not ROUGE_AVAILABLE:
        return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    
    try:
        scores = scorer.score(reference, candidate)
        return {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
        }
    except:
        return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}


def analyze_responses(results_file: str, num_samples: int = 10):
    """Analyze evaluation results to diagnose low scores."""
    
    print("=" * 80)
    print("Diagnosing Low BLEU/ROUGE Scores")
    print("=" * 80)
    print()
    
    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    print(f"Loaded results from: {results_file}")
    print(f"Model type: {results.get('model_type', 'unknown')}")
    print(f"Total samples evaluated: {results.get('test_samples', 0)}")
    print()
    
    # Check if responses are saved
    if 'responses' not in results or not results['responses']:
        print("⚠️  No individual responses found in results file.")
        print("   Re-run evaluation with --save_responses flag:")
        print("   python evaluate_model.py --save_responses --output results.json")
        return
    
    responses = results['responses']
    print(f"Found {len(responses)} saved responses")
    print()
    
    # Initialize ROUGE scorer
    rouge_scorer_obj = None
    if ROUGE_AVAILABLE:
        rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # Analyze first N samples
    print(f"Analyzing first {min(num_samples, len(responses))} samples:")
    print("=" * 80)
    print()
    
    total_bleu = 0.0
    total_rouge1 = 0.0
    total_rouge2 = 0.0
    total_rougeL = 0.0
    valid_samples = 0
    
    for i, resp_data in enumerate(responses[:num_samples]):
        sample_id = resp_data.get('sample_id', i)
        user_input = resp_data.get('input', '')
        reference = resp_data.get('reference', '')
        generated = resp_data.get('generated', '')
        
        print(f"\n{'='*80}")
        print(f"Sample {sample_id + 1}")
        print(f"{'='*80}")
        
        print(f"\n📝 User Input:")
        print(f"   {user_input[:200]}{'...' if len(user_input) > 200 else ''}")
        
        print(f"\n📖 Reference Response:")
        if reference:
            print(f"   {reference[:500]}{'...' if len(reference) > 500 else ''}")
            print(f"   Length: {len(reference)} characters, {len(reference.split())} words")
        else:
            print("   ⚠️  No reference response found!")
        
        print(f"\n🤖 Generated Response:")
        if generated:
            print(f"   {generated[:500]}{'...' if len(generated) > 500 else ''}")
            print(f"   Length: {len(generated)} characters, {len(generated.split())} words")
        else:
            print("   ⚠️  No generated response found!")
        
        # Calculate metrics
        if reference and generated:
            bleu = calculate_bleu(reference, generated)
            if rouge_scorer_obj:
                rouge = calculate_rouge(reference, generated, rouge_scorer_obj)
            else:
                rouge = {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
            
            print(f"\n📊 Metrics:")
            print(f"   BLEU: {bleu:.4f}")
            print(f"   ROUGE-1: {rouge['rouge1']:.4f}")
            print(f"   ROUGE-2: {rouge['rouge2']:.4f}")
            print(f"   ROUGE-L: {rouge['rougeL']:.4f}")
            
            total_bleu += bleu
            total_rouge1 += rouge['rouge1']
            total_rouge2 += rouge['rouge2']
            total_rougeL += rouge['rougeL']
            valid_samples += 1
            
            # Identify potential issues
            issues = []
            if len(generated) < 10:
                issues.append("Generated response is very short")
            if len(generated) > len(reference) * 3:
                issues.append("Generated response is much longer than reference")
            if len(generated) < len(reference) * 0.1:
                issues.append("Generated response is much shorter than reference")
            if not any(word in generated.lower() for word in reference.lower().split()[:10]):
                issues.append("No common words with reference")
            
            if issues:
                print(f"\n⚠️  Potential Issues:")
                for issue in issues:
                    print(f"   - {issue}")
        
        print()
    
    # Summary statistics
    if valid_samples > 0:
        print(f"\n{'='*80}")
        print(f"Summary Statistics (first {valid_samples} samples)")
        print(f"{'='*80}")
        print(f"Average BLEU: {total_bleu / valid_samples:.4f}")
        print(f"Average ROUGE-1: {total_rouge1 / valid_samples:.4f}")
        print(f"Average ROUGE-2: {total_rouge2 / valid_samples:.4f}")
        print(f"Average ROUGE-L: {total_rougeL / valid_samples:.4f}")
        print()
        
        # Compare with overall metrics
        if 'metrics' in results:
            overall_metrics = results['metrics']
            print("Comparison with overall metrics:")
            if 'bleu' in overall_metrics:
                print(f"  Overall BLEU: {overall_metrics['bleu']:.4f}")
                print(f"  Sample BLEU:  {total_bleu / valid_samples:.4f}")
            if 'rouge' in overall_metrics:
                print(f"  Overall ROUGE-1: {overall_metrics['rouge']['rouge1']:.4f}")
                print(f"  Sample ROUGE-1: {total_rouge1 / valid_samples:.4f}")
    
    # Recommendations
    print(f"\n{'='*80}")
    print("Recommendations:")
    print(f"{'='*80}")
    print("1. Check if generated responses are being properly extracted")
    print("2. Verify that the prompt format matches training format")
    print("3. Check if the model is generating complete responses")
    print("4. Compare a few samples manually to see if responses make sense")
    print("5. Consider using the compare_models.py script to see side-by-side comparisons")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose why BLEU/ROUGE scores are low"
    )
    parser.add_argument(
        "--results_file",
        type=str,
        required=True,
        help="Path to evaluation results JSON file (must have --save_responses enabled)"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of samples to analyze in detail (default: 10)"
    )
    
    args = parser.parse_args()
    
    if not Path(args.results_file).exists():
        print(f"Error: Results file not found: {args.results_file}")
        return
    
    analyze_responses(args.results_file, args.num_samples)


if __name__ == "__main__":
    main()

