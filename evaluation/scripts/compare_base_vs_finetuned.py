#!/usr/bin/env python3
"""
Compare evaluation results between base model and fine-tuned model.
Generates comparison tables and visualizations.
"""

import argparse
import json
from pathlib import Path


def load_results(filepath):
    """Load evaluation results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def print_comparison_table(base_results, finetuned_results):
    """Print comparison table."""
    print("\n" + "=" * 80)
    print("📊 MODEL COMPARISON: Base vs Fine-Tuned")
    print("=" * 80)
    
    # Extract metrics
    base_metrics = base_results['metrics']
    ft_metrics = finetuned_results['metrics']
    
    # Print table
    print("\n{:<20} {:>15} {:>15} {:>15}".format(
        "Metric", "Base Model", "Fine-Tuned", "Improvement"
    ))
    print("-" * 80)
    
    metrics_to_compare = [
        ('perplexity', 'Perplexity', True),  # Lower is better
        ('bleu', 'BLEU', False),
        ('rouge1', 'ROUGE-1', False),
        ('rouge2', 'ROUGE-2', False),
        ('rougeL', 'ROUGE-L', False),
        ('empathy', 'Empathy Score', False),
    ]
    
    for metric_key, metric_name, lower_is_better in metrics_to_compare:
        base_val = base_metrics.get(metric_key, 0)
        ft_val = ft_metrics.get(metric_key, 0)
        
        if lower_is_better:
            improvement = ((base_val - ft_val) / base_val) * 100
            symbol = "↓" if ft_val < base_val else "↑"
        else:
            improvement = ((ft_val - base_val) / base_val) * 100 if base_val > 0 else 0
            symbol = "↑" if ft_val > base_val else "↓"
        
        print("{:<20} {:>15.4f} {:>15.4f} {:>13.1f}% {}".format(
            metric_name, base_val, ft_val, abs(improvement), symbol
        ))
    
    print("=" * 80)
    
    # Summary
    print("\n📈 Summary:")
    if ft_metrics['perplexity'] < base_metrics['perplexity']:
        print("✅ Fine-tuned model has LOWER perplexity (better)")
    else:
        print("❌ Fine-tuned model has HIGHER perplexity (worse)")
    
    if ft_metrics['empathy'] > base_metrics['empathy']:
        print("✅ Fine-tuned model has HIGHER empathy score (better)")
    else:
        print("❌ Fine-tuned model has LOWER empathy score (worse)")
    
    if ft_metrics['bleu'] > base_metrics['bleu']:
        print("✅ Fine-tuned model has HIGHER BLEU score (better)")
    else:
        print("❌ Fine-tuned model has LOWER BLEU score (worse)")


def print_example_comparison(base_results, finetuned_results, num_examples=3):
    """Print side-by-side example comparisons."""
    print("\n" + "=" * 80)
    print("📝 EXAMPLE COMPARISONS")
    print("=" * 80)
    
    base_examples = base_results.get('examples', [])
    ft_examples = finetuned_results.get('examples', [])
    
    for i in range(min(num_examples, len(base_examples), len(ft_examples))):
        base_ex = base_examples[i]
        ft_ex = ft_examples[i]
        
        print(f"\n--- Example {i+1} ---")
        print(f"Emotion: {base_ex.get('emotion', 'N/A')}")
        print(f"Input: {base_ex.get('input', 'N/A')}")
        print(f"\nReference: {base_ex.get('reference', 'N/A')}")
        print(f"\nBase Model Response:")
        print(f"  {base_ex.get('generated', 'N/A')}")
        print(f"  (Empathy: {base_ex.get('empathy_score', 0):.2f}, BLEU: {base_ex.get('bleu', 0):.3f})")
        print(f"\nFine-Tuned Model Response:")
        print(f"  {ft_ex.get('generated', 'N/A')}")
        print(f"  (Empathy: {ft_ex.get('empathy_score', 0):.2f}, BLEU: {ft_ex.get('bleu', 0):.3f})")
        print()


def save_comparison_report(base_results, finetuned_results, output_file):
    """Save detailed comparison report."""
    report = {
        'base_model': base_results['model_path'],
        'finetuned_model': finetuned_results['model_path'],
        'dataset': base_results['dataset'],
        'comparison': {}
    }
    
    # Calculate improvements
    base_metrics = base_results['metrics']
    ft_metrics = finetuned_results['metrics']
    
    for metric in base_metrics.keys():
        base_val = base_metrics[metric]
        ft_val = ft_metrics[metric]
        improvement = ((ft_val - base_val) / base_val) * 100 if base_val > 0 else 0
        
        report['comparison'][metric] = {
            'base': base_val,
            'finetuned': ft_val,
            'improvement_percent': improvement
        }
    
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✅ Detailed comparison report saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare base and fine-tuned model evaluation results"
    )
    parser.add_argument(
        "--base",
        type=str,
        required=True,
        help="Path to base model evaluation results (JSON)"
    )
    parser.add_argument(
        "--finetuned",
        type=str,
        required=True,
        help="Path to fine-tuned model evaluation results (JSON)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="comparison_report.json",
        help="Output file for comparison report"
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=3,
        help="Number of examples to show (default: 3)"
    )
    
    args = parser.parse_args()
    
    # Load results
    print("📂 Loading evaluation results...")
    base_results = load_results(args.base)
    finetuned_results = load_results(args.finetuned)
    print("✅ Results loaded")
    
    # Print comparison
    print_comparison_table(base_results, finetuned_results)
    print_example_comparison(base_results, finetuned_results, args.num_examples)
    
    # Save report
    save_comparison_report(base_results, finetuned_results, args.output)
    
    print("\n" + "=" * 80)
    print("✅ Comparison Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

