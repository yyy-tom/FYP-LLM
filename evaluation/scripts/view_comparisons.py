
"""
Simple script to view model comparison results from evaluation JSON.
"""

import json
import sys
from pathlib import Path


def view_comparisons(json_path: str, max_examples: int = 10):
    """View comparison results in a readable format."""
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    comparisons = data.get('comparisons', [])
    if not comparisons:
        print("No comparisons found in the results file.")
        print("Make sure you ran evaluation with --compare_with_base flag.")
        return
    
    print("=" * 80)
    print(f"Model Comparison Results")
    print("=" * 80)
    print(f"Base Model: {data.get('base_model', 'Unknown')}")
    print(f"Fine-tuned Model: {data.get('model_path', 'Unknown')}")
    print(f"Total Comparisons: {len(comparisons)}")
    print("=" * 80)
    print()
    
    for i, comp in enumerate(comparisons[:max_examples]):
        print(f"\n{'='*80}")
        print(f"Example {comp.get('sample_id', i)}")
        print(f"{'='*80}")
        
        print(f"\n📝 Input:")
        input_text = comp.get('input', '')
        if input_text:
            print(f"   {input_text[:200]}{'...' if len(input_text) > 200 else ''}")
        else:
            print("   (Empty - check dataset format)")
        
        print(f"\n📋 Reference (Expected Response):")
        reference = comp.get('reference', '')
        if reference:
            print(f"   {reference[:300]}{'...' if len(reference) > 300 else ''}")
        else:
            print("   (Empty)")
        
        print(f"\n🤖 Base Model Response:")
        base_resp = comp.get('base_response', '')
        if base_resp:
            print(f"   {base_resp[:300]}{'...' if len(base_resp) > 300 else ''}")
        else:
            print("   (Empty)")
        
        print(f"\n✨ Fine-tuned Model Response:")
        ft_resp = comp.get('finetuned_response', '')
        if ft_resp:
            print(f"   {ft_resp[:300]}{'...' if len(ft_resp) > 300 else ''}")
        else:
            print("   (Empty)")
        
        # Metrics
        print(f"\n📊 Metrics:")
        base_rouge = comp.get('base_rouge', {})
        ft_rouge = comp.get('finetuned_rouge', {})
        print(f"   Base ROUGE-1: {base_rouge.get('rouge1', 0):.4f} | Fine-tuned: {ft_rouge.get('rouge1', 0):.4f}")
        print(f"   Base BLEU: {comp.get('base_bleu', 0):.4f} | Fine-tuned: {comp.get('finetuned_bleu', 0):.4f}")
        
        print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python view_comparisons.py <results_json> [max_examples]")
        print("\nExample:")
        print("  python view_comparisons.py evaluation/results/comparison_results.json 5")
        sys.exit(1)
    
    json_path = sys.argv[1]
    max_examples = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    
    if not Path(json_path).exists():
        print(f"Error: File not found: {json_path}")
        sys.exit(1)
    
    view_comparisons(json_path, max_examples)

