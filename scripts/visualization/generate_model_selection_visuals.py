#!/usr/bin/env python3
"""
Generate visualizations for Model Selection Justification presentation.
Creates professional charts and tables for slides.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10

# Output directory
OUTPUT_DIR = Path("present_png")
OUTPUT_DIR.mkdir(exist_ok=True)


def generate_cost_comparison():
    """Generate 5-year cost comparison chart."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Data
    years = [1, 2, 3, 4, 5]
    gpt4_costs = [180000, 360000, 540000, 720000, 900000]
    claude_costs = [270000, 540000, 810000, 1080000, 1350000]
    qwen_costs = [3000, 5000, 7000, 9000, 11000]
    
    # Left: Line chart
    ax1.plot(years, gpt4_costs, 'o-', linewidth=3, markersize=8, label='GPT-4', color='#FF6B6B')
    ax1.plot(years, claude_costs, 's-', linewidth=3, markersize=8, label='Claude 3', color='#4ECDC4')
    ax1.plot(years, qwen_costs, '^-', linewidth=3, markersize=8, label='Qwen 2.5 (Self-hosted)', color='#45B7D1')
    ax1.set_xlabel('Years', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Cumulative Cost (USD)', fontsize=12, fontweight='bold')
    ax1.set_title('5-Year Cost Comparison\n(1,000 Users)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    
    # Right: Bar chart (Year 5)
    models = ['Qwen 2.5\n(Self-hosted)', 'GPT-4', 'Claude 3']
    costs_5y = [11000, 900000, 1350000]
    colors = ['#45B7D1', '#FF6B6B', '#4ECDC4']
    
    bars = ax2.bar(models, costs_5y, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Total 5-Year Cost (USD)', fontsize=12, fontweight='bold')
    ax2.set_title('Total 5-Year Cost Comparison', fontsize=14, fontweight='bold')
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    
    # Add cost labels on bars
    for bar, cost in zip(bars, costs_5y):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'${cost/1000:.0f}K',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add savings annotation
    ax2.annotate('98.8% Savings!', xy=(0, 11000), xytext=(0.5, 500000),
                arrowprops=dict(arrowstyle='->', lw=2, color='green'),
                fontsize=12, fontweight='bold', color='green',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "model_cost_comparison.png", bbox_inches='tight')
    print("✅ Generated: model_cost_comparison.png")
    plt.close()


def generate_performance_radar():
    """Generate radar chart comparing model capabilities."""
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Categories
    categories = ['Cost\nEffectiveness', 'Privacy &\nSecurity', 'Customization', 
                  'Performance', 'Multilingual', 'Instruction\nFollowing']
    num_vars = len(categories)
    
    # Data (scores out of 10)
    qwen_scores = [10, 10, 10, 8, 10, 9]
    gpt4_scores = [2, 3, 4, 10, 8, 10]
    llama_scores = [10, 10, 10, 7, 5, 8]
    
    # Angles for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    qwen_scores += qwen_scores[:1]
    gpt4_scores += gpt4_scores[:1]
    llama_scores += llama_scores[:1]
    angles += angles[:1]
    
    # Plot
    ax.plot(angles, qwen_scores, 'o-', linewidth=3, label='Qwen 2.5', color='#45B7D1')
    ax.fill(angles, qwen_scores, alpha=0.25, color='#45B7D1')
    
    ax.plot(angles, gpt4_scores, 's-', linewidth=3, label='GPT-4', color='#FF6B6B')
    ax.fill(angles, gpt4_scores, alpha=0.25, color='#FF6B6B')
    
    ax.plot(angles, llama_scores, '^-', linewidth=2, label='LLaMA 3.1', color='#95E1D3', alpha=0.7)
    ax.fill(angles, llama_scores, alpha=0.15, color='#95E1D3')
    
    # Customize
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 10)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(['2', '4', '6', '8', '10'], fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title('Model Capability Comparison\n(Weighted for Mental Health Use Case)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "model_capability_radar.png", bbox_inches='tight')
    print("✅ Generated: model_capability_radar.png")
    plt.close()


def generate_model_size_tradeoffs():
    """Generate 7B vs 14B comparison chart."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Data
    metrics = ['Inference\nSpeed', 'Memory\nUsage', 'Training\nTime', 
               'Response\nQuality', 'Complex\nReasoning', 'Deployment\nEase']
    
    # Normalized scores (higher is better for all)
    qwen_7b = [10, 10, 10, 7, 6, 10]  # Fast, efficient, but lower quality
    qwen_14b = [5, 6, 5, 9, 9, 7]     # Slower, more memory, but better quality
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, qwen_7b, width, label='Qwen 2.5-7B', 
                    color='#45B7D1', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax1.bar(x + width/2, qwen_14b, width, label='Qwen 2.5-14B', 
                    color='#F38181', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax1.set_xlabel('Metrics', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Score (0-10, higher is better)', fontsize=12, fontweight='bold')
    ax1.set_title('Qwen 2.5: 7B vs 14B Trade-offs', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, fontsize=10)
    ax1.legend(fontsize=11)
    ax1.set_ylim(0, 11)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Right: Use case recommendations
    use_cases = {
        'Qwen 2.5-7B': [
            'Real-time chat',
            'High volume users',
            'Mobile/edge deployment',
            'Simple queries',
            'Proof-of-concept'
        ],
        'Qwen 2.5-14B': [
            'Complex counseling',
            'Quality-critical',
            'Low volume/research',
            'Offline processing',
            'Nuanced understanding'
        ]
    }
    
    ax2.axis('off')
    ax2.set_title('Use Case Recommendations', fontsize=14, fontweight='bold', pad=20)
    
    # 7B recommendations
    y_pos = 0.85
    ax2.text(0.1, y_pos, 'Qwen 2.5-7B:', fontsize=13, fontweight='bold', 
             color='#45B7D1', transform=ax2.transAxes)
    y_pos -= 0.08
    for use_case in use_cases['Qwen 2.5-7B']:
        ax2.text(0.12, y_pos, f'• {use_case}', fontsize=11, 
                transform=ax2.transAxes)
        y_pos -= 0.07
    
    # 14B recommendations
    y_pos -= 0.05
    ax2.text(0.1, y_pos, 'Qwen 2.5-14B:', fontsize=13, fontweight='bold', 
             color='#F38181', transform=ax2.transAxes)
    y_pos -= 0.08
    for use_case in use_cases['Qwen 2.5-14B']:
        ax2.text(0.12, y_pos, f'• {use_case}', fontsize=11, 
                transform=ax2.transAxes)
        y_pos -= 0.07
    
    # Add box around text
    ax2.add_patch(plt.Rectangle((0.08, 0.05), 0.84, 0.9, 
                                fill=False, edgecolor='gray', 
                                linewidth=2, transform=ax2.transAxes))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "model_size_tradeoffs.png", bbox_inches='tight')
    print("✅ Generated: model_size_tradeoffs.png")
    plt.close()


def generate_benchmark_comparison():
    """Generate benchmark comparison bar chart."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Data
    benchmarks = ['MMLU\n(Reasoning)', 'GSM8K\n(Math)', 'HumanEval\n(Code)', 
                  'CMMLU\n(Chinese)', 'BBH\n(Hard Tasks)']
    qwen_7b = [70.3, 82.1, 53.7, 74.8, 65.4]
    qwen_14b = [79.9, 87.9, 65.2, 83.1, 74.8]
    llama_8b = [69.4, 79.6, 48.1, 51.0, 63.5]
    mistral_7b = [62.5, 52.2, 40.2, 44.0, 56.7]
    gpt4 = [86.4, 92.0, 67.0, 71.0, 83.1]
    
    x = np.arange(len(benchmarks))
    width = 0.15
    
    ax.bar(x - 2*width, qwen_7b, width, label='Qwen 2.5-7B', 
           color='#45B7D1', alpha=0.8, edgecolor='black', linewidth=1)
    ax.bar(x - width, qwen_14b, width, label='Qwen 2.5-14B', 
           color='#F38181', alpha=0.8, edgecolor='black', linewidth=1)
    ax.bar(x, llama_8b, width, label='LLaMA 3.1-8B', 
           color='#95E1D3', alpha=0.8, edgecolor='black', linewidth=1)
    ax.bar(x + width, mistral_7b, width, label='Mistral 7B', 
           color='#FFBE76', alpha=0.8, edgecolor='black', linewidth=1)
    ax.bar(x + 2*width, gpt4, width, label='GPT-4', 
           color='#FF6B6B', alpha=0.8, edgecolor='black', linewidth=1)
    
    ax.set_xlabel('Benchmarks', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance on Standard Benchmarks', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks, fontsize=10)
    ax.legend(fontsize=10, loc='upper left')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add annotation
    ax.text(3, 85, 'Qwen excels in Chinese\n(CMMLU)', fontsize=10, 
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3),
            ha='center')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "benchmark_comparison.png", bbox_inches='tight')
    print("✅ Generated: benchmark_comparison.png")
    plt.close()


def generate_privacy_comparison():
    """Generate privacy & security comparison."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Data
    criteria = [
        'Data Control',
        'GDPR Compliance',
        'HIPAA Ready',
        'On-Premise\nDeployment',
        'No External API',
        'Audit Trail',
        'Custom Safety\nFilters',
        'Data Retention\nControl'
    ]
    
    qwen_scores = [10, 10, 10, 10, 10, 10, 10, 10]
    gpt4_scores = [2, 5, 3, 0, 0, 4, 5, 3]
    
    x = np.arange(len(criteria))
    width = 0.35
    
    bars1 = ax.barh(x + width/2, qwen_scores, width, label='Qwen 2.5 (Self-hosted)', 
                    color='#45B7D1', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.barh(x - width/2, gpt4_scores, width, label='GPT-4 (API)', 
                    color='#FF6B6B', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_yticks(x)
    ax.set_yticklabels(criteria, fontsize=11)
    ax.set_xlabel('Score (0-10, higher is better)', fontsize=12, fontweight='bold')
    ax.set_title('Privacy & Security Comparison\n(Critical for Mental Health Data)', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.set_xlim(0, 11)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add perfect score annotation
    ax.text(10.5, 7, '✓', fontsize=20, color='green', fontweight='bold')
    ax.text(10.5, 6, '✓', fontsize=20, color='green', fontweight='bold')
    ax.text(10.5, 5, '✓', fontsize=20, color='green', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "privacy_comparison.png", bbox_inches='tight')
    print("✅ Generated: privacy_comparison.png")
    plt.close()


def generate_decision_matrix():
    """Generate decision matrix heatmap."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Data
    criteria = [
        'Cost Effectiveness (15%)',
        'Privacy & Security (20%)',
        'Customization (20%)',
        'Performance (15%)',
        'Multilingual (10%)',
        'Instruction-Following (10%)',
        'Ease of Deployment (5%)',
        'Academic Suitability (5%)'
    ]
    
    models = ['Qwen 2.5', 'LLaMA 3.1', 'Mistral', 'GPT-4', 'Claude']
    
    scores = [
        [10, 10, 10, 2, 2],    # Cost
        [10, 10, 10, 3, 3],    # Privacy
        [10, 10, 9, 4, 2],     # Customization
        [8, 7, 7, 10, 9],      # Performance
        [10, 5, 4, 8, 8],      # Multilingual
        [9, 8, 8, 10, 10],     # Instruction
        [8, 8, 9, 10, 10],     # Deployment
        [10, 10, 10, 5, 5]     # Academic
    ]
    
    # Create heatmap
    im = ax.imshow(scores, cmap='RdYlGn', aspect='auto', vmin=0, vmax=10)
    
    # Set ticks
    ax.set_xticks(np.arange(len(models)))
    ax.set_yticks(np.arange(len(criteria)))
    ax.set_xticklabels(models, fontsize=12, fontweight='bold')
    ax.set_yticklabels(criteria, fontsize=11)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Score (0-10)', fontsize=11, fontweight='bold')
    
    # Add text annotations
    for i in range(len(criteria)):
        for j in range(len(models)):
            text = ax.text(j, i, scores[i][j],
                          ha="center", va="center", color="black", 
                          fontsize=11, fontweight='bold')
    
    ax.set_title('Decision Matrix: Model Selection Scores\n(Weighted for Mental Health Chatbot)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "decision_matrix.png", bbox_inches='tight')
    print("✅ Generated: decision_matrix.png")
    plt.close()


def generate_weighted_scores():
    """Generate weighted final scores bar chart."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Data (weighted scores out of 10)
    models = ['Qwen 2.5', 'LLaMA 3.1', 'Mistral', 'GPT-4', 'Claude']
    scores = [9.35, 8.85, 8.55, 5.65, 5.35]
    colors = ['#45B7D1', '#95E1D3', '#FFBE76', '#FF6B6B', '#4ECDC4']
    
    bars = ax.barh(models, scores, color=colors, alpha=0.8, 
                   edgecolor='black', linewidth=2)
    
    # Add score labels
    for i, (bar, score) in enumerate(zip(bars, scores)):
        width = bar.get_width()
        ax.text(width + 0.1, bar.get_y() + bar.get_height()/2,
                f'{score}/10',
                ha='left', va='center', fontsize=12, fontweight='bold')
    
    # Highlight winner
    ax.text(9.35 + 0.8, 4, '★ Winner', fontsize=13, fontweight='bold', 
            color='green', va='center')
    
    ax.set_xlabel('Weighted Score (0-10)', fontsize=12, fontweight='bold')
    ax.set_title('Final Model Selection Scores\n(Weighted for Mental Health Use Case)', 
                 fontsize=14, fontweight='bold')
    ax.set_xlim(0, 11)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add threshold line
    ax.axvline(x=8.0, color='gray', linestyle='--', linewidth=2, alpha=0.5)
    ax.text(8.0, -0.7, 'Excellence Threshold\n(8.0)', ha='center', fontsize=9, 
            color='gray')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "weighted_scores.png", bbox_inches='tight')
    print("✅ Generated: weighted_scores.png")
    plt.close()


def generate_qwen_advantages_summary():
    """Generate summary infographic of Qwen advantages."""
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.axis('off')
    
    # Title
    ax.text(0.5, 0.95, 'Why Qwen 2.5 for Mental Health Counseling?', 
            fontsize=18, fontweight='bold', ha='center', 
            transform=ax.transAxes)
    
    # Seven key advantages with icons/colors
    advantages = [
        ('💰 Cost-Effective', '98% savings vs GPT-4\n$11K vs $900K (5 years)', '#45B7D1'),
        ('🔒 Privacy First', 'Complete data control\nOn-premise deployment', '#A8E6CF'),
        ('🎨 Full Customization', '400K+ mental health samples\nSpecialized via LoRA', '#FFD93D'),
        ('🌍 Multilingual', 'Native Chinese support\nServes HK population', '#FF6B9D'),
        ('🎯 Domain-Specific', 'Fine-tuned for counseling\nEvidence-based techniques', '#C7CEEA'),
        ('📊 Proven Performance', '7B: Fast & efficient\n14B: High quality', '#B4A7D6'),
        ('🎓 Academic Integrity', 'Open-source & reproducible\nDemonstrates tech skills', '#95E1D3')
    ]
    
    y_start = 0.85
    y_spacing = 0.12
    
    for i, (title, desc, color) in enumerate(advantages):
        y = y_start - i * y_spacing
        
        # Colored box
        ax.add_patch(plt.Rectangle((0.05, y - 0.05), 0.9, 0.1, 
                                   facecolor=color, alpha=0.3, 
                                   edgecolor='black', linewidth=2,
                                   transform=ax.transAxes))
        
        # Title
        ax.text(0.08, y + 0.02, title, fontsize=14, fontweight='bold', 
                transform=ax.transAxes, va='center')
        
        # Description
        ax.text(0.08, y - 0.02, desc, fontsize=10, 
                transform=ax.transAxes, va='center', style='italic')
    
    # Bottom conclusion
    ax.text(0.5, 0.02, 'Optimal choice for cost, privacy, and performance in mental health AI', 
            fontsize=11, fontweight='bold', ha='center', style='italic',
            transform=ax.transAxes,
            bbox=dict(boxstyle='round,pad=0.8', facecolor='lightgreen', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "qwen_advantages_summary.png", bbox_inches='tight')
    print("✅ Generated: qwen_advantages_summary.png")
    plt.close()


def main():
    """Generate all visualizations."""
    print("=" * 60)
    print("Generating Model Selection Justification Visualizations")
    print("=" * 60)
    
    generate_cost_comparison()
    generate_performance_radar()
    generate_model_size_tradeoffs()
    generate_benchmark_comparison()
    generate_privacy_comparison()
    generate_decision_matrix()
    generate_weighted_scores()
    generate_qwen_advantages_summary()
    
    print("=" * 60)
    print(f"✅ All visualizations saved to: {OUTPUT_DIR}")
    print("=" * 60)
    print("\nFiles generated:")
    print("1. model_cost_comparison.png - 5-year cost analysis")
    print("2. model_capability_radar.png - Capability comparison")
    print("3. model_size_tradeoffs.png - 7B vs 14B analysis")
    print("4. benchmark_comparison.png - Performance benchmarks")
    print("5. privacy_comparison.png - Privacy & security scores")
    print("6. decision_matrix.png - Decision matrix heatmap")
    print("7. weighted_scores.png - Final weighted scores")
    print("8. qwen_advantages_summary.png - Summary infographic")
    print("\nUse these in your FYP presentation slides!")


if __name__ == "__main__":
    main()

