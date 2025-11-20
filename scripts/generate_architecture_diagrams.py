#!/usr/bin/env python3
"""
Generate visual architecture diagrams for FYP presentation.
Creates publication-quality figures for LoRA and training infrastructure.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
import numpy as np
import os
from pathlib import Path

# Set style for professional-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300

# Create output directory
output_dir = Path(__file__).parent.parent / "docs" / "figures"
output_dir.mkdir(parents=True, exist_ok=True)


def create_lora_architecture_diagram():
    """Create LoRA architecture diagram showing base model + adapters."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(6, 9.5, 'LoRA Fine-tuning Architecture', 
            fontsize=16, fontweight='bold', ha='center')
    
    # Base Model Box
    base_box = FancyBboxPatch((0.5, 5), 4, 3.5, 
                              boxstyle="round,pad=0.1", 
                              facecolor='lightblue', 
                              edgecolor='navy', linewidth=2)
    ax.add_patch(base_box)
    ax.text(2.5, 7.8, 'Qwen 2.5-14B', fontsize=12, fontweight='bold', ha='center')
    ax.text(2.5, 7.3, 'Base Model', fontsize=10, ha='center')
    ax.text(2.5, 6.9, '(14.2B parameters)', fontsize=9, ha='center', style='italic')
    ax.text(2.5, 6.5, '❄ Frozen', fontsize=10, ha='center', color='blue')
    ax.text(2.5, 6.0, '40 Transformer Layers', fontsize=8, ha='center')
    ax.text(2.5, 5.6, 'Hidden: 5120', fontsize=8, ha='center')
    ax.text(2.5, 5.2, 'Attention Heads: 40', fontsize=8, ha='center')
    
    # LoRA Adapters Box
    lora_box = FancyBboxPatch((5.5, 5), 4, 3.5, 
                              boxstyle="round,pad=0.1", 
                              facecolor='lightgreen', 
                              edgecolor='darkgreen', linewidth=2)
    ax.add_patch(lora_box)
    ax.text(7.5, 7.8, 'LoRA Adapters', fontsize=12, fontweight='bold', ha='center')
    ax.text(7.5, 7.3, '(14M parameters)', fontsize=10, ha='center')
    ax.text(7.5, 6.9, '0.1% of total', fontsize=9, ha='center', style='italic')
    ax.text(7.5, 6.5, '🔥 Trainable', fontsize=10, ha='center', color='red')
    ax.text(7.5, 6.0, 'Rank r = 8', fontsize=9, ha='center')
    ax.text(7.5, 5.7, 'Alpha α = 16', fontsize=9, ha='center')
    ax.text(7.5, 5.4, 'Target: Q,K,V,O + MLP', fontsize=8, ha='center')
    
    # Arrow from Base to LoRA
    arrow1 = FancyArrowPatch((4.5, 6.75), (5.5, 6.75),
                            arrowstyle='->', mutation_scale=20, 
                            linewidth=2, color='black')
    ax.add_patch(arrow1)
    ax.text(5, 7, '+', fontsize=14, ha='center', fontweight='bold')
    
    # Fine-tuned Model Box
    finetuned_box = FancyBboxPatch((3.5, 1.5), 5, 2.5, 
                                   boxstyle="round,pad=0.1", 
                                   facecolor='gold', 
                                   edgecolor='darkorange', linewidth=2)
    ax.add_patch(finetuned_box)
    ax.text(6, 3.3, 'Fine-tuned Model', fontsize=12, fontweight='bold', ha='center')
    ax.text(6, 2.9, 'Mental Health Counseling', fontsize=10, ha='center')
    ax.text(6, 2.5, '✓ Specialized for counseling', fontsize=9, ha='center', color='green')
    ax.text(6, 2.1, '✓ Maintains base knowledge', fontsize=9, ha='center', color='green')
    ax.text(6, 1.7, '✓ 14MB checkpoint size', fontsize=9, ha='center', color='green')
    
    # Arrow from combined to fine-tuned
    arrow2 = FancyArrowPatch((6, 5), (6, 4),
                            arrowstyle='->', mutation_scale=20, 
                            linewidth=2, color='black')
    ax.add_patch(arrow2)
    ax.text(6.5, 4.5, 'Training', fontsize=10, ha='center')
    
    # Efficiency box
    efficiency_box = FancyBboxPatch((10, 6), 1.8, 2, 
                                    boxstyle="round,pad=0.05", 
                                    facecolor='lightyellow', 
                                    edgecolor='orange', linewidth=1.5)
    ax.add_patch(efficiency_box)
    ax.text(10.9, 7.7, 'Efficiency', fontsize=9, fontweight='bold', ha='center')
    ax.text(10.9, 7.3, '140×', fontsize=11, ha='center', color='red', fontweight='bold')
    ax.text(10.9, 7.0, 'fewer params', fontsize=7, ha='center')
    ax.text(10.9, 6.6, '4×', fontsize=11, ha='center', color='red', fontweight='bold')
    ax.text(10.9, 6.3, 'less memory', fontsize=7, ha='center')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'lora_architecture.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Created: lora_architecture.png")


def create_parameter_efficiency_chart():
    """Create bar chart showing parameter efficiency."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = ['Full\nFine-tuning', 'LoRA\n(r=8)']
    params = [14.2e9, 14e6]  # in millions
    colors = ['#ff6b6b', '#51cf66']
    
    bars = ax.bar(methods, params, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for i, (bar, param) in enumerate(zip(bars, params)):
        height = bar.get_height()
        if param > 1e9:
            label = f'{param/1e9:.1f}B'
        else:
            label = f'{param/1e6:.0f}M'
        ax.text(bar.get_x() + bar.get_width()/2., height,
                label, ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Number of Trainable Parameters', fontsize=12, fontweight='bold')
    ax.set_title('Parameter Efficiency: Full Fine-tuning vs LoRA', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add efficiency label
    ax.text(0.5, params[0]/2, '140× reduction', 
            fontsize=12, ha='center', 
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'parameter_efficiency.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Created: parameter_efficiency.png")


def create_memory_breakdown():
    """Create memory usage breakdown chart."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Memory components
    components = ['Base Model\n(4-bit)', 'LoRA\nAdapters', 'Optimizer\nStates', 
                  'Activations\n(checkpointed)', 'Gradients\n(BF16)', 'Buffers']
    sizes = [1.8, 0.5, 1.0, 2.5, 0.5, 0.7]
    colors = ['#4c72b0', '#55a868', '#c44e52', '#8172b2', '#ccb974', '#64b5cd']
    
    # Pie chart
    wedges, texts, autotexts = ax1.pie(sizes, labels=components, autopct='%1.1f%%',
                                         colors=colors, startangle=90,
                                         textprops={'fontsize': 9})
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    ax1.set_title('Memory Distribution per GPU', fontsize=12, fontweight='bold')
    
    # Bar chart with utilization
    y_pos = np.arange(len(components))
    ax2.barh(y_pos, sizes, color=colors, edgecolor='black', linewidth=1)
    
    # Add size labels
    for i, size in enumerate(sizes):
        ax2.text(size + 0.1, i, f'{size:.1f} GB', 
                va='center', fontsize=9, fontweight='bold')
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(components, fontsize=9)
    ax2.set_xlabel('Memory Usage (GB)', fontsize=11, fontweight='bold')
    ax2.set_title('Memory Breakdown (Total: 7.0 GB / 11 GB)', 
                  fontsize=12, fontweight='bold')
    ax2.axvline(x=11, color='red', linestyle='--', linewidth=2, label='GPU Limit (11GB)')
    ax2.axvline(x=7, color='green', linestyle='--', linewidth=2, label='Used (7GB)')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'memory_breakdown.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Created: memory_breakdown.png")


def create_distributed_training_diagram():
    """Create distributed training architecture diagram."""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Title
    ax.text(7, 11.5, 'Distributed Data Parallel (DDP) Training', 
            fontsize=16, fontweight='bold', ha='center')
    
    # Data loader
    data_box = FancyBboxPatch((5, 10), 4, 0.8, 
                              boxstyle="round,pad=0.05", 
                              facecolor='lightcoral', 
                              edgecolor='darkred', linewidth=2)
    ax.add_patch(data_box)
    ax.text(7, 10.4, 'Dataset: Mental Health Counseling (~50K examples)', 
            fontsize=10, ha='center', fontweight='bold')
    
    # Data distribution
    for i in range(8):
        x = 1 + i * 1.5
        y = 9
        arrow = FancyArrowPatch((7, 10), (x + 0.5, y + 0.5),
                               arrowstyle='->', mutation_scale=15, 
                               linewidth=1, color='gray', alpha=0.5)
        ax.add_patch(arrow)
    
    ax.text(7, 9.2, 'Data Split', fontsize=9, ha='center', style='italic')
    
    # GPU boxes
    for i in range(8):
        x = 1 + i * 1.5
        y = 6
        
        # GPU box
        gpu_box = FancyBboxPatch((x, y), 1.2, 2.5, 
                                 boxstyle="round,pad=0.05", 
                                 facecolor='lightblue', 
                                 edgecolor='navy', linewidth=1.5)
        ax.add_patch(gpu_box)
        
        # GPU label
        ax.text(x + 0.6, y + 2.2, f'GPU {i}', fontsize=8, 
                ha='center', fontweight='bold')
        ax.text(x + 0.6, y + 1.9, f'Rank {i}', fontsize=7, ha='center')
        ax.text(x + 0.6, y + 1.6, '11GB', fontsize=7, ha='center')
        ax.text(x + 0.6, y + 1.3, '7GB used', fontsize=6, ha='center', color='green')
        
        # Model replica
        model_rect = Rectangle((x + 0.15, y + 0.3), 0.9, 0.8, 
                               facecolor='lightgreen', edgecolor='darkgreen')
        ax.add_patch(model_rect)
        ax.text(x + 0.6, y + 0.7, 'Model', fontsize=6, ha='center')
        ax.text(x + 0.6, y + 0.5, '+ LoRA', fontsize=6, ha='center')
    
    # Forward pass arrows
    for i in range(8):
        x = 1 + i * 1.5
        arrow = FancyArrowPatch((x + 0.6, 6), (x + 0.6, 5.2),
                               arrowstyle='->', mutation_scale=15, 
                               linewidth=1.5, color='blue')
        ax.add_patch(arrow)
    
    ax.text(7, 5.5, 'Forward Pass (local batch)', fontsize=9, ha='center', color='blue')
    
    # Loss boxes
    for i in range(8):
        x = 1 + i * 1.5
        y = 4.5
        loss_circle = Circle((x + 0.6, y), 0.3, facecolor='orange', edgecolor='darkorange')
        ax.add_patch(loss_circle)
        ax.text(x + 0.6, y, f'L{i}', fontsize=7, ha='center', fontweight='bold')
    
    # Backward pass arrows
    for i in range(8):
        x = 1 + i * 1.5
        arrow = FancyArrowPatch((x + 0.6, 4.2), (x + 0.6, 3.4),
                               arrowstyle='->', mutation_scale=15, 
                               linewidth=1.5, color='red')
        ax.add_patch(arrow)
    
    ax.text(7, 3.7, 'Backward Pass (compute gradients)', fontsize=9, ha='center', color='red')
    
    # Gradient boxes
    for i in range(8):
        x = 1 + i * 1.5
        y = 2.5
        grad_rect = Rectangle((x + 0.15, y), 0.9, 0.6, 
                              facecolor='plum', edgecolor='purple')
        ax.add_patch(grad_rect)
        ax.text(x + 0.6, y + 0.3, f'∇{i}', fontsize=8, ha='center', fontweight='bold')
    
    # Convergence arrows to AllReduce
    for i in range(8):
        x = 1 + i * 1.5
        arrow = FancyArrowPatch((x + 0.6, 2.5), (7, 1.5),
                               arrowstyle='->', mutation_scale=15, 
                               linewidth=1, color='purple', alpha=0.5)
        ax.add_patch(arrow)
    
    # AllReduce box
    allreduce_box = FancyBboxPatch((5.5, 0.8), 3, 0.8, 
                                   boxstyle="round,pad=0.05", 
                                   facecolor='yellow', 
                                   edgecolor='orange', linewidth=2)
    ax.add_patch(allreduce_box)
    ax.text(7, 1.2, 'AllReduce (NCCL): Average Gradients', 
            fontsize=10, ha='center', fontweight='bold')
    
    # Divergence arrows from AllReduce
    for i in range(8):
        x = 1 + i * 1.5
        arrow = FancyArrowPatch((7, 0.8), (x + 0.6, 0.2),
                               arrowstyle='->', mutation_scale=15, 
                               linewidth=1, color='green', alpha=0.5)
        ax.add_patch(arrow)
    
    # Config box
    config_box = FancyBboxPatch((0.2, 0.2), 3, 1.8, 
                                boxstyle="round,pad=0.05", 
                                facecolor='lightyellow', 
                                edgecolor='goldenrod', linewidth=1.5)
    ax.add_patch(config_box)
    ax.text(1.7, 1.8, 'Configuration', fontsize=9, fontweight='bold', ha='center')
    ax.text(1.7, 1.5, '• Batch/GPU: 1', fontsize=7, ha='center')
    ax.text(1.7, 1.25, '• Grad Accum: 16', fontsize=7, ha='center')
    ax.text(1.7, 1.0, '• Effective Batch: 128', fontsize=7, ha='center')
    ax.text(1.7, 0.75, '• Sync: Every 16 steps', fontsize=7, ha='center')
    ax.text(1.7, 0.5, '• Backend: NCCL', fontsize=7, ha='center')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'distributed_training.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Created: distributed_training.png")


def create_optimization_techniques():
    """Create visualization of memory optimization techniques."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(6, 9.5, 'Memory Optimization Stack', 
            fontsize=16, fontweight='bold', ha='center')
    
    # Technique 1: 4-bit Quantization
    box1 = FancyBboxPatch((0.5, 7), 11, 1.8, 
                          boxstyle="round,pad=0.1", 
                          facecolor='#e3f2fd', 
                          edgecolor='#1976d2', linewidth=2)
    ax.add_patch(box1)
    ax.text(1, 8.5, '① 4-bit Quantization (NF4)', fontsize=11, fontweight='bold')
    ax.text(1, 8.1, '• Reduces FP32 (32-bit) → 4-bit', fontsize=9)
    ax.text(1, 7.7, '• Double quantization for constants', fontsize=9)
    ax.text(1, 7.3, '• Compute in BF16, store in 4-bit', fontsize=9)
    
    # Savings arrow
    ax.annotate('', xy=(10.5, 8), xytext=(9, 8),
                arrowprops=dict(arrowstyle='->', lw=2, color='green'))
    ax.text(10.8, 8, '87.5%\nsaved', fontsize=9, ha='center', 
            bbox=dict(boxstyle='round', facecolor='lightgreen'))
    
    # Technique 2: Gradient Checkpointing
    box2 = FancyBboxPatch((0.5, 4.5), 11, 1.8, 
                          boxstyle="round,pad=0.1", 
                          facecolor='#f3e5f5', 
                          edgecolor='#7b1fa2', linewidth=2)
    ax.add_patch(box2)
    ax.text(1, 6, '② Gradient Checkpointing', fontsize=11, fontweight='bold')
    ax.text(1, 5.6, '• Store only layer boundaries', fontsize=9)
    ax.text(1, 5.2, '• Recompute activations during backward', fontsize=9)
    ax.text(1, 4.8, '• Trade: +20% time for -40% memory', fontsize=9)
    
    ax.annotate('', xy=(10.5, 5.5), xytext=(9, 5.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='green'))
    ax.text(10.8, 5.5, '40%\nsaved', fontsize=9, ha='center', 
            bbox=dict(boxstyle='round', facecolor='lightgreen'))
    
    # Technique 3: Mixed Precision
    box3 = FancyBboxPatch((0.5, 2), 11, 1.8, 
                          boxstyle="round,pad=0.1", 
                          facecolor='#fff3e0', 
                          edgecolor='#e65100', linewidth=2)
    ax.add_patch(box3)
    ax.text(1, 3.5, '③ Mixed Precision (BF16)', fontsize=11, fontweight='bold')
    ax.text(1, 3.1, '• Forward/backward in BF16 (16-bit)', fontsize=9)
    ax.text(1, 2.7, '• Master weights in FP32', fontsize=9)
    ax.text(1, 2.3, '• Same range as FP32, better stability', fontsize=9)
    
    ax.annotate('', xy=(10.5, 3), xytext=(9, 3),
                arrowprops=dict(arrowstyle='->', lw=2, color='green'))
    ax.text(10.8, 3, '50%\nsaved', fontsize=9, ha='center', 
            bbox=dict(boxstyle='round', facecolor='lightgreen'))
    
    # Result box
    result_box = FancyBboxPatch((2, 0.2), 8, 1.2, 
                                boxstyle="round,pad=0.1", 
                                facecolor='lightgreen', 
                                edgecolor='darkgreen', linewidth=3)
    ax.add_patch(result_box)
    ax.text(6, 1.1, 'Final Result: 7 GB / 11 GB per GPU', 
            fontsize=12, ha='center', fontweight='bold')
    ax.text(6, 0.7, '✓ 64% utilization  ✓ Safe for RTX 2080 Ti  ✓ 36% safety margin', 
            fontsize=9, ha='center', color='darkgreen')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'optimization_techniques.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Created: optimization_techniques.png")


def create_comparison_chart():
    """Create comparison chart: Full fine-tuning vs LoRA."""
    metrics = ['Training\nTime', 'Memory\nper GPU', 'Checkpoint\nSize', 'Training\nCost']
    full_ft = [8, 28, 28000, 200]  # hours, GB, MB, USD
    lora = [2, 7, 14, 7]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    axes = [ax1, ax2, ax3, ax4]
    units = ['hours', 'GB', 'MB', 'USD']
    colors_ft = ['#ff6b6b', '#ff6b6b', '#ff6b6b', '#ff6b6b']
    colors_lora = ['#51cf66', '#51cf66', '#51cf66', '#51cf66']
    
    for idx, (ax, metric, full, lora_val, unit) in enumerate(zip(axes, metrics, full_ft, lora, units)):
        bars = ax.bar(['Full FT', 'LoRA'], [full, lora_val], 
                     color=[colors_ft[idx], colors_lora[idx]], 
                     edgecolor='black', linewidth=1.5)
        
        # Add values on bars
        for bar, val in zip(bars, [full, lora_val]):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val}{unit}' if idx != 2 else f'{val/1000:.0f}GB' if val > 1000 else f'{val}MB',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Add improvement factor
        improvement = full / lora_val
        ax.text(0.5, max(full, lora_val) * 0.5, 
               f'{improvement:.1f}× faster' if idx == 0 else f'{improvement:.1f}× less',
               ha='center', fontsize=11, 
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        ax.set_ylabel(unit.upper(), fontsize=11, fontweight='bold')
        ax.set_title(metric.replace('\n', ' '), fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle('Full Fine-tuning vs LoRA: Comprehensive Comparison', 
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_chart.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Created: comparison_chart.png")


def create_gpu_hardware_diagram():
    """Create GPU hardware setup diagram."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(7, 9.5, 'Hardware Setup: 8× RTX 2080 Ti GPUs', 
            fontsize=16, fontweight='bold', ha='center')
    
    # GPU cards - 2 rows of 4
    gpu_specs = {
        'Model': 'NVIDIA GeForce RTX 2080 Ti',
        'VRAM': '11 GB GDDR6',
        'Compute': 'SM 7.5 (Turing)',
        'Cores': '4352 CUDA cores',
        'Tensor': 'Tensor Cores: Yes',
        'Bandwidth': '616 GB/s'
    }
    
    for row in range(2):
        for col in range(4):
            gpu_num = row * 4 + col
            x = 1 + col * 3
            y = 5.5 - row * 2.5
            
            # GPU box
            gpu_box = FancyBboxPatch((x, y), 2.5, 2, 
                                     boxstyle="round,pad=0.1", 
                                     facecolor='lightblue', 
                                     edgecolor='navy', linewidth=2)
            ax.add_patch(gpu_box)
            
            # GPU info
            ax.text(x + 1.25, y + 1.7, f'GPU {gpu_num}', 
                   fontsize=11, fontweight='bold', ha='center')
            ax.text(x + 1.25, y + 1.4, 'RTX 2080 Ti', 
                   fontsize=9, ha='center')
            ax.text(x + 1.25, y + 1.1, '11 GB VRAM', 
                   fontsize=8, ha='center', color='green', fontweight='bold')
            ax.text(x + 1.25, y + 0.8, 'Used: 7 GB', 
                   fontsize=7, ha='center')
            ax.text(x + 1.25, y + 0.5, 'Free: 4 GB', 
                   fontsize=7, ha='center', color='blue')
            
            # Memory bar
            mem_rect_used = Rectangle((x + 0.3, y + 0.1), 1.4 * 0.64, 0.2, 
                                     facecolor='orange', edgecolor='black')
            mem_rect_free = Rectangle((x + 0.3 + 1.4 * 0.64, y + 0.1), 1.4 * 0.36, 0.2, 
                                     facecolor='lightgray', edgecolor='black')
            ax.add_patch(mem_rect_used)
            ax.add_patch(mem_rect_free)
    
    # Connection lines (PCIe/NVLink)
    # Horizontal connections
    for row in range(2):
        y = 6.5 - row * 2.5
        ax.plot([3.5, 10], [y, y], 'k-', linewidth=2, alpha=0.5)
    
    # Vertical connections
    for col in range(4):
        x = 2.25 + col * 3
        ax.plot([x, x], [3, 7], 'k-', linewidth=2, alpha=0.5)
    
    ax.text(7, 4.8, 'PCIe Bus / NVLink', fontsize=9, ha='center', 
           style='italic', color='gray')
    
    # Stats box
    stats_box = FancyBboxPatch((0.5, 0.2), 6, 1.8, 
                               boxstyle="round,pad=0.1", 
                               facecolor='lightyellow', 
                               edgecolor='orange', linewidth=2)
    ax.add_patch(stats_box)
    ax.text(3.5, 1.8, 'Total Resources', fontsize=11, fontweight='bold', ha='center')
    ax.text(3.5, 1.5, '• Total VRAM: 88 GB', fontsize=9, ha='center')
    ax.text(3.5, 1.2, '• Total CUDA Cores: 34,816', fontsize=9, ha='center')
    ax.text(3.5, 0.9, '• Total Bandwidth: 4.9 TB/s', fontsize=9, ha='center')
    ax.text(3.5, 0.6, '• Power: 8 × 250W = 2000W', fontsize=9, ha='center')
    
    # Performance box
    perf_box = FancyBboxPatch((7.5, 0.2), 6, 1.8, 
                              boxstyle="round,pad=0.1", 
                              facecolor='lightgreen', 
                              edgecolor='darkgreen', linewidth=2)
    ax.add_patch(perf_box)
    ax.text(10.5, 1.8, 'Training Performance', fontsize=11, fontweight='bold', ha='center')
    ax.text(10.5, 1.5, '• Time per step: 3-5 sec', fontsize=9, ha='center')
    ax.text(10.5, 1.2, '• Samples/sec: 25-40', fontsize=9, ha='center')
    ax.text(10.5, 0.9, '• Epoch time: ~30 min', fontsize=9, ha='center')
    ax.text(10.5, 0.6, '• Total training: 1.5-2 hrs', fontsize=9, ha='center')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'gpu_hardware.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Created: gpu_hardware.png")


def main():
    """Generate all diagrams."""
    print("\n" + "="*60)
    print("Generating Technical Architecture Diagrams for FYP")
    print("="*60 + "\n")
    
    print("Creating diagrams...")
    create_lora_architecture_diagram()
    create_parameter_efficiency_chart()
    create_memory_breakdown()
    create_distributed_training_diagram()
    create_optimization_techniques()
    create_comparison_chart()
    create_gpu_hardware_diagram()
    
    print("\n" + "="*60)
    print(f"✓ All diagrams created successfully!")
    print(f"✓ Output directory: {output_dir}")
    print(f"✓ Total files: 7 PNG images")
    print("="*60 + "\n")
    
    print("Generated files:")
    print("  1. lora_architecture.png       - LoRA architecture overview")
    print("  2. parameter_efficiency.png    - Parameter comparison chart")
    print("  3. memory_breakdown.png        - Memory usage breakdown")
    print("  4. distributed_training.png    - DDP training architecture")
    print("  5. optimization_techniques.png - Memory optimization stack")
    print("  6. comparison_chart.png        - Full FT vs LoRA comparison")
    print("  7. gpu_hardware.png           - GPU hardware setup")
    print("\n✓ Ready for your presentation!\n")


if __name__ == "__main__":
    main()

