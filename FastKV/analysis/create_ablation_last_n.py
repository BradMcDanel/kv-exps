#!/usr/bin/env python3
"""
Ablation study visualization for last_n_layers parameter in CLAA.
Shows LongBench accuracy vs number of last layers used for aggregation.
"""

import os
import matplotlib.pyplot as plt
import numpy as np

try:
    from .viz_utils import set_publication_style, METHOD_COLORS
except ImportError:
    import sys
    sys.path.append('.')
    from analysis.viz_utils import set_publication_style, METHOD_COLORS

def create_ablation_plot(output_dir):
    """Create ablation plot for last_n_layers parameter."""
    set_publication_style()
    plt.rcParams.update({
        'axes.labelsize': 26,
        'lines.linewidth': 3,
    })
    
    # Data for different keep rates
    n_layers_full = [1, 2, 3, 4]
    n_layers_partial = [1, 4]
    
    # Accuracy data
    acc_10 = [47.05, 47.14]  # 10% keep rate (n=1, n=4 only)
    acc_20 = [47.86, 48.12]  # 20% keep rate (n=1, n=4 only)
    acc_40 = [48.69, 48.71, 48.73, 48.72]  # 40% keep rate (all n)
    
    # Colors - shades of green
    colors = {
        '10%': '#90EE90',  # Light green
        '20%': '#32CD32',  # Lime green  
        '40%': '#228B22'   # Forest green
    }
    
    # Create single plot
    fig, ax = plt.subplots(1, 1, figsize=(6, 8), dpi=300)
    
    # Plot all three lines
    ax.plot(n_layers_partial, acc_10, color=colors['10%'], marker='D', markersize=8, 
            linewidth=3, markerfacecolor=colors['10%'], markeredgecolor='white', 
            markeredgewidth=1)
    
    ax.plot(n_layers_partial, acc_20, color=colors['20%'], marker='D', markersize=8, 
            linewidth=3, markerfacecolor=colors['20%'], markeredgecolor='white', 
            markeredgewidth=1)
    
    ax.plot(n_layers_full, acc_40, color=colors['40%'], marker='D', markersize=8, 
            linewidth=3, markerfacecolor=colors['40%'], markeredgecolor='white', 
            markeredgewidth=1)
    
    # Add keep rate annotations in the middle of each line
    # 10% line annotation (middle between n=1 and n=4)
    ax.text(2.5, (acc_10[0] + acc_10[1])/2 + 0.05, 'Token Keep Rate 10%', fontsize=12, color='black', 
            fontweight='bold', ha='center', va='bottom')
    
    # 20% line annotation (middle between n=1 and n=4)  
    ax.text(2.5, (acc_20[0] + acc_20[1])/2 + 0.08, 'Token Keep Rate 20%', fontsize=12, color='black', 
            fontweight='bold', ha='center', va='bottom')
    
    # 40% line annotation (middle of full curve, around n=2.5)
    mid_40_y = (acc_40[1] + acc_40[2])/2  # Average of n=2 and n=3 values
    ax.text(2.5, mid_40_y + 0.05, 'Token Keep Rate 40%', fontsize=12, color='black', 
            fontweight='bold', ha='center', va='bottom')
    
    # Format the plot
    ax.set_xlabel('$n$')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Aggregation Window $n$', fontsize=24)
    ax.grid(True, which='major', linestyle=':', linewidth=0.6)
    
    # Set x-axis ticks to show only 1, 2, 3, 4
    ax.set_xticks(n_layers_full)
    ax.set_xlim(0.5, 4.5)
    
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    output_path_pdf = os.path.join(output_dir, "ablation_last_n.pdf")
    output_path_png = os.path.join(output_dir, "ablation_last_n.png")
    
    plt.savefig(output_path_pdf, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved ablation plot: {output_path_pdf}")
    print(f"Saved ablation plot: {output_path_png}")

def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create CLAA last_n_layers ablation plot")
    parser.add_argument("--output_dir", type=str, default="figures",
                       help="Output directory for plots")
    
    args = parser.parse_args()
    
    print("CLAA Last N Layers Ablation Visualization")
    print("=" * 50)
    
    # Generate visualization
    create_ablation_plot(args.output_dir)
    
    print(f"\nVisualization complete! Check {args.output_dir}/ for results")

if __name__ == "__main__":
    main()