#!/usr/bin/env python3

import matplotlib.pyplot as plt
from .viz_utils import set_publication_style, METHOD_COLORS

def create_vmas_alpha_plot():
    set_publication_style()
    
    x_values = [0.01, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.25, 2.5, 2.75, 3.0]
    y_values = [46.87, 46.79, 46.91, 46.97, 47.03, 47.05, 47.07, 46.92, 46.87, 46.66, 45.89, 44.44]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot all points as circles first
    ax.plot(x_values, y_values, 'o-', color=METHOD_COLORS['CLAA'], linewidth=3.5, markersize=9)
    
    # Add FastKV point at x=0 with same y value as first data point
    fastkv_x = 0
    fastkv_y = y_values[0]  # Using first y value as reference
    ax.plot(fastkv_x, fastkv_y, '*', color=METHOD_COLORS['FastKV'], markersize=15)
    ax.annotate('FastKV', (fastkv_x, fastkv_y), xytext=(10, 3), 
                textcoords='offset points', fontsize=22, color=METHOD_COLORS['FastKV'])
    
    ax.set_xlabel('VMAS Alpha')
    ax.set_ylabel('LongBench Accuracy')
    ax.set_title('LongBench Accuracy vs VMAS Alpha')
    
    ax.grid(True, linestyle=':', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('figures/ablation_vmas_alpha.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/ablation_vmas_alpha.png', dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    create_vmas_alpha_plot()