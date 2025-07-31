#!/usr/bin/env python3

import matplotlib.pyplot as plt
from .viz_utils import set_publication_style, METHOD_COLORS

def create_longbench_accuracy_plot():
    set_publication_style()
    
    x_values = [1, 2, 4, 8, 12]
    y_values = [46.81, 46.36, 45.97, 45.39, 45.02]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot all points as circles first
    ax.plot(x_values, y_values, 'o-', color=METHOD_COLORS['CLAA'], linewidth=3.5, markersize=9)
    
    # Overlay the first point as a star and annotate FastKV
    ax.plot(x_values[0], y_values[0], '*', color=METHOD_COLORS['FastKV'], markersize=15)
    ax.annotate('FastKV', (x_values[0], y_values[0]), xytext=(10, 3), 
                textcoords='offset points', fontsize=22, color=METHOD_COLORS['FastKV'])
    
    ax.set_xlabel('CLAA n layers')
    ax.set_ylabel('LongBench Accuracy')
    ax.set_title('LongBench Accuracy at 10% keep rate')
    
    ax.set_xticks(x_values)
    ax.grid(True, linestyle=':', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('figures/ablation_claa_n.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/ablation_claa_n.png', dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    create_longbench_accuracy_plot()
