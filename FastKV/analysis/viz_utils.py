# analysis/viz_utils.py

import matplotlib.pyplot as plt
import seaborn as sns

def set_publication_style():
    """Sets a consistent, high-quality plotting style for figures."""
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 18,          # Base font size
        'axes.labelsize': 26,     # X and Y labels
        'axes.titlesize': 22,     # Subplot titles
        'xtick.labelsize': 16,    # X-axis tick labels
        'ytick.labelsize': 16,    # Y-axis tick labels
        'legend.fontsize': 20,    # Legend text
        'legend.title_fontsize': 22, # Legend title
        'figure.titlesize': 30,   # Main figure title
        'grid.linestyle': ':',
        'grid.linewidth': 0.7,
        'lines.linewidth': 3.0,   # Make lines thicker
        'lines.markersize': 8,    # Marker size
    })
