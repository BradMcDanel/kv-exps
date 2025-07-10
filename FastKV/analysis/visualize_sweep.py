# analysis/visualize_sweep.py

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def set_publication_style():
    """Sets matplotlib/seaborn parameters for a professional, publication-quality look."""
    sns.set_theme(context='paper', style='whitegrid', palette='colorblind', font_scale=1.5)
    plt.rcParams.update({
        'text.color': 'black', 'axes.labelcolor': 'black',
        'xtick.color': 'black', 'ytick.color': 'black',
        'grid.color': '#cccccc', 'grid.linestyle': ':',
        'axes.edgecolor': 'black', 'figure.facecolor': 'white',
        'axes.facecolor': 'white', 'savefig.facecolor': 'white',
        'legend.frameon': True, 'legend.framealpha': 0.9,
        'legend.facecolor': 'white',
    })

def plot_sweep_metric(df, output_dir, x_col, y_col, title, y_label, file_name):
    """Creates and saves a single line plot for a given metric, with a special baseline."""
    plt.figure(figsize=(14, 8))
    
    baseline_data = df[df['Method'] == 'Full KV Cache']
    dynamic_data = df[df['Method'] != 'Full KV Cache']

    ax = sns.lineplot(
        data=dynamic_data, x=x_col, y=y_col, hue='Method', style='Method',
        linewidth=3.0, marker='o', markersize=9, dashes=True
    )
    
    if not baseline_data.empty:
        baseline_value = baseline_data[y_col].iloc[0]
        ax.axhline(
            y=baseline_value, color='black', linestyle='--', linewidth=2.5,
            label='Full KV Cache (Baseline)'
        )
    
    plt.title(title, fontsize=22, weight='bold', pad=20)
    plt.xlabel("Effective Prefill Cache Size (%)", fontsize=18, labelpad=10)
    plt.ylabel(y_label, fontsize=18, labelpad=10)
    
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    handles, labels = ax.get_legend_handles_labels()
    try:
        baseline_index = labels.index('Full KV Cache (Baseline)')
        handles = [handles[baseline_index]] + [h for i, h in enumerate(handles) if i != baseline_index]
        labels = [labels[baseline_index]] + [l for i, l in enumerate(labels) if i != baseline_index]
    except (ValueError, IndexError):
        pass

    ax.legend(handles=handles, labels=labels, title='Method', fontsize=14, title_fontsize=15)
    
    ax.set_axisbelow(True)
    ax.set_facecolor('#f7f7f7')
    
    plt.tight_layout()
    
    output_path = output_dir / f"{file_name}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_path}")
    plt.close()

def main():
    """Main function to load data and generate all plots."""
    parser = argparse.ArgumentParser(description="Visualize performance sweep results from a CSV file.")
    parser.add_argument("input_csv", type=str, help="Path to the input CSV file.")
    parser.add_argument("--output_dir", type=str, default="analysis_results", help="Directory to save the plots.")
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    if not input_path.exists():
        print(f"Error: Input file not found at {input_path}")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    set_publication_style()
    
    df = pd.read_csv(input_path)
    
    df['pruning_level_pct'] = df['pruning_level'] * 100
    
    method_map = {
        'fullkv': 'Full KV Cache',
        'fastkv': 'FastKV',
        'speculative_prefill': 'Speculative Prefill',
        'hgp': 'HGP'
    }
    df['Method'] = df['method'].map(method_map)
    
    file_prefix = input_path.stem

    try:
        seq_len = int(df["seq_len"].iloc[0])
        seq_len_str = f"(Sequence Length: {seq_len})"
    except (KeyError, IndexError):
        seq_len_str = ""

    plot_sweep_metric(
        df=df, output_dir=output_dir, x_col='pruning_level_pct', y_col='ttft_ms',
        title=f'Time to First Token (TTFT) vs. Cache Size {seq_len_str}',
        y_label='TTFT (milliseconds)', file_name=f"{file_prefix}_ttft"
    )

    try:
        steps = int(df["num_decode_steps"].iloc[0])
        e2e_title = f'End-to-End Time ({steps} Tokens) vs. Cache Size {seq_len_str}'
    except (KeyError, IndexError):
        e2e_title = f'End-to-End Time vs. Cache Size {seq_len_str}'

    plot_sweep_metric(
        df=df, output_dir=output_dir, x_col='pruning_level_pct', y_col='e2e_time_ms',
        title=e2e_title, y_label='E2E Time (milliseconds)',
        file_name=f"{file_prefix}_e2e_time"
    )

    plot_sweep_metric(
        df=df, output_dir=output_dir, x_col='pruning_level_pct', y_col='max_memory_gb',
        title=f'Peak Memory Usage vs. Cache Size {seq_len_str}',
        y_label='Peak GPU Memory (GB)', file_name=f"{file_prefix}_memory"
    )

if __name__ == "__main__":
    main()
