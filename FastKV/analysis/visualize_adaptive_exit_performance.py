# analysis/visualize_adaptive_exit_performance.py
"""
This script generates a visualization comparing the performance of an adaptive,
stability-based layer selection strategy for FastKV against a fixed-layer heuristic.
"""

import os
import argparse
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- Local Project Imports ---
from .viz_utils import (
    set_publication_style,
    METHOD_COLORS,
    TASKS_AND_DATASETS,
    ALL_DATASETS_TO_PLOT,
    DATASET_NAME_MAP,
    FIXED_EXIT_LAYER,
)
from .retrieval_metrics import (
    load_npz_data_for_dataset,
    compute_adaptive_exit_metrics_for_dataset,
)

# --- MODELS CONFIGURATION ---
MODELS_TO_LOAD = {
    'oracle': {
        'sanitized_name': 'meta-llama_Llama-3.1-8B-Instruct',
        'base_path': 'analysis_results/oracles'
    },
    'fastkv': {
        'sanitized_name': 'meta-llama_Llama-3.1-8B-Instruct',
        'base_path': 'analysis_results/approx_rankings'
    }
}

def generate_dummy_data(tau: float) -> dict:
    """Generates random data for layout and debugging purposes."""
    print("--- Using --debug mode: Generating random dummy data ---")
    results = {}
    for ds_name in ALL_DATASETS_TO_PLOT:
        base_acc = np.random.uniform(0.6, 0.9)
        adaptive_acc = base_acc + np.random.uniform(-0.05, 0.05)
        fixed_acc = base_acc + np.random.uniform(-0.03, 0.02)
        avg_layer = 15 - (tau * 10) + np.random.uniform(-2, 2)
        results[ds_name] = {
            'avg_adaptive_layer': max(5, avg_layer),
            'avg_adaptive_accuracy': min(1.0, adaptive_acc),
            'avg_fixed_accuracy': min(1.0, fixed_acc),
        }
    return results

def plot_performance_grid(task_results: dict, k_percentage: float, tau: float, fixed_layer: int, output_prefix: str):
    """Creates the 2x3 grid of grouped bar charts."""
    set_publication_style()
    n_rows, n_cols = 2, 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, 12), sharey=True)
    fig.subplots_adjust(hspace=0.6, wspace=0.1)

    max_acc = 0
    for task, datasets_data in task_results.items():
        for _, metrics in datasets_data:
            max_acc = max(max_acc, metrics['avg_adaptive_accuracy'], metrics['avg_fixed_accuracy'])
    
    y_max_limit = (max_acc // 0.1 + 1) * 0.1 # Round up to nearest 0.1

    task_items = list(TASKS_AND_DATASETS.keys())
    for i, ax in enumerate(axes.flat):
        if i >= len(task_items):
            ax.axis('off')
            continue
        
        task_name = task_items[i]
        ax.set_title(task_name, fontsize=28, pad=20)
        
        if task_name not in task_results or not task_results[task_name]:
            ax.text(0.5, 0.5, "No data available", transform=ax.transAxes, ha='center', va='center')
            continue
            
        dataset_info = task_results[task_name]
        labels = [DATASET_NAME_MAP.get(ds_name, ds_name) for ds_name, _ in dataset_info]
        adaptive_accs = [metrics['avg_adaptive_accuracy'] for _, metrics in dataset_info]
        fixed_accs = [metrics['avg_fixed_accuracy'] for _, metrics in dataset_info]
        avg_layers = [f"L={metrics['avg_adaptive_layer']:.1f}" for _, metrics in dataset_info]

        x = np.arange(len(labels))
        width = 0.35

        rects1 = ax.bar(x - width/2, adaptive_accs, width, label=f'Adaptive Exit (τ={tau})', color=METHOD_COLORS['Adaptive Exit'])
        rects2 = ax.bar(x + width/2, fixed_accs, width, label=f'Fixed Exit (Layer {fixed_layer})', color=METHOD_COLORS['Fixed Exit'])
        
        ax.bar_label(rects1, labels=avg_layers, padding=5, fontsize=18, color='black')
        
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=22)
        ax.set_ylim(0, y_max_limit)
        ax.grid(True, which='major', axis='y', linestyle=':', linewidth=0.8)
        ax.spines[['top', 'right']].set_visible(False)

    # Common labels and legend
    fig.text(0.5, 0.02, 'Dataset', ha='center', va='center', fontsize=34)
    fig.text(0.06, 0.5, f'Top-{k_percentage:.0%} Token Overlap with Oracle', ha='center', va='center', rotation='vertical', fontsize=34)
    
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=2, frameon=False, fontsize=24)
    
    fig.suptitle(f'FastKV: Adaptive vs. Fixed Layer Selection Performance', y=1.03, fontsize=36, weight='bold')

    plt.savefig(f"{output_prefix}.pdf", format='pdf', dpi=300, bbox_inches='tight')
    print(f"\nPerformance grid plot saved to: {output_prefix}.pdf")
    plt.savefig(f"{output_prefix}.png", format='png', dpi=300, bbox_inches='tight')
    print(f"Performance grid plot saved to: {output_prefix}.png")
    plt.close(fig)


def main():
    """Main execution block."""
    parser = argparse.ArgumentParser(description="Visualize adaptive vs. fixed layer selection for FastKV.")
    parser.add_argument("--k_percentage", type=float, default=0.1, help="Percentage for top-k analysis.")
    parser.add_argument("--tau", type=float, default=0.9, help="Jaccard similarity threshold for adaptive exit.")
    parser.add_argument("--fixed_layer", type=int, default=FIXED_EXIT_LAYER, help="The fixed layer to compare against.")
    parser.add_argument("--debug", action="store_true", help="Use dummy data for fast layout iteration.")
    args = parser.parse_args()
    
    output_dir = "figures"
    os.makedirs(output_dir, exist_ok=True)
    output_prefix = os.path.join(output_dir, f"adaptive_exit_perf")

    if args.debug:
        all_results = generate_dummy_data(args.tau)
    else:
        print("Loading oracle and FastKV data...")
        oracle_results = {
            ds: load_npz_data_for_dataset(MODELS_TO_LOAD['oracle']['base_path'], MODELS_TO_LOAD['oracle']['sanitized_name'], ds)
            for ds in tqdm(ALL_DATASETS_TO_PLOT, desc="Loading Oracle Data")
        }
        fastkv_results = {
            ds: load_npz_data_for_dataset(MODELS_TO_LOAD['fastkv']['base_path'], MODELS_TO_LOAD['fastkv']['sanitized_name'], ds)
            for ds in tqdm(ALL_DATASETS_TO_PLOT, desc="Loading FastKV Data")
        }

        print("\nComputing metrics for each dataset...")
        all_results = {}
        for ds in tqdm(ALL_DATASETS_TO_PLOT, desc="Analyzing Datasets"):
            if ds in oracle_results and ds in fastkv_results and oracle_results[ds] and fastkv_results[ds]:
                metrics = compute_adaptive_exit_metrics_for_dataset(
                    oracle_data_for_ds=oracle_results[ds],
                    fastkv_data_for_ds=fastkv_results[ds],
                    k_percentage=args.k_percentage,
                    stability_threshold=args.tau,
                    fixed_layer=args.fixed_layer,
                )
                if metrics:
                    all_results[ds] = metrics
    
    if not all_results:
        print("No valid data found to plot. Exiting.")
        return

    # Group results by task for plotting
    task_level_results = defaultdict(list)
    for task, datasets in TASKS_AND_DATASETS.items():
        for ds_name in datasets:
            if ds_name in all_results:
                task_level_results[task].append((ds_name, all_results[ds_name]))

    plot_performance_grid(task_level_results, args.k_percentage, args.tau, args.fixed_layer, output_prefix)

if __name__ == "__main__":
    main()
