# analysis/visualize_adaptive_exit_performance.py
"""
This script generates a visualization comparing the performance of an adaptive,
stability-based (Spearman's ρ) layer selection strategy for FastKV against a
fixed-layer heuristic.
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
    compute_adaptive_exit_metrics_for_dataset, # This function is now the Spearman version
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

def generate_dummy_data(threshold: float, window: int) -> dict:
    """Generates random data for layout and debugging purposes."""
    print("--- Using --debug mode: Generating random dummy data ---")
    results = {}
    for ds_name in ALL_DATASETS_TO_PLOT:
        base_acc = np.random.uniform(0.75, 0.95)
        # Spearman should yield higher accuracy
        adaptive_acc = base_acc + np.random.uniform(-0.03, 0.03)
        fixed_acc = base_acc + np.random.uniform(-0.05, 0.01)
        # A higher threshold should result in a later exit layer
        avg_layer = 10 + (threshold - 0.95) * 50 + (window - 2) * 2 + np.random.uniform(-2, 2)
        results[ds_name] = {
            'avg_adaptive_layer': max(8, min(28, avg_layer)),
            'avg_adaptive_accuracy': min(1.0, adaptive_acc),
            'avg_fixed_accuracy': min(1.0, fixed_acc),
        }
    return results

def plot_performance_grid(task_results: dict, k_percentage: float, threshold: float, window: int, fixed_layer: int, output_prefix: str):
    """Creates the 2x3 grid of grouped bar charts."""
    set_publication_style()
    n_rows, n_cols = 2, 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, 12), sharey=True)
    fig.subplots_adjust(hspace=0.6, wspace=0.1)

    max_acc = 0
    for task, datasets_data in task_results.items():
        if not datasets_data: continue
        for _, metrics in datasets_data:
            max_acc = max(max_acc, metrics.get('avg_adaptive_accuracy', 0), metrics.get('avg_fixed_accuracy', 0))
    
    y_max_limit = min(1.0, (max_acc // 0.1 + 1) * 0.1) if max_acc > 0 else 1.0

    task_items = list(TASKS_AND_DATASETS.keys())
    for i, ax in enumerate(axes.flat):
        if i >= len(task_items):
            ax.axis('off'); continue
        
        task_name = task_items[i]
        ax.set_title(task_name, fontsize=28, pad=20)
        
        if task_name not in task_results or not task_results[task_name]:
            ax.text(0.5, 0.5, "No data available", transform=ax.transAxes, ha='center', va='center', style='italic')
            continue
            
        dataset_info = task_results[task_name]
        labels = [DATASET_NAME_MAP.get(ds_name, ds_name) for ds_name, _ in dataset_info]
        adaptive_accs = [metrics['avg_adaptive_accuracy'] for _, metrics in dataset_info]
        fixed_accs = [metrics['avg_fixed_accuracy'] for _, metrics in dataset_info]
        avg_layers = [f"L={metrics['avg_adaptive_layer']:.1f}" for _, metrics in dataset_info]

        x = np.arange(len(labels))
        width = 0.35

        rects1 = ax.bar(x - width/2, adaptive_accs, width, label=f'Adaptive Exit (ρ={threshold}, W={window})', color=METHOD_COLORS['Adaptive Exit'])
        rects2 = ax.bar(x + width/2, fixed_accs, width, label=f'Fixed Exit (Layer {fixed_layer})', color=METHOD_COLORS['Fixed Exit'])
        
        ax.bar_label(rects1, labels=avg_layers, padding=5, fontsize=18, color='black', weight='bold')
        
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=22)
        ax.set_ylim(0, y_max_limit)
        ax.grid(True, which='major', axis='y', linestyle=':', linewidth=0.8)
        ax.spines[['top', 'right']].set_visible(False)

    fig.text(0.5, 0.02, 'Dataset', ha='center', va='center', fontsize=34)
    fig.text(0.06, 0.5, f'Top-{k_percentage:.0%} Token Overlap with Oracle', ha='center', va='center', rotation='vertical', fontsize=34)
    
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=2, frameon=False, fontsize=24)
    
    fig.suptitle(f'FastKV Performance: Adaptive (Spearman\'s ρ) vs. Fixed Layer Selection', y=1.03, fontsize=36, weight='bold')

    plt.savefig(f"{output_prefix}.pdf", format='pdf', dpi=300, bbox_inches='tight')
    print(f"\nPerformance grid plot saved to: {output_prefix}.pdf")
    plt.savefig(f"{output_prefix}.png", format='png', dpi=300, bbox_inches='tight')
    print(f"Performance grid plot saved to: {output_prefix}.png")
    plt.close(fig)


def main():
    """Main execution block."""
    parser = argparse.ArgumentParser(description="Visualize adaptive (Spearman's ρ) vs. fixed layer selection for FastKV.")
    parser.add_argument("--k_percentage", type=float, default=0.1, help="Percentage for top-k ACCURACY EVALUATION.")
    parser.add_argument("--stability_threshold", type=float, default=0.99, help="Spearman's ρ threshold for the sliding window average.")
    parser.add_argument("--stability_window", type=int, default=3, help="Number of layers for the sliding window on Spearman's ρ.")
    parser.add_argument("--fixed_layer", type=int, default=FIXED_EXIT_LAYER, help="The fixed layer to compare against.")
    parser.add_argument("--debug", action="store_true", help="Use dummy data for fast layout iteration.")
    args = parser.parse_args()
    
    output_dir = "figures"
    os.makedirs(output_dir, exist_ok=True)
    output_prefix = os.path.join(output_dir, f"adaptive_exit_perf_spearman_thresh{int(args.stability_threshold*100)}_win{args.stability_window}")

    if args.debug:
        all_results = generate_dummy_data(args.stability_threshold, args.stability_window)
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

        print("\nComputing metrics for each dataset using Spearman's ρ...")
        all_results = {}
        for ds in tqdm(ALL_DATASETS_TO_PLOT, desc="Analyzing Datasets"):
            if ds in oracle_results and ds in fastkv_results and oracle_results[ds] and fastkv_results[ds]:
                metrics = compute_adaptive_exit_metrics_for_dataset(
                    oracle_data_for_ds=oracle_results[ds],
                    fastkv_data_for_ds=fastkv_results[ds],
                    k_percentage=args.k_percentage,
                    stability_threshold=args.stability_threshold,
                    stability_window=args.stability_window,
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

    plot_performance_grid(
        task_level_results,
        args.k_percentage,
        args.stability_threshold,
        args.stability_window,
        args.fixed_layer,
        output_prefix
    )

if __name__ == "__main__":
    main()
