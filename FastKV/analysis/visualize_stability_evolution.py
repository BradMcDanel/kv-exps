# analysis/visualize_stability_evolution.py
"""
This script generates a visualization showing the evolution of FastKV ranking
quality and practical, Top-K stability across model layers for a single data sample.
"""

import argparse
import os
import pickle
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- Local Project Imports ---
# Assuming this script is run from the project root or the path is configured.
from .retrieval_metrics import (
    calculate_stability_evolution_metrics,
    load_npz_data_for_dataset,
    deserialize_rankings_in_sample
)
from .viz_utils import set_publication_style, DATASET_NAME_MAP


def plot_stability_evolution(
    evolution_metrics: Dict[str, Any],
    stability_threshold: float,
    k_percentage: float,
    dataset_name: str,
    sample_key: str,
    output_path: Optional[str] = None
):
    """
    Generates a two-panel plot showing ranking quality and Top-K stability.

    This function is purely for visualization and expects pre-calculated metrics.

    Args:
        evolution_metrics: A dictionary of metrics from `calculate_stability_evolution_metrics`.
        stability_threshold: The Jaccard similarity threshold for the early-exit decision.
        k_percentage: Top-k percentage used for display purposes in titles and labels.
        dataset_name: The name of the dataset for the plot title.
        sample_key: The identifier for the specific sample for the plot title.
        output_path: If provided, the path to save the generated plot image.
    """
    valid_layers = evolution_metrics.get('valid_layers', [])
    accuracies = evolution_metrics.get('accuracies', [])
    pairwise_jaccard = evolution_metrics.get('pairwise_jaccard', [])
    exit_layer = evolution_metrics.get('exit_layer', -1)

    if not valid_layers:
        print("Warning: No valid layers found in metrics. Cannot generate plot.")
        return

    # Use the consistent, high-quality style for the plot
    set_publication_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 11), sharex=True)

    display_name = DATASET_NAME_MAP.get(dataset_name, dataset_name)
    fig.suptitle(f'FastKV: Ranking Quality & Top-K Stability Evolution\nDataset: {display_name} | Sample: {sample_key}', fontsize=30)

    # Panel 1: Ranking Quality Evolution (vs. Oracle)
    ax1.plot(valid_layers, accuracies, marker='.', linestyle='-', color='dodgerblue', label=f'Top-{k_percentage:.0%} Accuracy vs. Oracle')
    ax1.set_ylabel(f'Top-{k_percentage:.0%} Retrieval Accuracy')
    ax1.set_ylim(bottom=-0.05, top=1.05)
    ax1.set_title('Panel 1: Ranking Quality (Ground Truth)', loc='left', style='italic', fontsize=24)

    # Panel 2: Practical Stability Evolution (Pairwise Top-K Jaccard)
    if pairwise_jaccard:
        plot_x_values = valid_layers[1:]
        ax2.plot(plot_x_values, pairwise_jaccard, marker='o', markersize=8, linestyle='--', color='crimson', label='Pairwise Top-K Jaccard')

    ax2.axhline(y=stability_threshold, color='black', linestyle='--', label=f'Stability Threshold (τ={stability_threshold})')
    ax2.set_ylabel(f"Top-{k_percentage:.0%} Jaccard\n(Layer L vs L-1)")
    ax2.set_ylim(bottom=-0.05, top=1.05)
    ax2.set_title('Panel 2: Practical Stability Signal', loc='left', style='italic', fontsize=24)

    # Add vertical line and annotation for early exit decision
    if exit_layer != -1:
        try:
            exit_layer_index = valid_layers.index(exit_layer)
            accuracy_at_exit = accuracies[exit_layer_index]

            for ax in [ax1, ax2]:
                ax.axvline(x=exit_layer, color='green', linestyle=':', linewidth=3, label=f'Early Exit at Layer {exit_layer}')

            # Smartly position the annotation text
            x_pos_offset = 1
            ha = 'left'
            if exit_layer > (max(valid_layers) * 0.75):
                x_pos_offset = -1
                ha = 'right'

            ax1.annotate(f'Exit Acc: {accuracy_at_exit:.3f}',
                         xy=(exit_layer, accuracy_at_exit),
                         xytext=(exit_layer + x_pos_offset, min(0.8, accuracy_at_exit + 0.1)),
                         ha=ha,
                         arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                         bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="black", lw=1, alpha=0.9),
                         fontsize=18)
        except (ValueError, IndexError):
            print(f"Warning: Could not find exit layer {exit_layer} in valid layers to annotate plot.")

    ax1.legend(loc='lower right')
    ax2.legend(loc='lower right')
    ax2.set_xlabel('Model Layer Index')

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    if output_path:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved to: {output_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(
        description="Visualize the evolution of FastKV ranking quality and practical, Top-K stability.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    ORACLE_MODEL = 'meta-llama/Llama-3.1-8B-Instruct'
    FASTKV_MODEL = 'meta-llama/Llama-3.1-8B-Instruct'

    MODELS = {
        'oracle': {'sanitized_name': ORACLE_MODEL.replace('/', '_'), 'base_path': 'analysis_results/oracles'},
        'fastkv': {'sanitized_name': FASTKV_MODEL.replace('/', '_'), 'base_path': 'analysis_results/approx_rankings'},
    }

    parser.add_argument("--dataset", type=str, default='2wikimqa', help="Dataset to analyze.")
    parser.add_argument("--sample_idx", type=int, default=0, help="The 0-based index of the sample to analyze.")
    parser.add_argument("--k_percentage", type=float, default=0.1, help="Top-k percentage for retrieval accuracy and stability.")
    parser.add_argument("--stability_threshold", type=float, default=0.90, help="Top-K Jaccard similarity threshold for early exit.")
    
    args = parser.parse_args()

    # --- Step 1: Load Data ---
    print("Loading data...")
    oracle_data = load_npz_data_for_dataset(MODELS['oracle']['base_path'], MODELS['oracle']['sanitized_name'], args.dataset)
    fastkv_data = load_npz_data_for_dataset(MODELS['fastkv']['base_path'], MODELS['fastkv']['sanitized_name'], args.dataset)

    if not oracle_data or not fastkv_data:
        print("Could not load necessary data. Exiting.")
        return

    common_keys = sorted(list(set(oracle_data.keys()) & set(fastkv_data.keys())))
    if not common_keys or args.sample_idx >= len(common_keys):
        print(f"Error: Could not find sample at index {args.sample_idx}. Only {len(common_keys)} common samples found.")
        return

    target_sample_key = common_keys[args.sample_idx]
    
    # Ensure rankings are deserialized if they were stored as pickled bytes
    deserialize_rankings_in_sample(fastkv_data[target_sample_key])

    # --- Step 2: Calculate Evolution Metrics ---
    print("Calculating evolution metrics...")
    evolution_metrics = calculate_stability_evolution_metrics(
        oracle_ranking=oracle_data[target_sample_key]['ranking'],
        fastkv_rankings=fastkv_data[target_sample_key].get('fastkv_rankings', {}),
        k_percentage=args.k_percentage,
        stability_threshold=args.stability_threshold,
    )

    # --- Step 3: Generate Plot ---
    print("Generating plot...")
    output_dir = "figures/stability_evolution"
    filename = f"stability_{args.dataset}_{target_sample_key}_k{int(args.k_percentage*100)}.png"
    output_path = os.path.join(output_dir, filename)

    plot_stability_evolution(
        evolution_metrics=evolution_metrics,
        stability_threshold=args.stability_threshold,
        k_percentage=args.k_percentage,
        dataset_name=args.dataset,
        sample_key=target_sample_key,
        output_path=output_path
    )

if __name__ == "__main__":
    main()
