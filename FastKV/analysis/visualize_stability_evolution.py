# analysis/visualize_stability_evolution.py
"""
This script generates a visualization showing the evolution of FastKV ranking
quality and the stability of the full ranking (via Spearman's ρ) across model layers.
"""

import argparse
import os
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np

# --- Local Project Imports ---
from .retrieval_metrics import (
    calculate_stability_evolution_metrics,
    load_npz_data_for_dataset,
    deserialize_rankings_in_sample
)
from .viz_utils import set_publication_style, DATASET_NAME_MAP


def plot_stability_evolution(
    evolution_metrics: Dict[str, Any],
    stability_threshold: float,
    stability_window: int,
    k_percentage: float,
    dataset_name: str,
    sample_key: str,
    output_path: Optional[str] = None
):
    """
    Generates a two-panel plot showing ranking quality and Spearman rank correlation stability.
    """
    valid_layers = evolution_metrics.get('valid_layers', [])
    accuracies = evolution_metrics.get('accuracies', [])
    # The stability signal is now Spearman's Rho
    pairwise_stability = evolution_metrics.get('pairwise_stability', [])
    exit_layer = evolution_metrics.get('exit_layer', -1)

    if not valid_layers:
        print("Warning: No valid layers found in metrics. Cannot generate plot.")
        return

    set_publication_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 11), sharex=True)

    display_name = DATASET_NAME_MAP.get(dataset_name, dataset_name)
    fig.suptitle(f'FastKV: Ranking Quality & Stability Evolution (Spearman\'s ρ)\nDataset: {display_name} | Sample: {sample_key}', fontsize=28)

    # Panel 1: Ranking Quality Evolution (vs. Oracle) - UNCHANGED
    ax1.plot(valid_layers, accuracies, marker='.', linestyle='-', color='dodgerblue', label=f'Top-{k_percentage:.0%} Accuracy vs. Oracle')
    ax1.set_ylabel(f'Top-{k_percentage:.0%} Retrieval Accuracy')
    ax1.set_ylim(bottom=-0.05, top=1.05)
    ax1.set_title('Panel 1: Ranking Quality (vs. Final Layer Oracle)', loc='left', style='italic', fontsize=24)

    # Panel 2: NEW Stability Evolution (Pairwise Spearman's Rank Correlation)
    if pairwise_stability:
        plot_x_values = valid_layers[1:]
        ax2.plot(plot_x_values, pairwise_stability, marker='o', markersize=6, linestyle='--', color='crimson', label='Pairwise Spearman\'s ρ')

    ax2.axhline(y=stability_threshold, color='black', linestyle=':', linewidth=3, label=f'Stability Threshold (τ={stability_threshold})')
    ax2.set_ylabel("Spearman's ρ\n(Layer L vs L-1)")
    ax2.set_ylim(bottom=min(0.0, np.min(pairwise_stability) - 0.1) if pairwise_stability else -0.05, top=1.05)
    ax2.set_title(f'Panel 2: Full Ranking Stability (Sliding Window: {stability_window})', loc='left', style='italic', fontsize=24)

    # Add vertical line and annotation for early exit decision - UNCHANGED LOGIC
    if exit_layer != -1:
        try:
            exit_layer_index = valid_layers.index(exit_layer)
            accuracy_at_exit = accuracies[exit_layer_index]

            for ax in [ax1, ax2]:
                ax.axvline(x=exit_layer, color='green', linestyle=':', linewidth=3, label=f'Adaptive Exit at Layer {exit_layer}')

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
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved to: {output_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(
        description="Visualize the evolution of FastKV ranking quality and Spearman stability.",
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
    parser.add_argument("--k_percentage", type=float, default=0.1, help="Top-k percentage for ACCURACY EVALUATION ONLY.")

    parser.add_argument("--stability_threshold", type=float, default=0.99, help="Spearman's ρ threshold for the sliding window average.")
    parser.add_argument("--stability_window", type=int, default=3, help="Number of layers for the sliding window on Spearman's ρ.")
    
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
    
    deserialize_rankings_in_sample(fastkv_data[target_sample_key])

    # --- Step 2: Calculate Evolution Metrics (with new method) ---
    print("Calculating evolution metrics with Spearman's ρ and sliding window...")
    evolution_metrics = calculate_stability_evolution_metrics(
        oracle_ranking=oracle_data[target_sample_key]['ranking'],
        fastkv_rankings=fastkv_data[target_sample_key].get('fastkv_rankings', {}),
        k_percentage=args.k_percentage,
        stability_threshold=args.stability_threshold,
        stability_window=args.stability_window,
    )

    # --- Step 3: Generate Plot ---
    print("Generating plot...")
    output_dir = "figures/stability_evolution_spearman"
    filename = f"spearman_{args.dataset}_{target_sample_key}_thresh{int(args.stability_threshold*100)}_win{args.stability_window}.png"
    output_path = os.path.join(output_dir, filename)

    plot_stability_evolution(
        evolution_metrics=evolution_metrics,
        stability_threshold=args.stability_threshold,
        stability_window=args.stability_window,
        k_percentage=args.k_percentage,
        dataset_name=args.dataset,
        sample_key=target_sample_key,
        output_path=output_path
    )

if __name__ == "__main__":
    main()
