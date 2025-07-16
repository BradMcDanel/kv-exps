# analysis/visualize_stability_evolution.py

import argparse
import os
import pickle
import math
from typing import Any, Dict, Set

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

# --- Data Loading (Re-used) ---
def load_npz_data_for_dataset(base_path: str, model_name_sanitized: str, dataset_name: str) -> Dict[str, Any]:
    file_path = os.path.join(base_path, model_name_sanitized, f"{dataset_name}.npz")
    if not os.path.exists(file_path): return {}
    try:
        npz_file = np.load(file_path, allow_pickle=True)
        data = {}
        for key, value in npz_file.items():
            item = value.item()
            if isinstance(item, dict) and 'fastkv_rankings' in item and isinstance(item['fastkv_rankings'], bytes):
                item['fastkv_rankings'] = pickle.loads(item['fastkv_rankings'])
            data[key] = item
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return {}

# --- Metric Calculation Helpers ---
def get_top_k_indices(ranking: np.ndarray, k: int) -> Set[int]:
    """Efficiently returns the set of indices for the top-k scores."""
    if ranking.size < k: return set()
    # Use argpartition for efficiency; it finds the k-th largest element and partitions around it.
    # We want the indices of the k largest values.
    indices = np.argpartition(ranking, -k)[-k:]
    return set(indices)

def calculate_jaccard_similarity(set1: Set[int], set2: Set[int]) -> float:
    """Calculates Jaccard similarity (intersection over union) between two sets."""
    intersection_size = len(set1.intersection(set2))
    union_size = len(set1.union(set2))
    return intersection_size / union_size if union_size > 0 else 0.0

# --- Main Plotting Logic ---
def plot_stability_evolution(
    oracle_ranking: np.ndarray,
    fastkv_rankings: Dict[int, np.ndarray],
    k_percentage: float,
    stability_threshold: float,
    dataset_name: str,
    sample_key: str
):
    """Generates a two-panel plot showing ranking quality and practical, Top-K stability."""
    
    # --- Data Preparation ---
    prompt_len = len(oracle_ranking)
    k_absolute = max(1, math.ceil(prompt_len * k_percentage))

    oracle_top_k_set = get_top_k_indices(oracle_ranking, k_absolute)
    
    valid_layers = sorted(fastkv_rankings.keys())
    if not valid_layers:
        print("No FastKV layers found to plot.")
        return
        
    # --- Metric Calculation ---
    # 1. Accuracy vs. Oracle (calculated for all available layers for a smooth curve)
    accuracies = [calculate_jaccard_similarity(get_top_k_indices(fastkv_rankings[layer], k_absolute), oracle_top_k_set) for layer in valid_layers]

    # 2. Practical, Pairwise Top-K Jaccard Similarity
    pairwise_jaccard = []
    if len(valid_layers) >= 2:
        for i in range(1, len(valid_layers)):
            prev_layer_idx = valid_layers[i-1]
            curr_layer_idx = valid_layers[i]
            
            prev_top_k = get_top_k_indices(fastkv_rankings[prev_layer_idx], k_absolute)
            curr_top_k = get_top_k_indices(fastkv_rankings[curr_layer_idx], k_absolute)
            
            similarity = calculate_jaccard_similarity(prev_top_k, curr_top_k)
            pairwise_jaccard.append(similarity)

    # --- Determine Early Exit Point based on Practical Signal ---
    exit_layer = -1
    if pairwise_jaccard:
        for i, similarity in enumerate(pairwise_jaccard):
            if similarity >= stability_threshold:
                # The exit layer is the *current* layer in the pair
                exit_layer = valid_layers[i+1] # i is index for jaccard list, which starts from layer pair (0,1)
                break
            
    # --- Plotting ---
    sns.set_theme(style="whitegrid", context="talk")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), sharex=True)
    fig.suptitle(f'Evolution of Ranking Quality & Top-K Stability (FastKV)\nDataset: {dataset_name}, Sample: {sample_key}', fontsize=24)
    
    # Panel 1: Ranking Quality Evolution (vs. Oracle)
    ax1.plot(valid_layers, accuracies, marker='.', linestyle='-', color='dodgerblue', label=f'Top-{k_percentage:.0%} Accuracy vs. Oracle')
    ax1.set_ylabel(f'Top-{k_percentage:.0%} Retrieval Accuracy')
    ax1.set_ylim(bottom=-0.05, top=1.05)
    ax1.set_title('Panel 1: Ranking Quality (Ground Truth)', loc='left', style='italic')
    
    # Panel 2: Practical Stability Evolution (Pairwise Top-K Jaccard)
    if pairwise_jaccard:
        # The x-values for this plot are the layers where the comparison was made (from layer 1 onwards)
        plot_x_values = valid_layers[1:]
        ax2.plot(plot_x_values, pairwise_jaccard, marker='o', markersize=8, linestyle='--', color='crimson', label='Pairwise Top-K Jaccard')
    ax2.axhline(y=stability_threshold, color='black', linestyle='--', label=f'Stability Threshold (τ={stability_threshold})')
    ax2.set_ylabel(f"Top-{k_percentage:.0%} Jaccard (Layer_i vs Layer_{{i-1}})")
    ax2.set_ylim(bottom=-0.05, top=1.05)
    ax2.set_title('Panel 2: Practical Stability Signal', loc='left', style='italic')
    
    # Add vertical line for early exit decision
    if exit_layer != -1:
        accuracy_at_exit = accuracies[valid_layers.index(exit_layer)]
        for ax in [ax1, ax2]:
            ax.axvline(x=exit_layer, color='green', linestyle=':', linewidth=2.5, label=f'Early Exit at Layer {exit_layer}')
        
        ax1.annotate(f'Acc at Exit: {accuracy_at_exit:.3f}', 
                     xy=(exit_layer, accuracy_at_exit), 
                     xytext=(exit_layer - 5 if exit_layer > 16 else exit_layer + 1, min(0.85, accuracy_at_exit + 0.15)),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
                     bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="black", lw=1, alpha=0.8),
                     fontsize=14)

    ax1.legend(loc='lower right')
    ax2.legend(loc='lower right')
    ax2.set_xlabel('Model Layer Index')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save the figure
    output_dir = "figures"
    os.makedirs(output_dir, exist_ok=True)
    filename = f"stability_evolution_topk_{dataset_name}_{sample_key}.png"
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300)
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
    parser.add_argument("--stability_threshold", type=float, default=0.90, help="Top-K Jaccard similarity threshold.")
    
    args = parser.parse_args()

    # --- Load Data ---
    oracle_data = load_npz_data_for_dataset(MODELS['oracle']['base_path'], MODELS['oracle']['sanitized_name'], args.dataset)
    fastkv_data = load_npz_data_for_dataset(MODELS['fastkv']['base_path'], MODELS['fastkv']['sanitized_name'], args.dataset)
    
    if not oracle_data or not fastkv_data:
        print("Could not load necessary data. Exiting.")
        return

    common_keys = sorted(list(set(oracle_data.keys()) & set(fastkv_data.keys())))
    if not common_keys or args.sample_idx >= len(common_keys):
        print(f"Error: Could not find sample at index {args.sample_idx}. You may need to generate more samples for this dataset.")
        return
        
    target_sample_key = common_keys[args.sample_idx]
    
    # --- Generate Plot ---
    plot_stability_evolution(
        oracle_ranking=oracle_data[target_sample_key]['ranking'],
        fastkv_rankings=fastkv_data[target_sample_key]['fastkv_rankings'],
        k_percentage=args.k_percentage,
        stability_threshold=args.stability_threshold,
        dataset_name=args.dataset,
        sample_key=target_sample_key
    )

if __name__ == "__main__":
    main()
