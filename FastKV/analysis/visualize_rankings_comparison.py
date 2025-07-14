# analysis/visualize_rankings_comparison.py

import os
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch
import math
from typing import Dict, List

# --- Plotting Style ---
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
})

def calculate_retrieval_accuracy(
    approx_rankings: Dict[int, List[float]],
    oracle_ranking: torch.Tensor,
    k_percentage: float
) -> Dict[int, float]:
    """Calculates the retrieval accuracy for each layer's ranking against the oracle."""
    if not approx_rankings or k_percentage <= 0 or oracle_ranking.numel() == 0:
        return {}

    k = max(1, math.ceil(len(oracle_ranking) * k_percentage))
    _, top_k_oracle_indices = torch.topk(oracle_ranking, k=k)
    oracle_set = set(top_k_oracle_indices.tolist())
    
    accuracies = {}
    for layer_idx, layer_scores in approx_rankings.items():
        scores_tensor = torch.tensor(layer_scores, dtype=torch.float32)
        if scores_tensor.numel() == 0:
            accuracies[layer_idx] = 0.0
            continue
        
        # Ensure we have enough scores to select k tokens
        num_scores = len(scores_tensor)
        current_k = min(k, num_scores)
        if current_k == 0:
            accuracies[layer_idx] = 0.0
            continue

        _, top_k_layer_indices = torch.topk(scores_tensor, k=current_k)
        layer_set = set(top_k_layer_indices.tolist())
        
        intersection_size = len(oracle_set.intersection(layer_set))
        accuracy = intersection_size / k if k > 0 else 0.0
        accuracies[layer_idx] = accuracy
        
    return accuracies

def load_data(path: str) -> Dict:
    """Safely loads a pickle file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found at: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)

def plot_comparison_grid(all_results: Dict, args: argparse.Namespace):
    """Creates a comparison grid for ranking methods across models."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 10), sharey=True)
    axes_flat = axes.flatten()
    
    styles = {
        '8B_cumulative': {'color': '#d62728', 'linestyle': '-', 'marker': 'o', 'label': '8B Cumulative (SP-style)'},
        '8B_layerwise':  {'color': '#ff7f0e', 'linestyle': '--', 'marker': 's', 'label': '8B Layerwise (FastKV-style)'},
        '1B_cumulative': {'color': '#1f77b4', 'linestyle': '-', 'marker': 'o', 'label': '1B Cumulative (SP-style)'},
        '1B_layerwise':  {'color': '#17becf', 'linestyle': '--', 'marker': 's', 'label': '1B Layerwise (FastKV-style)'},
    }
    
    approx_data_keys = ['8B', '1B']
    oracle_data = all_results['oracle']

    for i, dataset_name in enumerate(tqdm(args.datasets_to_plot, desc="Plotting Datasets")):
        ax = axes_flat[i]
        
        if dataset_name not in oracle_data:
            ax.set_title(f"{dataset_name.replace('_', ' ').title()} (No Oracle Data)")
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=14, color='gray')
            continue
        
        num_common_samples_map = {}
        for key in approx_data_keys:
            results_dict = all_results.get(key, {})
            if dataset_name not in results_dict or not results_dict.get(dataset_name): continue

            oracle_samples = {s['sample_idx']: s for s in oracle_data.get(dataset_name, [])}
            approx_samples = {s['sample_idx']: s for s in results_dict.get(dataset_name, [])}
            
            layer_accuracies_cumulative = {}
            layer_accuracies_layerwise = {}
            
            valid_sample_count = 0
            for sample_idx in sorted(oracle_samples.keys() & approx_samples.keys()):
                o_data = oracle_samples[sample_idx]
                a_data = approx_samples[sample_idx]
                
                # Use the relaxed assertion checking only length
                if len(o_data['input_ids']) != len(a_data['input_ids']):
                    continue
                
                valid_sample_count += 1
                oracle_ranking = torch.tensor(o_data['ranking'])

                cum_acc = calculate_retrieval_accuracy(a_data.get('cumulative_rankings', {}), oracle_ranking, args.k_percentage)
                for layer, acc in cum_acc.items():
                    if layer not in layer_accuracies_cumulative: layer_accuracies_cumulative[layer] = []
                    layer_accuracies_cumulative[layer].append(acc)

                lay_acc = calculate_retrieval_accuracy(a_data.get('layerwise_rankings', {}), oracle_ranking, args.k_percentage)
                for layer, acc in lay_acc.items():
                    if layer not in layer_accuracies_layerwise: layer_accuracies_layerwise[layer] = []
                    layer_accuracies_layerwise[layer].append(acc)

            num_common_samples_map[key] = valid_sample_count
            
            # Plotting logic
            if layer_accuracies_cumulative:
                sorted_layers = sorted(layer_accuracies_cumulative.keys())
                mean_acc = [np.mean(layer_accuracies_cumulative[l]) for l in sorted_layers]
                ax.plot(sorted_layers, mean_acc, **styles[f"{key}_cumulative"], markersize=5, zorder=5)
            
            if layer_accuracies_layerwise:
                sorted_layers = sorted(layer_accuracies_layerwise.keys())
                mean_acc = [np.mean(layer_accuracies_layerwise[l]) for l in sorted_layers]
                ax.plot(sorted_layers, mean_acc, **styles[f"{key}_layerwise"], markersize=5, zorder=3)

        sample_count_str = ", ".join([f"{k}:{v}" for k, v in num_common_samples_map.items()])
        ax.set_title(f"{dataset_name.replace('_', ' ').title()} (n={sample_count_str})")
        ax.grid(True, which='both', linestyle=':', linewidth=0.7)

    # --- Final Figure Formatting ---
    legend_ax = axes_flat[-1]
    if not any(legend_ax.get_lines()):
        legend_ax.set_axis_off()

    handles = [plt.Line2D([0], [0], **s) for s in styles.values()]
    labels = [s['label'] for s in styles.values()]
    legend_ax.legend(handles, labels, loc='center', title='Model & Ranking Method', fontsize=12)
        
    fig.supxlabel('Model Layer Index', fontsize=16)
    fig.supylabel(f'Top-{args.k_percentage:.0%} Retrieval Accuracy (F1-Score)', fontsize=16)
    axes[0, 0].set_ylim(bottom=0.0, top=1.05)
    
    fig.suptitle(f'Comparison of Ranking Methods vs. Oracle (Top-{args.k_percentage:.0%} Kept)', fontsize=20, y=0.98)
    
    plt.tight_layout(rect=[0.04, 0.02, 0.98, 0.95])
    plt.savefig(args.output_file, format='pdf', dpi=300)
    plt.close(fig)
    print(f"\nComparison plot saved to: {args.output_file}")


def main():
    parser = argparse.ArgumentParser(description="Visualize and compare ranking methods.")
    
    ORACLE_PATH = 'analysis_results/oracles/oracles_model_meta-llama_Llama-3.1-8B-Instruct.pkl'
    APPROX_8B_PATH = 'analysis_results/approx_rankings/approx_rankings_model_meta-llama_Llama-3.1-8B-Instruct.pkl'
    APPROX_1B_PATH = 'analysis_results/approx_rankings/approx_rankings_model_meta-llama_Llama-3.2-1B-Instruct.pkl'

    parser.add_argument("--datasets_to_plot", nargs='+', default=['qasper', 'multifieldqa_en', '2wikimqa', 'multi_news', 'trec', 'repobench-p'])
    parser.add_argument("--k_percentage", type=float, default=0.1, help="Percentage for top-k analysis (e.g., 0.1 for top 10%).")
    parser.add_argument("--output_file", type=str, default="ranking_comparison_full.pdf", help="Path to save the output PDF plot.")
    args = parser.parse_args()
    
    if not (0 < args.k_percentage <= 1):
        raise ValueError("--k_percentage must be between 0 and 1.")

    print("Loading data files...")
    try:
        all_results = {
            'oracle': load_data(ORACLE_PATH),
            '8B': load_data(APPROX_8B_PATH),
            '1B': load_data(APPROX_1B_PATH),
        }
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    plot_comparison_grid(all_results, args)

if __name__ == "__main__":
    main()
