import os
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch
import math
from typing import Dict, List, Any

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
    approx_rankings: Dict[int, np.ndarray],
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
        # Scores are already numpy arrays
        scores_tensor = torch.from_numpy(layer_scores).float()
        if scores_tensor.numel() == 0:
            accuracies[layer_idx] = 0.0
            continue
        
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

def load_npz_data_for_dataset(base_path: str, model_name_sanitized: str, dataset_name: str) -> Dict[str, Any]:
    """Loads all samples from a single .npz file for a given dataset."""
    file_path = os.path.join(base_path, model_name_sanitized, f"{dataset_name}.npz")
    if not os.path.exists(file_path):
        return {}
    
    try:
        # Load the entire .npz file, which contains keys like 'sample_0', 'sample_1', etc.
        npz_file = np.load(file_path, allow_pickle=True)
        # Convert the NpzFile object to a standard dictionary
        return {key: npz_file[key].item() for key in npz_file.files}
    except Exception as e:
        print(f"Warning: Could not load or parse {file_path}. Error: {e}")
        return {}

def load_all_data(models_to_load: Dict, datasets_to_plot: List[str]) -> Dict:
    """Loads all data from .npz files for the required models and datasets."""
    all_data = {}
    
    print("Loading data files...")
    for model_key, model_info in tqdm(models_to_load.items(), desc="Loading Models"):
        model_data_for_all_datasets = {}
        for dataset_name in datasets_to_plot:
            # Load all samples for this specific dataset from its .npz file
            dataset_samples = load_npz_data_for_dataset(
                model_info['base_path'], model_info['sanitized_name'], dataset_name
            )
            if dataset_samples:
                model_data_for_all_datasets[dataset_name] = dataset_samples
        
        if model_data_for_all_datasets:
            all_data[model_key] = model_data_for_all_datasets
            
    return all_data

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
    
    approx_model_keys = ['8B', '1B']
    oracle_data = all_results.get('oracle', {})

    for i, dataset_name in enumerate(tqdm(args.datasets_to_plot, desc="Plotting Datasets")):
        ax = axes_flat[i]
        
        # Check if we have oracle data for this dataset
        oracle_samples_for_dataset = oracle_data.get(dataset_name)
        if not oracle_samples_for_dataset:
            ax.set_title(f"{dataset_name.replace('_', ' ').title()} (No Oracle Data)")
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=14, color='gray')
            continue
        
        num_common_samples_map = {}
        for model_key in approx_model_keys:
            approx_samples_for_dataset = all_results.get(model_key, {}).get(dataset_name)
            if not approx_samples_for_dataset: continue

            # Find common samples based on sample keys (e.g., 'sample_0')
            common_sample_keys = sorted(oracle_samples_for_dataset.keys() & approx_samples_for_dataset.keys())
            num_common_samples_map[model_key] = len(common_sample_keys)
            
            layer_accuracies_cumulative = {}
            layer_accuracies_layerwise = {}
            
            for sample_key in common_sample_keys:
                o_data = oracle_samples_for_dataset[sample_key]
                a_data = approx_samples_for_dataset[sample_key]
                
                # Unpickle the ranking dictionaries if they are stored as bytes
                if isinstance(a_data.get('cumulative_rankings'), bytes):
                    a_data['cumulative_rankings'] = pickle.loads(a_data['cumulative_rankings'])
                if isinstance(a_data.get('layerwise_rankings'), bytes):
                    a_data['layerwise_rankings'] = pickle.loads(a_data['layerwise_rankings'])
                    
                oracle_ranking = torch.from_numpy(o_data['ranking']).float()

                cum_acc = calculate_retrieval_accuracy(a_data.get('cumulative_rankings', {}), oracle_ranking, args.k_percentage)
                for layer, acc in cum_acc.items():
                    if layer not in layer_accuracies_cumulative: layer_accuracies_cumulative[layer] = []
                    layer_accuracies_cumulative[layer].append(acc)

                lay_acc = calculate_retrieval_accuracy(a_data.get('layerwise_rankings', {}), oracle_ranking, args.k_percentage)
                for layer, acc in lay_acc.items():
                    if layer not in layer_accuracies_layerwise: layer_accuracies_layerwise[layer] = []
                    layer_accuracies_layerwise[layer].append(acc)

            # Plotting logic
            if layer_accuracies_cumulative:
                sorted_layers = sorted(layer_accuracies_cumulative.keys())
                mean_acc = [np.mean(layer_accuracies_cumulative[l]) for l in sorted_layers]
                ax.plot(sorted_layers, mean_acc, **styles[f"{model_key}_cumulative"], markersize=5, zorder=5)
            
            if layer_accuracies_layerwise:
                sorted_layers = sorted(layer_accuracies_layerwise.keys())
                mean_acc = [np.mean(layer_accuracies_layerwise[l]) for l in sorted_layers]
                ax.plot(sorted_layers, mean_acc, **styles[f"{model_key}_layerwise"], markersize=5, zorder=3)

        sample_count_str = ", ".join([f"{k}:{v}" for k, v in num_common_samples_map.items() if v > 0])
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
    fig.supylabel(f'Top-{args.k_percentage:.0%} Retrieval Accuracy', fontsize=16)
    axes[0, 0].set_ylim(bottom=0.0, top=1.05)
    
    fig.suptitle(f'Comparison of Ranking Methods vs. Oracle (Top-{args.k_percentage:.0%} Kept)', fontsize=20, y=0.98)
    
    plt.tight_layout(rect=[0.04, 0.02, 0.98, 0.95])
    plt.savefig(args.output_file, format='pdf', dpi=300)
    plt.close(fig)
    print(f"\nComparison plot saved to: {args.output_file}")


def main():
    parser = argparse.ArgumentParser(description="Visualize and compare ranking methods.")
    
    # Define models and their paths
    MODELS = {
        'oracle': {
            'full_name': 'meta-llama/Llama-3.1-8B-Instruct',
            'sanitized_name': 'meta-llama_Llama-3.1-8B-Instruct',
            'base_path': 'analysis_results/oracles'
        },
        '8B': {
            'full_name': 'meta-llama/Llama-3.1-8B-Instruct',
            'sanitized_name': 'meta-llama_Llama-3.1-8B-Instruct',
            'base_path': 'analysis_results/approx_rankings'
        },
        '1B': {
            'full_name': 'meta-llama/Llama-3.2-1B-Instruct',
            'sanitized_name': 'meta-llama_Llama-3.2-1B-Instruct',
            'base_path': 'analysis_results/approx_rankings'
        },
    }

    parser.add_argument("--datasets_to_plot", nargs='+', default=['qasper', 'multifieldqa_en', '2wikimqa', 'multi_news', 'trec', 'repobench-p'])
    parser.add_argument("--k_percentage", type=float, default=0.1, help="Percentage for top-k analysis (e.g., 0.1 for top 10%).")
    parser.add_argument("--output_file", type=str, default="ranking_comparison_full.pdf", help="Path to save the output PDF plot.")
    args = parser.parse_args()
    
    if not (0 < args.k_percentage <= 1):
        raise ValueError("--k_percentage must be between 0 and 1.")

    all_results = load_all_data(MODELS, args.datasets_to_plot)
    
    if not all_results:
        print("Error: No data was loaded. Please check file paths and ensure data exists.")
        return

    plot_comparison_grid(all_results, args)

if __name__ == "__main__":
    main()
