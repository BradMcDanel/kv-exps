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
from typing import Dict, List, Any

# --- Plotting Style ---
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 13,
    'axes.labelsize': 15,
    'axes.titlesize': 17,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
})

def calculate_retrieval_accuracy(
    approx_rankings: Dict[int, np.ndarray],
    oracle_ranking: torch.Tensor,
    k_percentage: float
) -> Dict[int, float]:
    """Calculates the retrieval accuracy for each item's ranking against the oracle."""
    if not approx_rankings or k_percentage <= 0 or oracle_ranking.numel() == 0:
        return {}

    k = max(1, math.ceil(len(oracle_ranking) * k_percentage))
    _, top_k_oracle_indices = torch.topk(oracle_ranking, k=k)
    oracle_set = set(top_k_oracle_indices.tolist())
    
    accuracies = {}
    for key, scores_np in approx_rankings.items():
        scores_tensor = torch.from_numpy(scores_np).float()
        if scores_tensor.numel() == 0:
            accuracies[key] = 0.0
            continue
        
        num_scores = len(scores_tensor)
        current_k = min(k, num_scores)
        if current_k == 0:
            accuracies[key] = 0.0
            continue

        _, top_k_approx_indices = torch.topk(scores_tensor, k=current_k)
        approx_set = set(top_k_approx_indices.tolist())
        
        intersection_size = len(oracle_set.intersection(approx_set))
        accuracy = intersection_size / k if k > 0 else 0.0
        accuracies[key] = accuracy
        
    return accuracies

def load_npz_data_for_dataset(base_path: str, model_name_sanitized: str, dataset_name: str) -> Dict[str, Any]:
    """Loads all samples from a single .npz file for a given dataset."""
    file_path = os.path.join(base_path, model_name_sanitized, f"{dataset_name}.npz")
    if not os.path.exists(file_path):
        print(f"[DEBUG] NPZ file not found at: {file_path}")
        return {}
    
    try:
        npz_file = np.load(file_path, allow_pickle=True)
        print(f"[DEBUG] Successfully loaded {len(npz_file.files)} samples from {file_path}")
        # Unpack the 0-d array to get the dictionary inside
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
            dataset_samples = load_npz_data_for_dataset(
                model_info['base_path'], model_info['sanitized_name'], dataset_name
            )
            if dataset_samples:
                model_data_for_all_datasets[dataset_name] = dataset_samples
        if model_data_for_all_datasets:
            all_data[model_key] = model_data_for_all_datasets
            print(f"[DEBUG] Loaded data for model '{model_key}'. Datasets: {list(model_data_for_all_datasets.keys())}")
    return all_data

def process_and_plot(ax: plt.Axes, results_by_key: Dict, plot_styles: Dict, model_key: str, method_key: str):
    """Helper to calculate mean accuracies and plot them."""
    if not results_by_key:
        print(f"[DEBUG] process_and_plot: No results to plot for {model_key} - {method_key}")
        return
    sorted_keys = sorted(results_by_key.keys())
    mean_acc = [np.mean(results_by_key[k]) for k in sorted_keys]
    style_key = f"{model_key}_{method_key}"
    if style_key in plot_styles:
        print(f"[DEBUG] Plotting {model_key} - {method_key} with {len(sorted_keys)} data points.")
        ax.plot(sorted_keys, mean_acc, **plot_styles[style_key], markersize=6)

def plot_comparison_grid(all_results: Dict, args: argparse.Namespace):
    """Creates a comparison grid for ranking methods across models and datasets."""
    num_datasets = len(args.datasets)
    fig, axes = plt.subplots(2, num_datasets, figsize=(6 * num_datasets, 10), sharey='row', squeeze=False)
    
    styles = {
        '8B_fastkv': {'color': '#d62728', 'linestyle': '-', 'marker': 'o', 'label': '8B FastKV-style'},
        '8B_gemfilter': {'color': '#ff7f0e', 'linestyle': '--', 'marker': 's', 'label': '8B GemFilter-style'},
        '8B_speculative': {'color': '#2ca02c', 'linestyle': '-.', 'marker': '^', 'label': '8B Speculative'},
        '1B_fastkv': {'color': '#1f77b4', 'linestyle': '-', 'marker': 'o', 'label': '1B FastKV-style'},
        '1B_gemfilter': {'color': '#9467bd', 'linestyle': '--', 'marker': 's', 'label': '1B GemFilter-style'},
        '1B_speculative': {'color': '#8c564b', 'linestyle': '-.', 'marker': '^', 'label': '1B Speculative'},
    }
    
    approx_model_keys = ['8B', '1B']
    oracle_data = all_results.get('oracle', {})

    for i, dataset_name in enumerate(tqdm(args.datasets, desc="Plotting Datasets")):
        ax_layerwise = axes[0, i]
        ax_speculative = axes[1, i]

        oracle_samples = oracle_data.get(dataset_name)
        if not oracle_samples:
            for ax in [ax_layerwise, ax_speculative]:
                ax.set_title(f"{dataset_name.replace('_', ' ').title()}\n(No Oracle Data)")
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=14, color='gray')
            continue

        num_common_samples_map = {}
        for model_key in approx_model_keys:
            approx_samples = all_results.get(model_key, {}).get(dataset_name)
            if not approx_samples:
                print(f"[DEBUG] No approx data found for model '{model_key}' on dataset '{dataset_name}'")
                continue

            oracle_keys = set(oracle_samples.keys())
            approx_keys = set(approx_samples.keys())
            common_keys = sorted(list(oracle_keys & approx_keys))
            print(f"[DEBUG] ----- Model: {model_key}, Dataset: {dataset_name} -----")
            print(f"[DEBUG] Oracle samples: {len(oracle_keys)}. Approx samples: {len(approx_keys)}.")
            print(f"[DEBUG] Found {len(common_keys)} common sample keys. First 5: {common_keys[:5]}")
            if not common_keys:
                print(f"[WARNING] No common samples found between oracle and {model_key} for {dataset_name}!")
                continue

            num_common_samples_map[model_key] = len(common_keys)
            
            accs = {'fastkv': {}, 'gemfilter': {}, 'speculative': {}}
            
            for sample_key in common_keys:
                o_data = oracle_samples[sample_key]
                a_data = approx_samples[sample_key]
                oracle_ranking = torch.from_numpy(o_data['ranking']).float()
                
                # Unpickle all ranking types
                for rank_type in ['fastkv_rankings', 'gemfilter_rankings', 'speculative_rankings']:
                    if isinstance(a_data.get(rank_type), bytes):
                        a_data[rank_type] = pickle.loads(a_data[rank_type])
                
                # DEBUG print for the first sample
                if sample_key == common_keys[0]:
                    for rank_type in ['fastkv', 'gemfilter', 'speculative']:
                        if a_data.get(f'{rank_type}_rankings'):
                            print(f"[DEBUG] Unpickled '{rank_type}_rankings' for sample {sample_key}. Keys: {list(a_data[f'{rank_type}_rankings'].keys())[:5]}")

                fk_acc = calculate_retrieval_accuracy(a_data.get('fastkv_rankings', {}), oracle_ranking, args.k_percentage)
                gf_acc = calculate_retrieval_accuracy(a_data.get('gemfilter_rankings', {}), oracle_ranking, args.k_percentage)
                sp_acc = calculate_retrieval_accuracy(a_data.get('speculative_rankings', {}), oracle_ranking, args.k_percentage)
                
                for key, acc in fk_acc.items():
                    accs['fastkv'].setdefault(key, []).append(acc)
                for key, acc in gf_acc.items():
                    accs['gemfilter'].setdefault(key, []).append(acc)
                for key, acc in sp_acc.items():
                    accs['speculative'].setdefault(key, []).append(acc)

            print(f"[DEBUG] Aggregated accuracies for {model_key}:")
            for name, data in accs.items():
                print(f"  - {name}: {len(data)} keys. First key: {list(data.keys())[0] if data else 'N/A'}")
            
            process_and_plot(ax_layerwise, accs['fastkv'], styles, model_key, 'fastkv')
            process_and_plot(ax_layerwise, accs['gemfilter'], styles, model_key, 'gemfilter')
            process_and_plot(ax_speculative, accs['speculative'], styles, model_key, 'speculative')

        sample_count_str = ", ".join([f"{k}: {v}" for k, v in num_common_samples_map.items() if v > 0])
        title_dataset = dataset_name.replace('_', ' ').title()
        ax_layerwise.set_title(f"{title_dataset} (n={sample_count_str})")
        ax_speculative.set_title(f"{title_dataset} (n={sample_count_str})")
        ax_layerwise.grid(True, which='both', linestyle=':', linewidth=0.7)
        ax_speculative.grid(True, which='both', linestyle=':', linewidth=0.7)

    # --- Final Figure Formatting ---
    axes[0, 0].set_ylabel(f'Top-{args.k_percentage:.0%} Retrieval Accuracy')
    axes[1, 0].set_ylabel(f'Top-{args.k_percentage:.0%} Retrieval Accuracy')
    
    for i in range(num_datasets):
        axes[0, i].set_xlabel('Model Layer Index')
        axes[1, i].set_xlabel('Speculative Lookahead (k)')

    if any(ax.has_data() for ax in axes[0]): axes[0, 0].set_ylim(bottom=0.0)
    if any(ax.has_data() for ax in axes[1]): axes[1, 0].set_ylim(bottom=0.0)

    handles = [plt.Line2D([0], [0], **s) for s in styles.values()]
    labels = [s['label'] for s in styles.values()]
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=3, title='Model & Ranking Method')
        
    fig.suptitle(f'Comparison of Ranking Methods vs. Oracle (Top-{args.k_percentage:.0%} Kept)', fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(args.output_file, format='pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\nComparison plot saved to: {args.output_file}")

def main():
    parser = argparse.ArgumentParser(description="Visualize and compare ranking methods.")
    
    # FIX 1: The MODELS dictionary is now complete with all required keys.
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
            'full_name': 'meta-llama/Llama-3.1-8B-Instruct', # Note: Your log shows you used 3.2-1B, correcting here
            'sanitized_name': 'meta-llama_Llama-3.2-1B-Instruct',
            'base_path': 'analysis_results/approx_rankings'
        },
    }

    # FIX 2: Use a consistent argument name ('--datasets') and ensure it's a list.
    parser.add_argument("--datasets", nargs='+', default=['qasper'],
                        help="List of datasets to plot.")
    parser.add_argument("--k_percentage", type=float, default=0.1, help="Percentage for top-k analysis (e.g., 0.1 for top 10%).")
    parser.add_argument("--output_file", type=str, default="ranking_comparison_full.pdf", help="Path to save the output PDF plot.")
    args = parser.parse_args()
    
    if len(args.datasets) > 6:
        print(f"Warning: More than 6 datasets provided. The plot may be crowded.")

    if not (0 < args.k_percentage <= 1):
        raise ValueError("--k_percentage must be between 0 and 1.")

    all_results = load_all_data(MODELS, args.datasets)
    if not all_results or 'oracle' not in all_results:
        print("Error: No data was loaded, or oracle data is missing. Please check file paths and ensure generation scripts ran correctly.")
        return

    plot_comparison_grid(all_results, args)

if __name__ == "__main__":
    main()
