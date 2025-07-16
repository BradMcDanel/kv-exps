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

# --- Plotting Style (matched to the final reference image) ---
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 18,
    'axes.labelsize': 20,
    'axes.titlesize': 22,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 14,
    'grid.linestyle': ':',
    'grid.linewidth': 0.7,
})

def calculate_retrieval_accuracy(
    approx_rankings: Dict[int, np.ndarray],
    oracle_ranking: torch.Tensor,
    k_percentage: float
) -> Dict[int, float]:
    """Calculates the retrieval accuracy for each item's ranking against the oracle."""
    if not approx_rankings or k_percentage <= 0 or oracle_ranking.numel() == 0:
        return {}

    prompt_len = len(oracle_ranking)
    k = max(1, math.ceil(prompt_len * k_percentage))
    _, top_k_oracle_indices = torch.topk(oracle_ranking, k=k)
    oracle_set = set(top_k_oracle_indices.tolist())
    
    accuracies = {}
    for key, scores_np in approx_rankings.items():
        scores_tensor = torch.from_numpy(scores_np).float()
        
        # This check is crucial for correctness, ensuring we only use prompt scores
        if len(scores_tensor) > prompt_len:
            scores_tensor = scores_tensor[:prompt_len]

        if scores_tensor.numel() < k:
            continue

        _, top_k_approx_indices = torch.topk(scores_tensor, k=k)
        approx_set = set(top_k_approx_indices.tolist())
        
        intersection_size = len(oracle_set.intersection(approx_set))
        accuracy = intersection_size / k if k > 0 else 0.0
        accuracies[key] = accuracy
        
    return accuracies

def load_npz_data_for_dataset(base_path: str, model_name_sanitized: str, dataset_name: str) -> Dict[str, Any]:
    """Loads all samples from a single .npz file for a given dataset."""
    file_path = os.path.join(base_path, model_name_sanitized, f"{dataset_name}.npz")
    if not os.path.exists(file_path):
        return {}
    try:
        npz_file = np.load(file_path, allow_pickle=True)
        return {key: npz_file[key].item() for key in npz_file.files}
    except Exception as e:
        print(f"Warning: Could not load or parse {file_path}. Error: {e}")
        return {}

def load_all_data(models_to_load: Dict, datasets_to_plot: List[str]) -> Dict:
    """Loads all data from .npz files for the required models and datasets."""
    all_data = {}
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
    return all_data

def plot_focused_comparison(all_results: Dict, args: argparse.Namespace):
    """Creates a focused comparison plot with speculative methods as horizontal benchmarks."""
    num_datasets = len(args.datasets)
    fig, axes = plt.subplots(1, num_datasets, figsize=(10 * num_datasets, 7), squeeze=False)
    
    SPEC_K_POINTS = [1, 8, 32]
    
    styles = {
        '8B_fastkv':    {'color': '#d62728', 'linestyle': '-', 'marker': 'o', 'label': 'FastKV-style (8B)'},
        '8B_gemfilter':   {'color': '#ff7f0e', 'linestyle': '--', 'marker': 's', 'label': 'GemFilter-style (8B)'},
        '1B_spec_32':     {'color': '#2ca02c', 'linestyle': (0, (5, 2, 1, 2)), 'label': 'Speculative (1B, k=32)'}, # dash-dot-dot
        '1B_spec_8':      {'color': '#2ca02c', 'linestyle': '--', 'label': 'Speculative (1B, k=8)'}, # dashed
        '1B_spec_1':      {'color': '#2ca02c', 'linestyle': ':', 'label': 'Speculative (1B, k=1)'}, # dotted
    }
    
    oracle_data = all_results.get('oracle', {})
    approx_8b_data = all_results.get('8B', {})
    approx_1b_data = all_results.get('1B', {})

    for i, dataset_name in enumerate(tqdm(args.datasets, desc="Plotting Datasets")):
        ax = axes[0, i]
        oracle_samples = oracle_data.get(dataset_name, {})
        approx_8b_samples = approx_8b_data.get(dataset_name, {})
        approx_1b_samples = approx_1b_data.get(dataset_name, {})
        common_keys = sorted(list(set(oracle_samples.keys()) & set(approx_8b_samples.keys()) & set(approx_1b_samples.keys())))
        
        if not common_keys:
            ax.set_title(f"{dataset_name.replace('_', ' ').title()}\n(Missing Data)")
            continue

        # --- Aggregate 8B accuracies ---
        accs_8b = {'fastkv': {}, 'gemfilter': {}}
        for sample_key in common_keys:
            # ... (data loading and unpickling is the same)
            o_data = oracle_samples[sample_key]
            a_data = approx_8b_samples[sample_key]
            oracle_ranking = torch.from_numpy(o_data['ranking']).float()
            for rank_type in ['fastkv_rankings', 'gemfilter_rankings']:
                if isinstance(a_data.get(rank_type), bytes): a_data[rank_type] = pickle.loads(a_data[rank_type])
            fk_acc = calculate_retrieval_accuracy(a_data.get('fastkv_rankings', {}), oracle_ranking, args.k_percentage)
            gf_acc = calculate_retrieval_accuracy(a_data.get('gemfilter_rankings', {}), oracle_ranking, args.k_percentage)
            for key, acc in fk_acc.items(): accs_8b['fastkv'].setdefault(key, []).append(acc)
            for key, acc in gf_acc.items(): accs_8b['gemfilter'].setdefault(key, []).append(acc)

        # --- Aggregate 1B accuracies ---
        accs_1b_spec = {}
        for sample_key in common_keys:
            # ... (data loading and unpickling is the same)
            o_data = oracle_samples[sample_key]
            a_data = approx_1b_samples[sample_key]
            oracle_ranking = torch.from_numpy(o_data['ranking']).float()
            if isinstance(a_data.get('speculative_rankings'), bytes): a_data['speculative_rankings'] = pickle.loads(a_data['speculative_rankings'])
            sp_acc = calculate_retrieval_accuracy(a_data.get('speculative_rankings', {}), oracle_ranking, args.k_percentage)
            for key, acc in sp_acc.items(): accs_1b_spec.setdefault(key, []).append(acc)

        # --- Plotting ---
        if accs_8b['fastkv']:
            sorted_keys = sorted(accs_8b['fastkv'].keys())
            mean_acc = [np.mean(accs_8b['fastkv'][k]) for k in sorted_keys]
            ax.plot(sorted_keys, mean_acc, **styles['8B_fastkv'], markersize=7, linewidth=2)
        if accs_8b['gemfilter']:
            sorted_keys = sorted(accs_8b['gemfilter'].keys())
            mean_acc = [np.mean(accs_8b['gemfilter'][k]) for k in sorted_keys]
            ax.plot(sorted_keys, mean_acc, **styles['8B_gemfilter'], markersize=7, linewidth=2)

        for k_point in sorted(SPEC_K_POINTS, reverse=True): # Plot k=32 first
            if k_point in accs_1b_spec:
                mean_acc_spec = np.mean(accs_1b_spec[k_point])
                style_key = f'1B_spec_{k_point}'
                ax.axhline(y=mean_acc_spec, **styles[style_key], linewidth=2.5)

        ax.set_title(f"{dataset_name.replace('_', ' ').title()} (n={len(common_keys)})")
        ax.set_xlabel('Model Layer Index')

    # --- Final Figure Formatting ---
    axes[0, 0].set_ylabel(f'Top-{args.k_percentage:.0%} Retrieval Accuracy')
    axes[0, 0].set_ylim(bottom=0.0)
    for ax in axes.flatten():
        ax.set_xlim(left=0, right=31)

    handles = [plt.Line2D([0], [0], **s) for s in styles.values()]
    labels = [s['label'] for s in styles.values()]
    # Reorder for the legend to match the visual
    order = [0, 2, 4, 1, 3] 
    fig.legend([handles[i] for i in order], [labels[i] for i in order], loc='lower center', bbox_to_anchor=(0.5, -0.08), ncol=3, title='Ranking Method', frameon=False)
        
    fig.suptitle(f'Comparison of Ranking Methods vs. Oracle (Top-{args.k_percentage:.0%} Kept)', fontsize=24, y=1.0)
    plt.tight_layout(rect=[0, 0.08, 1, 0.92])
    plt.savefig(args.output_file, format='pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\nFocused comparison plot saved to: {args.output_file}")

def main():
    parser = argparse.ArgumentParser(description="Visualize and compare ranking methods.")
    MODELS = {
        'oracle': {'sanitized_name': 'meta-llama_Llama-3.1-8B-Instruct', 'base_path': 'analysis_results/oracles'},
        '8B': {'sanitized_name': 'meta-llama_Llama-3.1-8B-Instruct', 'base_path': 'analysis_results/approx_rankings'},
        '1B': {'sanitized_name': 'meta-llama_Llama-3.2-1B-Instruct', 'base_path': 'analysis_results/approx_rankings'},
    }
    parser.add_argument("--datasets", nargs='+', default=['qasper'], help="List of datasets to plot.")
    parser.add_argument("--k_percentage", type=float, default=0.2, help="Percentage for top-k analysis.")
    parser.add_argument("--output_file", type=str, default="ranking_comparison.pdf", help="Path to save the output PDF plot.")
    args = parser.parse_args()
    
    if not (0 < args.k_percentage <= 1): raise ValueError("--k_percentage must be between 0 and 1.")

    all_results = load_all_data(MODELS, args.datasets)
    if not all_results or 'oracle' not in all_results or '8B' not in all_results or '1B' not in all_results:
        print("Error: Missing data for oracle, 8B, or 1B models. Please check file paths.")
        return

    plot_focused_comparison(all_results, args)

if __name__ == "__main__":
    main()
