# analysis/visualize_rankings_comparison.py

import os
import argparse
import pickle
import math
from typing import Dict, List, Any
from collections import defaultdict

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

# Import shared constants and styling functions from viz_utils
from .viz_utils import (
    set_publication_style,
    METHOD_COLORS,
    TASKS_AND_DATASETS,
    ALL_DATASETS_TO_PLOT,
    TASK_SHORT_NAMES,
    DATASET_NAME_MAP,
)

# --- MODELS CONFIGURATION ---
MODELS_TO_LOAD = {
    'oracle': {
        'sanitized_name': 'meta-llama_Llama-3.1-8B-Instruct',
        'base_path': 'analysis_results/oracles'
    },
    '8B': {
        'sanitized_name': 'meta-llama_Llama-3.1-8B-Instruct',
        'base_path': 'analysis_results/approx_rankings'
    },
    '1B': {
        'sanitized_name': 'meta-llama_Llama-3.2-1B-Instruct',
        'base_path': 'analysis_results/approx_rankings'
    },
}

# --- DATA LOADING AND PROCESSING ---

def generate_dummy_data() -> Dict[str, Any]:
    """Generates random data for layout and debugging purposes."""
    print("--- Using --debug mode: Generating random dummy data ---")
    all_data = {'oracle': {}, '8B': {}, '1B': {}}
    for ds_name in ALL_DATASETS_TO_PLOT:
        all_data['oracle'][ds_name] = {'sample_0': {'ranking': np.random.rand(4096)}}
        all_data['8B'][ds_name] = {
            'sample_0': {
                'fastkv_rankings': {i: np.random.rand(4096) * 0.05 + 0.08 for i in range(32)},
                'gemfilter_rankings': {i: np.random.rand(4096) * 0.06 + 0.07 for i in range(32)}
            }
        }
        all_data['1B'][ds_name] = {
            'sample_0': {
                'speculative_rankings': {k: np.random.rand(4096) * 0.04 + (k/32)*0.03 + 0.09 for k in [1, 8, 32]}
            }
        }
    return all_data

def calculate_retrieval_accuracy(
    approx_rankings: Dict, oracle_ranking: torch.Tensor, k_percentage: float
) -> Dict[Any, float]:
    """Calculates the retrieval accuracy for approximate rankings against an oracle."""
    if not approx_rankings: return {}
    prompt_len = len(oracle_ranking)
    k = max(1, math.ceil(prompt_len * k_percentage))
    _, top_k_oracle_indices = torch.topk(oracle_ranking, k=k)
    oracle_set = set(top_k_oracle_indices.tolist())
    accuracies = {}
    for key, scores_np in approx_rankings.items():
        scores_tensor = torch.from_numpy(scores_np).float()[:prompt_len]
        if scores_tensor.numel() < k: continue
        _, top_k_approx_indices = torch.topk(scores_tensor, k=k)
        approx_set = set(top_k_approx_indices.tolist())
        accuracies[key] = len(oracle_set.intersection(approx_set)) / k
    return accuracies

def load_npz_data_for_dataset(base_path: str, model_name: str, dataset_name: str) -> Dict[str, Any]:
    """Loads and deserializes data from a single NPZ file for a given dataset."""
    file_path = os.path.join(base_path, model_name, f"{dataset_name}.npz")
    if not os.path.exists(file_path): return {}
    try:
        npz_file = np.load(file_path, allow_pickle=True)
        return {key: npz_file[key].item() for key in npz_file.files}
    except Exception: return {}

def load_all_data(
    models_to_load: Dict, datasets_to_plot: List[str]
) -> Dict[str, Any]:
    """Loads all required data across specified models and datasets."""
    all_data = {}
    for model_key, model_info in tqdm(models_to_load.items(), desc="Loading Models"):
        model_data = {
            ds: load_npz_data_for_dataset(
                model_info['base_path'], model_info['sanitized_name'], ds
            ) for ds in datasets_to_plot
        }
        all_data[model_key] = {k: v for k, v in model_data.items() if v}
    return all_data

def get_mean_accuracies(
    dataset_name: str, all_results: Dict, k_percentage: float
) -> Dict[str, Any]:
    """Computes mean retrieval accuracies for a single dataset, simplifying speculative results."""
    oracle_samples = all_results.get('oracle', {}).get(dataset_name, {})
    approx_8b = all_results.get('8B', {}).get(dataset_name, {})
    approx_1b = all_results.get('1B', {}).get(dataset_name, {})
    common_keys = sorted(list(set(oracle_samples.keys()) & set(approx_8b.keys()) & set(approx_1b.keys())))
    if not common_keys: return {}

    accs = {'fastkv': defaultdict(list), 'gemfilter': defaultdict(list), 'spec_prefill': []}
    k_priority = [8, 32, 1]  # Priority for speculative k: try 8, then 32, then 1

    for sample_key in common_keys:
        oracle_ranking = torch.from_numpy(oracle_samples[sample_key]['ranking']).float()
        for data_source in [approx_8b, approx_1b]:
            for rank_type in ['fastkv_rankings', 'gemfilter_rankings', 'speculative_rankings']:
                if rank_type in data_source[sample_key] and isinstance(data_source[sample_key][rank_type], bytes):
                    data_source[sample_key][rank_type] = pickle.loads(data_source[sample_key][rank_type])

        fk_acc = calculate_retrieval_accuracy(approx_8b[sample_key].get('fastkv_rankings', {}), oracle_ranking, k_percentage)
        gf_acc = calculate_retrieval_accuracy(approx_8b[sample_key].get('gemfilter_rankings', {}), oracle_ranking, k_percentage)
        sp_acc = calculate_retrieval_accuracy(approx_1b[sample_key].get('speculative_rankings', {}), oracle_ranking, k_percentage)

        for k, v in fk_acc.items(): accs['fastkv'][k].append(v)
        for k, v in gf_acc.items(): accs['gemfilter'][k].append(v)
        
        chosen_k_acc = None
        for k_val in k_priority:
            if k_val in sp_acc:
                chosen_k_acc = sp_acc[k_val]
                break
        if chosen_k_acc is not None:
            accs['spec_prefill'].append(chosen_k_acc)

    mean_accs = {}
    if accs['fastkv']: mean_accs['fastkv'] = {k: np.mean(v) for k, v in accs['fastkv'].items()}
    if accs['gemfilter']: mean_accs['gemfilter'] = {k: np.mean(v) for k, v in accs['gemfilter'].items()}
    if accs['spec_prefill']: mean_accs['spec_prefill'] = np.mean(accs['spec_prefill'])
    return mean_accs

def aggregate_accuracies_by_task(all_mean_accuracies: Dict, tasks_and_datasets: Dict) -> Dict:
    """Averages per-dataset accuracies into per-task-category accuracies."""
    agg = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for task, datasets in tasks_and_datasets.items():
        for ds in datasets:
            if ds in all_mean_accuracies:
                for method, data in all_mean_accuracies[ds].items():
                    if method == 'spec_prefill':
                        agg[task][method]['values'].append(data)
                    else:
                        for k, v in data.items(): agg[task][method][k].append(v)
    final_agg = defaultdict(dict)
    for task, methods in agg.items():
        for method, data in methods.items():
            if method == 'spec_prefill':
                final_agg[task][method] = np.mean(data['values'])
            else:
                final_agg[task][method] = {k: np.mean(v) for k, v in data.items()}
    return final_agg

def find_global_max_accuracy(all_accuracies: Dict) -> float:
    """Finds the maximum accuracy value for consistent y-axis scaling."""
    max_val = 0.0
    for acc_group in all_accuracies.values():
        for method, method_accs in acc_group.items():
            if method == 'spec_prefill':
                max_val = max(max_val, method_accs)
            elif method_accs:
                max_val = max(max_val, max(method_accs.values()))
    return max_val if max_val > 0 else 1.0


# --- PLOTTING ---
# CORRECTED: Updated PLOT_STYLES as per request
PLOT_STYLES = {
    'gemfilter':    {'color': METHOD_COLORS['GemFilter'], 'linestyle': '-', 'marker': 's', 'label': 'GemFilter'},
    'fastkv':       {'color': METHOD_COLORS['FastKV'], 'linestyle': '-', 'marker': 'o', 'label': 'FastKV'},
    'spec_prefill': {'color': METHOD_COLORS['Speculative'], 'linestyle': '--', 'label': 'Spec. Prefill'},
}

def plot_methods_on_ax(ax, accuracies: Dict):
    """Helper function to plot all ranking methods on a given Matplotlib axis."""
    for method in ['gemfilter', 'fastkv']:
        if method in accuracies and accuracies[method]:
            sorted_keys = sorted(accuracies[method].keys())
            # CORRECTED: Removed `markevery` to show all markers
            ax.plot(sorted_keys, [accuracies[method][k] for k in sorted_keys], **PLOT_STYLES[method])
    if 'spec_prefill' in accuracies:
        ax.axhline(y=accuracies['spec_prefill'], **PLOT_STYLES['spec_prefill'])

def create_appendix_legend(fig, gs_spec):
    """Creates a shared legend for the appendix figure."""
    legend_ax = fig.add_subplot(gs_spec)
    legend_ax.axis('off')
    handle_keys = ['gemfilter', 'fastkv', 'spec_prefill']
    handles = [plt.Line2D([0], [0], **{k:v for k,v in PLOT_STYLES[key].items() if k != 'label'}) for key in handle_keys]
    labels = [PLOT_STYLES[key]['label'] for key in handle_keys]
    legend_ax.legend(handles, labels, loc='center', ncol=1, title='Ranking Method', frameon=True, facecolor='white', framealpha=0.9, fontsize=24, title_fontsize=26)

def plot_paper_version(aggregated_accuracies: Dict, k_percentage: float, global_max_acc: float, output_prefix: str):
    """Creates the 2x3 grid with data averaged by task for the main paper."""
    set_publication_style()
    plt.rcParams.update({'lines.linewidth': 2.0, 'lines.markersize': 5})
    n_rows, n_cols = 2, 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 10), sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0.25, wspace=0.05)
    y_max_limit = global_max_acc * 1.25 if global_max_acc > 0 else 0.2
    plt.ylim(bottom=-0.05 * y_max_limit, top=y_max_limit)
    plt.xlim(left=-1, right=32)
    task_items = list(TASKS_AND_DATASETS.keys())
    for i, ax in enumerate(axes.flat):
        task_name = task_items[i]
        plot_methods_on_ax(ax, aggregated_accuracies.get(task_name, {}))
        ax.set_title(task_name, fontsize=28)
        ax.set_xticks([0, 8, 16, 24, 31])
        ax.grid(True, which='major', linestyle=':', linewidth=0.6)

    handle_keys = ['gemfilter', 'fastkv', 'spec_prefill']
    handles = [plt.Line2D([0], [0], **{k:v for k,v in PLOT_STYLES[key].items() if k != 'label'}) for key in handle_keys]
    labels = [PLOT_STYLES[key]['label'] for key in handle_keys]
    axes.flat[-1].legend(handles, labels, loc='lower right', ncol=1, title='Ranking Method',
                         frameon=True, facecolor='white', framealpha=0.9,
                         fontsize=18, title_fontsize=20)

    fig.text(0.5, 0.04, 'Model Layer Index', ha='center', va='center', fontsize=34)
    fig.text(0.06, 0.5, f'Top-{k_percentage:.0%} Retrieval Accuracy', ha='center', va='center', rotation='vertical', fontsize=34)
    fig.suptitle(f'Comparing Token Ranking Methods Against Oracle', y=0.99, fontsize=32, weight='bold')
    
    plt.savefig(f"{output_prefix}.pdf", format='pdf', dpi=300, bbox_inches='tight')
    print(f"\nPaper version plot saved to: {output_prefix}.pdf")
    plt.savefig(f"{output_prefix}.png", format='png', dpi=300, bbox_inches='tight')
    print(f"Paper version plot saved to: {output_prefix}.png")
    plt.close(fig)

def plot_appendix_version(all_mean_accuracies: Dict, k_percentage: float, global_max_acc: float, output_prefix: str):
    """Creates the detailed per-dataset grid plot for the appendix."""
    set_publication_style()
    plt.rcParams.update({'lines.linewidth': 1.5, 'lines.markersize': 4})
    n_rows, n_cols = 6, 3
    fig = plt.figure(figsize=(16, 28))
    gs = gridspec.GridSpec(n_rows, n_cols + 1, figure=fig, hspace=0.8, wspace=0.1, width_ratios=[0.2] + [1]*n_cols)
    y_max_limit = global_max_acc * 1.15 if global_max_acc > 0 else 0.2
    for row_idx, (task_name, datasets) in enumerate(TASKS_AND_DATASETS.items()):
        row_ax = fig.add_subplot(gs[row_idx, 0])
        row_ax.set_title(task_name, y=0.5, x=1.0, fontsize=28, weight='bold', rotation='vertical', ha='right', va='center')
        row_ax.axis('off')
        for col_idx, ds_name in enumerate(datasets):
            ax = fig.add_subplot(gs[row_idx, col_idx + 1])
            plot_methods_on_ax(ax, all_mean_accuracies.get(ds_name, {}))
            ax.set_title(DATASET_NAME_MAP.get(ds_name), fontsize=24)
            ax.set_xlim(left=-1, right=32); ax.set_ylim(bottom=-0.05 * y_max_limit, top=y_max_limit)
            ax.set_xticks([0, 8, 16, 24, 31]); ax.grid(True, which='major', linestyle=':', linewidth=0.6)
            if col_idx != 0: ax.tick_params(axis='y', labelleft=False)
    
    fig_legend_gs = gridspec.GridSpec(1, 1, top=0.08, bottom=0.01)
    create_appendix_legend(fig, fig_legend_gs[0])
    fig.text(0.5, 0.1, 'Model Layer Index', ha='center', va='center', fontsize=34)
    fig.text(0.06, 0.5, f'Top-{k_percentage:.0%} Retrieval Accuracy', ha='center', va='center', rotation='vertical', fontsize=34)
    fig.suptitle(f'Per-Dataset Comparison of Ranking Methods (Top-{k_percentage:.0%} Kept)', y=0.97, fontsize=38, weight='bold')
    fig.subplots_adjust(left=0.12, top=0.93, bottom=0.15, right=0.98)
    plt.savefig(f"{output_prefix}.pdf", format='pdf', dpi=300, bbox_inches='tight')
    print(f"\nAppendix version plot saved to: {output_prefix}.pdf")
    plt.savefig(f"{output_prefix}.png", format='png', dpi=300, bbox_inches='tight')
    print(f"Appendix version plot saved to: {output_prefix}.png")
    plt.close(fig)

def main():
    """Main execution block."""
    parser = argparse.ArgumentParser(description="Visualize and compare ranking methods.")
    parser.add_argument("--k_percentage", type=float, default=0.1, help="Percentage for top-k analysis.")
    parser.add_argument("--debug", action="store_true", help="Use dummy data for fast layout iteration.")
    args = parser.parse_args()
    output_dir = "figures"; os.makedirs(output_dir, exist_ok=True)
    if args.debug: results = generate_dummy_data()
    else: results = load_all_data(MODELS_TO_LOAD, ALL_DATASETS_TO_PLOT)
    all_mean_accuracies = {
        ds: get_mean_accuracies(ds, results, args.k_percentage)
        for ds in tqdm(ALL_DATASETS_TO_PLOT, desc="Calculating Accuracies")
    }
    all_mean_accuracies = {k: v for k, v in all_mean_accuracies.items() if v}
    if not all_mean_accuracies:
        print("No valid data found to plot. Exiting."); return
    
    aggregated_accuracies = aggregate_accuracies_by_task(all_mean_accuracies, TASKS_AND_DATASETS)
    paper_max_acc = find_global_max_accuracy(aggregated_accuracies)
    plot_paper_version(aggregated_accuracies, args.k_percentage, paper_max_acc, os.path.join(output_dir, "ranking_comparison"))

    appendix_max_acc = find_global_max_accuracy(all_mean_accuracies)
    plot_appendix_version(all_mean_accuracies, args.k_percentage, appendix_max_acc, os.path.join(output_dir, "ranking_comparison_appendix"))

if __name__ == "__main__":
    main()
