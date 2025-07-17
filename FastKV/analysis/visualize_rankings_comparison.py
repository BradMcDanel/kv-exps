# analysis/visualize_rankings_comparison.py

import os
import argparse
from typing import Dict, List, Any

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

from .viz_utils import (
    set_publication_style,
    METHOD_COLORS,
    TASKS_AND_DATASETS,
    ALL_DATASETS_TO_PLOT,
    DATASET_NAME_MAP,
)

from .retrieval_metrics import (
    load_npz_data_for_dataset,
    get_mean_accuracies,
    aggregate_accuracies_by_task,
    find_global_max_accuracy,
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

# --- DATA LOADING ---
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


# --- PLOTTING ---
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
    legend_ax.legend(handles, labels, loc='center', ncol=1, title='Pruning Heuristic', frameon=True, facecolor='white', framealpha=0.9, fontsize=24, title_fontsize=26)

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
    axes.flat[-1].legend(handles, labels, loc='lower right', ncol=1, title='Pruning Heuristic',
                         frameon=True, facecolor='white', framealpha=0.9,
                         fontsize=18, title_fontsize=20)

    fig.text(0.5, 0.04, 'Model Layer Index', ha='center', va='center', fontsize=34)
    fig.text(0.06, 0.5, f'Top-{k_percentage:.0%} Token Overlap with Oracle', ha='center', va='center', rotation='vertical', fontsize=34)
    fig.suptitle(f'Evaluating the Ranking Fidelity of KV Pruning Heuristics', y=0.99, fontsize=32, weight='bold')
    
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
    fig.text(0.06, 0.5, f'Top-{k_percentage:.0%} Token Overlap with Oracle', ha='center', va='center', rotation='vertical', fontsize=34)
    fig.suptitle(f'Evaluating the Ranking Fidelity of KV Pruning Heuristics', y=0.97, fontsize=38, weight='bold')
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
    if args.debug:
        results = generate_dummy_data()
    else:
        results = load_all_data(MODELS_TO_LOAD, ALL_DATASETS_TO_PLOT)
    
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
