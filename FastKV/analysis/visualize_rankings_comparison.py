# analysis/visualize_rankings_comparison.py

import os
import argparse
import pickle
import math
from typing import Dict, List, Any

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
                'fastkv_rankings': {i: np.random.rand(4096) for i in range(32)},
                'gemfilter_rankings': {i: np.random.rand(4096) for i in range(32)}
            }
        }
        all_data['1B'][ds_name] = {
            'sample_0': {
                'speculative_rankings': {k: np.random.rand(4096) for k in [1, 8, 32]}
            }
        }
    return all_data

def calculate_retrieval_accuracy(
    approx_rankings: Dict, oracle_ranking: torch.Tensor, k_percentage: float
) -> Dict[Any, float]:
    """Calculates the retrieval accuracy for approximate rankings against an oracle."""
    if not approx_rankings:
        return {}

    prompt_len = len(oracle_ranking)
    k = max(1, math.ceil(prompt_len * k_percentage))
    _, top_k_oracle_indices = torch.topk(oracle_ranking, k=k)
    oracle_set = set(top_k_oracle_indices.tolist())

    accuracies = {}
    for key, scores_np in approx_rankings.items():
        scores_tensor = torch.from_numpy(scores_np).float()[:prompt_len]
        if scores_tensor.numel() < k:
            continue
        _, top_k_approx_indices = torch.topk(scores_tensor, k=k)
        approx_set = set(top_k_approx_indices.tolist())
        accuracies[key] = len(oracle_set.intersection(approx_set)) / k
    return accuracies

def load_npz_data_for_dataset(base_path: str, model_name: str, dataset_name: str) -> Dict[str, Any]:
    """Loads and deserializes data from a single NPZ file for a given dataset."""
    file_path = os.path.join(base_path, model_name, f"{dataset_name}.npz")
    if not os.path.exists(file_path):
        return {}
    try:
        npz_file = np.load(file_path, allow_pickle=True)
        return {key: npz_file[key].item() for key in npz_file.files}
    except Exception:
        return {}

def load_all_data(
    models_to_load: Dict, datasets_to_plot: List[str]
) -> Dict[str, Any]:
    """Loads all required data across specified models and datasets."""
    all_data = {}
    for model_key, model_info in tqdm(models_to_load.items(), desc="Loading Models"):
        model_data = {
            ds: load_npz_data_for_dataset(
                model_info['base_path'], model_info['sanitized_name'], ds
            )
            for ds in datasets_to_plot
        }
        all_data[model_key] = {k: v for k, v in model_data.items() if v}
    return all_data

def get_mean_accuracies(
    dataset_name: str, all_results: Dict, k_percentage: float
) -> Dict[str, Dict[Any, float]]:
    """Computes mean retrieval accuracies for a single dataset across all methods."""
    oracle_samples = all_results.get('oracle', {}).get(dataset_name, {})
    approx_8b = all_results.get('8B', {}).get(dataset_name, {})
    approx_1b = all_results.get('1B', {}).get(dataset_name, {})

    common_keys = sorted(
        list(set(oracle_samples.keys()) & set(approx_8b.keys()) & set(approx_1b.keys()))
    )
    if not common_keys:
        return {}

    accs = {'fastkv': {}, 'gemfilter': {}, 'speculative': {}}
    for sample_key in common_keys:
        oracle_ranking = torch.from_numpy(oracle_samples[sample_key]['ranking']).float()

        # Handle pickled data if necessary
        for data_source in [approx_8b, approx_1b]:
            for rank_type in ['fastkv_rankings', 'gemfilter_rankings', 'speculative_rankings']:
                if rank_type in data_source[sample_key] and isinstance(data_source[sample_key][rank_type], bytes):
                    data_source[sample_key][rank_type] = pickle.loads(data_source[sample_key][rank_type])

        # Calculate accuracies for each method
        fk_acc = calculate_retrieval_accuracy(approx_8b[sample_key].get('fastkv_rankings', {}), oracle_ranking, k_percentage)
        gf_acc = calculate_retrieval_accuracy(approx_8b[sample_key].get('gemfilter_rankings', {}), oracle_ranking, k_percentage)
        sp_acc = calculate_retrieval_accuracy(approx_1b[sample_key].get('speculative_rankings', {}), oracle_ranking, k_percentage)

        # Append results to accumulator
        for k, v in fk_acc.items(): accs['fastkv'].setdefault(k, []).append(v)
        for k, v in gf_acc.items(): accs['gemfilter'].setdefault(k, []).append(v)
        for k, v in sp_acc.items(): accs['speculative'].setdefault(k, []).append(v)

    # Compute the mean for each collected list of accuracies
    mean_accs = {
        method: {k: np.mean(v) for k, v in data.items()}
        for method, data in accs.items() if data
    }
    return mean_accs


# --- PLOTTING ---

def plot_grid_comparison(
    all_results: Dict, k_percentage: float, output_pdf_file: str, output_png_file: str
):
    """Creates and saves a publication-quality grid plot with task group labels."""
    set_publication_style()
    plt.rcParams.update({
        'lines.linewidth': 1.5,
        'lines.markersize': 6,
    })

    all_mean_accuracies = {
        ds: get_mean_accuracies(ds, all_results, k_percentage)
        for ds in ALL_DATASETS_TO_PLOT
    }

    n_rows, n_cols = 6, 3
    fig = plt.figure(figsize=(16, 32))
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.75, wspace=0.05)

    SPEC_K_POINTS = [1, 8, 32]
    styles = {
        'fastkv':    {'color': METHOD_COLORS['FastKV'], 'linestyle': '-', 'marker': 'o', 'label': 'FastKV (8B)'},
        'gemfilter': {'color': METHOD_COLORS['GemFilter'], 'linestyle': '-', 'marker': 's', 'label': 'GemFilter (8B)'},
        'spec_1':    {'color': METHOD_COLORS['Speculative (k=1)'], 'linestyle': ':', 'label': 'Speculative (1B, k=1)'},
        'spec_8':    {'color': METHOD_COLORS['Speculative (k=8)'], 'linestyle': '--', 'label': 'Speculative (1B, k=8)'},
        'spec_32':   {'color': METHOD_COLORS['Speculative (k=32)'], 'linestyle': (0, (5, 2, 1, 2)), 'label': 'Speculative (1B, k=32)'},
    }

    plot_idx = 0
    for row_idx, (task_name, datasets) in enumerate(TASKS_AND_DATASETS.items()):
        # Add a title for the entire row of plots (task category)
        row_ax = fig.add_subplot(gs[row_idx, :])
        row_ax.set_title(task_name, y=1.20, fontsize=28, weight='bold', color='black')
        row_ax.axis('off')

        for col_idx, dataset_name in enumerate(datasets):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            dataset_accuracies = all_mean_accuracies.get(dataset_name, {})
            pretty_name = DATASET_NAME_MAP.get(dataset_name)
            task_short_name = TASK_SHORT_NAMES[task_name]

            if not dataset_accuracies:
                ax.text(0.5, 0.5, f"{pretty_name}\n(No Data)", ha='center', va='center', style='italic')
            else:
                # Plot FastKV and GemFilter (lines with markers)
                for method, style_key in [('fastkv', 'fastkv'), ('gemfilter', 'gemfilter')]:
                    if method in dataset_accuracies:
                        sorted_keys = sorted(dataset_accuracies[method].keys())
                        ax.plot(sorted_keys, [dataset_accuracies[method][k] for k in sorted_keys], **styles[style_key])

                # Plot Speculative methods (horizontal lines)
                spec_data = dataset_accuracies.get('speculative', {})
                for k_point in sorted(SPEC_K_POINTS, reverse=True):
                    if k_point in spec_data:
                        ax.axhline(y=spec_data[k_point], **styles[f'spec_{k_point}'])

            ax.set_title(f"{pretty_name}")
            ax.set_xlim(left=-1, right=32)
            ax.set_ylim(bottom=-0.05, top=1.05)
            ax.set_xticks([0, 8, 16, 24, 31])
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.tick_params(axis='x', labelbottom=True)
            if col_idx != 0:
                ax.tick_params(axis='y', labelleft=False)
            plot_idx += 1

    # Hide unused subplots before placing the legend
    all_axes = fig.get_axes()
    for i in range(plot_idx, len(all_axes)):
        # Don't hide the row title axes
        if not all_axes[i].get_title():
            all_axes[i].axis('off')

    # Place a single, shared legend in an empty subplot space
    legend_ax = fig.add_subplot(gs[5, 2])
    legend_ax.axis('off')
    handle_keys = ['fastkv', 'gemfilter', 'spec_32', 'spec_8', 'spec_1']
    handles = [plt.Line2D([0], [0], **{k:v for k,v in styles[key].items() if k!='label'}) for key in handle_keys]
    labels = [styles[key]['label'] for key in handle_keys]
    legend_ax.legend(handles, labels, loc='center', ncol=1, title='Ranking Method', frameon=True, facecolor='white', framealpha=0.9)

    # Add shared axis labels and a main title
    fig.text(0.5, 0.06, 'Model Layer Index', ha='center', va='center', fontsize=34)
    fig.text(0.02, 0.5, f'Top-{k_percentage:.0%} Retrieval Accuracy', ha='center', va='center', rotation='vertical', fontsize=34)
    fig.suptitle(f'Comparison of Ranking Methods vs. Oracle (Top-{k_percentage:.0%} Kept)', y=0.99)
    fig.subplots_adjust(left=0.1, top=0.93, bottom=0.1, right=0.98)

    # Save the figure in multiple formats
    plt.savefig(output_pdf_file, format='pdf', dpi=300, bbox_inches='tight')
    print(f"\nGrid plot saved to: {output_pdf_file}")
    plt.savefig(output_png_file, format='png', dpi=300, bbox_inches='tight')
    print(f"Grid plot saved to: {output_png_file}")
    plt.close(fig)


def main():
    """Main execution block."""
    parser = argparse.ArgumentParser(description="Visualize and compare ranking methods.")
    parser.add_argument("--k_percentage", type=float, default=0.1, help="Percentage for top-k analysis.")
    parser.add_argument("--debug", action="store_true", help="Use dummy data for fast layout iteration.")
    args = parser.parse_args()

    # Define output paths
    output_dir = "figures"
    os.makedirs(output_dir, exist_ok=True)
    output_pdf = os.path.join(output_dir, "ranking_comparison.pdf")
    output_png = os.path.join(output_dir, "ranking_comparison.png")

    # Load data or generate dummy data
    if args.debug:
        results = generate_dummy_data()
    else:
        results = load_all_data(MODELS_TO_LOAD, ALL_DATASETS_TO_PLOT)

    # Generate and save the plot
    plot_grid_comparison(results, args.k_percentage, output_pdf, output_png)


if __name__ == "__main__":
    main()
