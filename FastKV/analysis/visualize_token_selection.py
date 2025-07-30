# analysis/visualize_token_selection.py

import argparse
import os
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import gaussian_kde
from tqdm import tqdm

from .viz_utils import (
    set_publication_style,
    METHOD_COLORS,
    DATASET_TO_TASK_MAP,
    DATASET_NAME_MAP,
)
from .retrieval_metrics import (
    load_npz_data_for_dataset,
    get_top_k_indices,
    deserialize_rankings_in_sample,
)

# ==============================================================================
# SCRIPT REVISION NOTES:
# This script has been updated based on feedback to create a more robust and
# impactful visualization.
#
# KEY CHANGES:
# 1. AGGREGATION: Instead of plotting a single "cherry-picked" sample, this
#    script now aggregates data across ALL valid samples in a dataset. This
#    provides a statistically meaningful view of token importance.
# 2. NORMALIZATION: All token positions are normalized to a [0, 1] scale
#    to allow for aggregation of prompts with varying lengths.
# 3. VISUAL SIMPLIFICATION:
#    - The Oracle is now a gray, filled background area to serve as a clear
#      ground-truth reference.
#    - The plot focuses on comparing CLAA against its main competitors
#      (FastKV, GemFilter) and removes Speculative Prefill for clarity.
#    - The text box with accuracy percentages has been removed to de-clutter
#      the plot, as this information is better suited for tables.
# ==============================================================================


# ==============================================================================
# DATA LOADING AND PROCESSING
# ==============================================================================

def generate_dummy_data_for_grid(k_percentage: float) -> Dict[str, Any]:
    """Generates random aggregated data for layout and debugging purposes."""
    print("--- Using --debug mode: Generating random dummy data ---")
    all_plot_data = {}
    task_configs = {
        '2WikiMQA': {'id': '2wikimqa', 'task_category': 'Multi-Doc QA'},
        'GovReport': {'id': 'gov_report', 'task_category': 'Summarization'},
    }
    # Omit Speculative Prefill for visual clarity in this plot
    methods = ['Oracle', 'FastKV', 'GemFilter', 'CLAA']
    num_dummy_samples = 100

    for task_name, config in task_configs.items():
        dataset_id = config['id']
        aggregated_indices = {method: [] for method in methods}
        
        for _ in range(num_dummy_samples):
            # Simulate different sequence lengths for each sample
            seq_len = np.random.randint(2000, 6000)
            k_absolute = int(seq_len * k_percentage)
            
            for method in methods:
                if dataset_id == '2wikimqa':  # Multi-peak distribution
                    means = np.array([0.1, 0.4, 0.75])
                    stds = np.array([0.05, 0.08, 0.06])
                    if method != 'Oracle': # Add noise to baselines
                        means += np.random.normal(0, 0.05, size=means.shape)
                else:  # GovReport-like distribution (start/end heavy)
                    means = np.array([0.1, 0.9])
                    stds = np.array([0.1, 0.1])
                    if method != 'Oracle':
                        means += np.random.normal(0, 0.08, size=means.shape)

                # Generate indices and normalize them
                indices_list = []
                for mean, std in zip(means, stds):
                    num_points = k_absolute // len(means)
                    sample_indices = torch.normal(mean=mean * seq_len, std=std * seq_len, size=(num_points,)).long()
                    indices_list.append(sample_indices)

                dummy_indices = torch.clamp(torch.cat(indices_list), 0, seq_len - 1)
                normalized_indices = dummy_indices.float() / seq_len
                aggregated_indices[method].append(normalized_indices)

        # Concatenate all normalized indices for the final plot data
        final_indices = {m: torch.cat(aggregated_indices[m]) for m in methods}
        
        all_plot_data[task_name] = {
            'name': DATASET_NAME_MAP.get(dataset_id, dataset_id.title()),
            'task_category': config['task_category'],
            'indices': final_indices,
            'seq_len': 1.0, # Sequence length is now 1.0 due to normalization
        }
    return all_plot_data

# ==============================================================================
# CORE PLOTTING LOGIC
# ==============================================================================

def plot_aggregated_density(
    ax: plt.Axes,
    aggregated_indices: Dict[str, torch.Tensor],
    plot_order_map: Dict,
):
    """Plots the aggregated density for one task on a given axis."""
    x_grid = np.linspace(0, 1, 1000)

    # First, plot the Oracle as a gray, filled background
    if 'Oracle' in aggregated_indices:
        indices_np = aggregated_indices['Oracle'].cpu().numpy()
        if len(indices_np) > 1:
            try:
                kde = gaussian_kde(indices_np, bw_method=0.03)
                density = kde(x_grid)
                ax.fill_between(x_grid, density, color='gray', alpha=0.3, label='Oracle (Ground Truth)')
            except (np.linalg.LinAlgError, ValueError):
                print("Warning: KDE failed for 'Oracle'.")

    # Then, plot the other methods as lines on top
    for key, props in plot_order_map.items():
        if key == 'Oracle' or key not in aggregated_indices:
            continue # Skip oracle as it's already plotted

        indices_np = aggregated_indices[key].cpu().numpy()
        if len(indices_np) > 1:
            try:
                kde = gaussian_kde(indices_np, bw_method=0.03)
                density = kde(x_grid)
                ax.plot(x_grid, density, color=props['color'], label=props['label'], linewidth=2.5)
            except (np.linalg.LinAlgError, ValueError):
                print(f"Warning: KDE failed for '{key}'. Plotting a histogram.")
                ax.hist(indices_np, bins=50, density=True, color=props['color'], alpha=0.5, label=f"{props['label']} (hist)")

    ax.set_xlim(0, 1)
    ax.set_yticklabels([])
    ax.tick_params(axis='y', length=0)
    ax.spines[['right', 'top', 'left']].set_visible(False)

def plot_density_grid(
    all_plot_data: Dict[str, Any],
    k_percentage: float,
    output_pdf_file: str,
    output_png_file: str
):
    """Creates the 1x2 grid plot showing aggregated token selection density."""
    set_publication_style()
    plt.rcParams.update({'axes.labelsize': 30, 'xtick.labelsize': 26, 'legend.fontsize': 24})

    fig, axes = plt.subplots(1, 2, figsize=(24, 8), sharey=True)
    fig.suptitle(f'Aggregated Positional Distribution of Top-{int(100.*k_percentage)}% Ranked Tokens', fontsize=32, weight='bold')

    # Define plot order and styles, omitting Speculative Prefill for clarity
    plot_order_map = {
        'Oracle': {'label': 'Oracle (Ground Truth)', 'color': 'gray'},
        'FastKV': {'label': 'FastKV', 'color': METHOD_COLORS['FastKV']},
        'GemFilter': {'label': 'GemFilter', 'color': METHOD_COLORS['GemFilter']},
        'CLAA': {'label': 'CLAA (Ours)', 'color': METHOD_COLORS['CLAA']},
    }

    task_order = ['2WikiMQA', 'GovReport']
    for i, task_name in enumerate(task_order):
        ax = axes[i]
        if task_name not in all_plot_data:
            ax.text(0.5, 0.5, f"Data not found for\n{task_name}", ha='center', va='center', style='italic', fontsize=20)
            ax.set_title(f"{task_name}", fontsize=28)
            ax.set_xticks([]); ax.set_yticks([])
            continue

        data = all_plot_data[task_name]
        dataset_name = data['name']
        task_category = data.get('task_category', '')

        plot_aggregated_density(ax, data['indices'], plot_order_map)
        
        title_text = f"{dataset_name}\n({task_category})"
        ax.set_title(title_text, fontsize=28)
        
        ax.set_xlabel('Normalized Token Position in Prompt')

    axes[0].set_ylabel('Density of Selected Tokens')
    
    # Create legend below the plots
    handles = [plt.Rectangle((0,0),1,1, color='gray', alpha=0.3)] + \
              [plt.Line2D([0], [0], color=props['color'], lw=4) for key, props in plot_order_map.items() if key != 'Oracle']
    labels = [props['label'] for props in plot_order_map.values()]
    
    fig.legend(handles, labels, loc='lower center',
               bbox_to_anchor=(0.5, -0.05), ncol=len(labels), frameon=False)

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    plt.savefig(output_pdf_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved PDF to: {output_pdf_file}")
    plt.savefig(output_png_file, dpi=300, bbox_inches='tight')
    print(f"Saved PNG to: {output_png_file}")
    plt.close(fig)

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate aggregated token selection density plots.")
    parser.add_argument("--k_percentage", type=float, default=0.1, help="Top-k percentage for token selection.")
    parser.add_argument("--debug", action="store_true", help="Use dummy data for fast layout iteration.")
    args = parser.parse_args()

    output_dir = "figures"
    os.makedirs(output_dir, exist_ok=True)
    # Use a more descriptive filename for the new plot
    output_pdf = os.path.join(output_dir, "token_selection_aggregated.pdf")
    output_png = os.path.join(output_dir, "token_selection_aggregated.png")

    if args.debug:
        all_plot_data = generate_dummy_data_for_grid(args.k_percentage)
        plot_density_grid(all_plot_data, args.k_percentage, output_pdf, output_png)
        return

    # --- Real Data Loading and Processing ---
    TARGET_MODEL = 'meta-llama/Llama-3.1-8B-Instruct'
    MODELS = {
        'oracle': {'sanitized_name': TARGET_MODEL.replace('/', '_'), 'base_path': 'analysis_results/oracles'},
        'approx_target': {'sanitized_name': TARGET_MODEL.replace('/', '_'), 'base_path': 'analysis_results/approx_rankings'},
    }
    
    DATASETS_BY_TASK = {'2WikiMQA': '2wikimqa', 'GovReport': 'gov_report'}
    all_plot_data = {}
    print("Loading and processing data for the aggregated grid plot...")
    
    fastkv_layer, gemfilter_layer, claa_layer = 15, 15, 15
    
    for task_name, dataset_id in tqdm(DATASETS_BY_TASK.items(), desc="Processing datasets"):
        oracle_data = load_npz_data_for_dataset(MODELS['oracle']['base_path'], MODELS['oracle']['sanitized_name'], dataset_id)
        approx_target_data = load_npz_data_for_dataset(MODELS['approx_target']['base_path'], MODELS['approx_target']['sanitized_name'], dataset_id)
        
        common_keys = sorted(list(set(oracle_data.keys()) & set(approx_target_data.keys())))
        if not common_keys: continue
        
        aggregated_indices: Dict[str, List[torch.Tensor]] = {'Oracle': [], 'FastKV': [], 'GemFilter': [], 'CLAA': []}

        # --- AGGREGATION LOGIC ---
        # Loop over all common samples instead of picking one
        for key in tqdm(common_keys, desc=f"Aggregating {dataset_id}", leave=False):
            oracle_sample = oracle_data.get(key, {})
            approx_target_sample = approx_target_data.get(key, {})
            deserialize_rankings_in_sample(approx_target_sample)

            if 'ranking' not in oracle_sample: continue
            
            seq_len = len(oracle_sample['ranking'])
            if seq_len == 0: continue
            k_absolute = int(seq_len * args.k_percentage)
            
            # --- Collect indices for each method and normalize ---
            oracle_ranking = torch.from_numpy(oracle_sample['ranking']).float()
            oracle_indices = get_top_k_indices(oracle_ranking, k_absolute, seq_len)
            aggregated_indices['Oracle'].append(oracle_indices.float() / seq_len)

            fk_rankings = approx_target_sample.get('fastkv_rankings', {})
            if fastkv_layer in fk_rankings:
                fk_indices = get_top_k_indices(torch.from_numpy(fk_rankings[fastkv_layer]).float(), k_absolute, seq_len)
                aggregated_indices['FastKV'].append(fk_indices.float() / seq_len)
            
            gf_rankings = approx_target_sample.get('gemfilter_rankings', {})
            if gemfilter_layer in gf_rankings:
                gf_indices = get_top_k_indices(torch.from_numpy(gf_rankings[gemfilter_layer]).float(), k_absolute, seq_len)
                aggregated_indices['GemFilter'].append(gf_indices.float() / seq_len)

            claa_rankings = approx_target_sample.get('claa_rankings', {})
            if claa_layer in claa_rankings:
                claa_indices = get_top_k_indices(torch.from_numpy(claa_rankings[claa_layer]).float(), k_absolute, seq_len)
                aggregated_indices['CLAA'].append(claa_indices.float() / seq_len)

        # --- Finalize data for plotting ---
        # Concatenate all lists of tensors into a single tensor per method
        final_aggregated_indices = {
            method: torch.cat(tensors) for method, tensors in aggregated_indices.items() if tensors
        }

        if not final_aggregated_indices:
            print(f"Warning: No data to plot for {dataset_id}")
            continue

        all_plot_data[task_name] = {
            'name': DATASET_NAME_MAP.get(dataset_id, dataset_id.title()),
            'task_category': DATASET_TO_TASK_MAP.get(dataset_id, 'Unknown'),
            'indices': final_aggregated_indices,
            'seq_len': 1.0,  # The sequence length for plotting is now 1.0
        }

    plot_density_grid(all_plot_data, args.k_percentage, output_pdf, output_png)

if __name__ == "__main__":
    main()
