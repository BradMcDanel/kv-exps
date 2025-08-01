# analysis/visualize_token_selection.py

import argparse
import os
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import gaussian_kde
from tqdm import tqdm

# Assuming these utilities are in a sibling file or an installed package.
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


def generate_dummy_data_for_grid(k_percentage: float) -> Dict[str, Any]:
    """Generates random aggregated data for layout and debugging purposes."""
    print("--- Using --debug mode: Generating random dummy data ---")
    all_plot_data = {}
    task_configs = {
        '2WikiMQA': {'id': '2wikimqa', 'task_category': 'Multi-Doc QA'},
        'GovReport': {'id': 'gov_report', 'task_category': 'Summarization'},
    }
    methods = ['Oracle', 'FastKV', 'GemFilter', 'CLAA']
    num_dummy_samples = 100

    for task_name, config in task_configs.items():
        dataset_id = config['id']
        aggregated_indices = {method: [] for method in methods}
        
        for _ in range(num_dummy_samples):
            seq_len = np.random.randint(2000, 6000)
            k_absolute = int(seq_len * k_percentage)
            
            for method in methods:
                if dataset_id == '2wikimqa':
                    means, stds = (np.array([0.1, 0.4, 0.75]), np.array([0.05, 0.08, 0.06]))
                else:
                    means, stds = (np.array([0.1, 0.9]), np.array([0.1, 0.1]))
                if method != 'Oracle':
                    means += np.random.normal(0, 0.05, size=means.shape)

                indices_list = [torch.normal(mean=m*seq_len, std=s*seq_len, size=(k_absolute//len(means),)) for m, s in zip(means, stds)]
                dummy_indices = torch.clamp(torch.cat(indices_list), 0, seq_len - 1)
                aggregated_indices[method].append(dummy_indices.float() / seq_len)

        final_indices = {m: torch.cat(aggregated_indices[m]) for m in methods}
        all_plot_data[task_name] = {'name': DATASET_NAME_MAP.get(dataset_id, dataset_id.title()), 'task_category': config['task_category'], 'indices': final_indices}
    return all_plot_data


def calculate_density_curves(aggregated_indices: Dict[str, torch.Tensor], x_grid: np.ndarray) -> Dict[str, np.ndarray]:
    """Calculates KDE density curves for all methods."""
    density_curves = {}
    for key, indices in aggregated_indices.items():
        indices_np = indices.cpu().numpy()
        if len(indices_np) > 1:
            try:
                density_curves[key] = gaussian_kde(indices_np, bw_method=0.03)(x_grid)
            except (np.linalg.LinAlgError, ValueError) as e:
                print(f"Warning: KDE failed for '{key}': {e}")
                density_curves[key] = np.zeros_like(x_grid)
        else:
            density_curves[key] = np.zeros_like(x_grid)
    return density_curves


def plot_density_difference(ax: plt.Axes, density_curves: Dict[str, np.ndarray], plot_order_map: Dict, x_grid: np.ndarray):
    """Plots the density *difference* from the Oracle on a given axis."""
    oracle_density = density_curves.get('Oracle', np.zeros_like(x_grid))
    ax.axhline(0, color='black', linestyle='-', linewidth=1.0, alpha=0.8)

    for key, props in plot_order_map.items():
        if key in density_curves:
            difference = density_curves[key] - oracle_density
            ax.plot(x_grid, difference, color=props['color'], label=props['label'], linewidth=2.5)
    
    # Highlighting has been removed for neutral presentation.
    ax.set_xlim(0, 1)
    ax.spines[['right', 'top']].set_visible(False)
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')


def plot_density_grid(all_plot_data: Dict[str, Any], k_percentage: float, output_pdf_file: str, output_png_file: str):
    """Creates the 1x2 grid plot showing aggregated token selection density difference."""
    set_publication_style()
    plt.rcParams.update({'axes.labelsize': 30, 'xtick.labelsize': 26, 'legend.fontsize': 24})

    fig, axes = plt.subplots(1, 2, figsize=(24, 8), sharey=True)
    fig.suptitle(f'Deviation from Oracle in Top-{int(100.*k_percentage)}% Token Selection', fontsize=32, weight='bold')

    plot_order_map = {
        'FastKV': {'label': 'FastKV', 'color': METHOD_COLORS['FastKV']},
        'GemFilter': {'label': 'GemFilter', 'color': METHOD_COLORS['GemFilter']},
        'CLAA': {'label': 'CLAA (Ours)', 'color': METHOD_COLORS['CLAA']},
    }
    
    x_grid = np.linspace(0, 1, 1000)
    
    task_order = ['2WikiMQA', 'GovReport']
    for i, task_name in enumerate(task_order):
        ax = axes[i]
        
        if task_name not in all_plot_data:
            ax.text(0.5, 0.5, f"Data not found", ha='center', va='center', fontsize=20)
            ax.set_title(task_name, fontsize=28)
            continue

        data = all_plot_data[task_name]
        density_curves = calculate_density_curves(data['indices'], x_grid)
        plot_density_difference(ax, density_curves, plot_order_map, x_grid)
        
        # Calculate and display MADE scores on the plot
        oracle_density = density_curves.get('Oracle', np.zeros_like(x_grid))
        task_scores = {}
        for method_key in plot_order_map.keys():
            if method_key in density_curves:
                made = np.mean(np.abs(density_curves[method_key] - oracle_density))
                task_scores[method_key] = made
        
        text_lines = ["MADE (Lower is better):"]
        sorted_scores = sorted(task_scores.items(), key=lambda item: item[1])
        for method, score in sorted_scores:
            text_lines.append(f"{plot_order_map[method]['label']}: {score:.3f}")
        
        ax.text(0.97, 0.97, "\n".join(text_lines), transform=ax.transAxes, fontsize=18,
                ha='right', va='top', bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.85))
        
        ax.set_title(f"{data['name']}\n({data['task_category']})", fontsize=28)
        ax.set_xlabel('Normalized Token Position in Prompt')

    axes[0].set_ylabel('Density Difference from Oracle')
    
    handles = [plt.Line2D([0], [0], color='gray', linestyle='--', lw=2),
               *[plt.Line2D([0], [0], color=p['color'], lw=4) for p in plot_order_map.values()]]
    labels = ['Oracle (Baseline at 0)', *[p['label'] for p in plot_order_map.values()]]
    
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=len(labels), frameon=False)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    plt.savefig(output_pdf_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved FINAL PDF to: {output_pdf_file}")
    plt.savefig(output_png_file, dpi=300, bbox_inches='tight')
    print(f"Saved FINAL PNG to: {output_png_file}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate final token selection density plots.")
    parser.add_argument("--k_percentage", type=float, default=0.1, help="Top-k percentage.")
    parser.add_argument("--debug", action="store_true", help="Use dummy data for layout testing.")
    args = parser.parse_args()

    output_dir = "figures"
    os.makedirs(output_dir, exist_ok=True)
    # Use the final, simple filename as requested
    output_pdf = os.path.join(output_dir, "token_selection.pdf")
    output_png = os.path.join(output_dir, "token_selection.png")

    if args.debug:
        all_plot_data = generate_dummy_data_for_grid(args.k_percentage)
        plot_density_grid(all_plot_data, args.k_percentage, output_pdf, output_png)
        return

    TARGET_MODEL = 'meta-llama/Llama-3.1-8B-Instruct'
    MODELS = {
        'oracle': {'sanitized_name': TARGET_MODEL.replace('/', '_'), 'base_path': 'analysis_results/oracles'},
        'approx_target': {'sanitized_name': TARGET_MODEL.replace('/', '_'), 'base_path': 'analysis_results/approx_rankings'},
    }
    
    DATASETS_BY_TASK = {'2WikiMQA': '2wikimqa', 'GovReport': 'gov_report'}
    all_plot_data = {}
    fastkv_layer, gemfilter_layer, claa_layer = 15, 15, 15
    
    for task_name, dataset_id in tqdm(DATASETS_BY_TASK.items(), desc="Processing datasets"):
        oracle_data = load_npz_data_for_dataset(MODELS['oracle']['base_path'], MODELS['oracle']['sanitized_name'], dataset_id)
        approx_target_data = load_npz_data_for_dataset(MODELS['approx_target']['base_path'], MODELS['approx_target']['sanitized_name'], dataset_id)
        
        common_keys = sorted(list(set(oracle_data.keys()) & set(approx_target_data.keys())))
        if not common_keys: continue
        
        aggregated_indices: Dict[str, List[torch.Tensor]] = {m: [] for m in ['Oracle', 'FastKV', 'GemFilter', 'CLAA']}

        for key in tqdm(common_keys, desc=f"Aggregating {dataset_id}", leave=False):
            oracle_sample = oracle_data.get(key, {})
            approx_target_sample = approx_target_data.get(key, {})
            deserialize_rankings_in_sample(approx_target_sample)

            if 'ranking' not in oracle_sample or len(oracle_sample['ranking']) == 0: continue
            
            seq_len = len(oracle_sample['ranking'])
            k_absolute = int(seq_len * args.k_percentage)
            
            def get_norm_indices(ranking, k, slen):
                return get_top_k_indices(torch.from_numpy(ranking).float(), k, slen).float() / slen

            aggregated_indices['Oracle'].append(get_norm_indices(oracle_sample['ranking'], k_absolute, seq_len))

            for method_name, rankings_key, layer in [('FastKV', 'fastkv_rankings', fastkv_layer), 
                                                     ('GemFilter', 'gemfilter_rankings', gemfilter_layer), 
                                                     ('CLAA', 'claa_rankings', claa_layer)]:
                rankings = approx_target_sample.get(rankings_key, {})
                if layer in rankings:
                    aggregated_indices[method_name].append(get_norm_indices(rankings[layer], k_absolute, seq_len))

        final_aggregated_indices = {m: torch.cat(t) for m, t in aggregated_indices.items() if t}
        if not final_aggregated_indices: continue

        all_plot_data[task_name] = {
            'name': DATASET_NAME_MAP.get(dataset_id, dataset_id.title()),
            'task_category': DATASET_TO_TASK_MAP.get(dataset_id, 'Unknown'),
            'indices': final_aggregated_indices,
        }

    plot_density_grid(all_plot_data, args.k_percentage, output_pdf, output_png)

if __name__ == "__main__":
    main()
