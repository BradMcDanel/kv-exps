# analysis/visualize_token_selection.py

import argparse
import os
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import gaussian_kde
from tqdm import tqdm

from .viz_utils import set_publication_style, METHOD_COLORS
from .retrieval_metrics import (
    load_npz_data_for_dataset,
    get_top_k_indices,
    deserialize_rankings_in_sample,
    calculate_oracle_overlap,
)

# ==============================================================================
# DATA LOADING AND PROCESSING
# ==============================================================================

def generate_dummy_data_for_grid(k_percentage: float) -> Dict[str, Any]:
    """Generates random data for layout and debugging purposes."""
    print("--- Using --debug mode: Generating random dummy data ---")
    all_plot_data = {}
    difficulty_configs = {
        'Easy': {'name': 'Dummy Easy', 'seq_len': 2048},
        'Medium': {'name': 'Dummy Medium', 'seq_len': 8192},
        'Hard': {'name': 'Dummy Hard', 'seq_len': 4096},
    }
    methods = ['Oracle', 'FastKV (Layer 15)', 'GemFilter (Layer 13)', 'Speculative Prefill']

    for difficulty, config in difficulty_configs.items():
        seq_len = config['seq_len']
        k_absolute = int(seq_len * k_percentage)
        indices = {}
        for method in methods:
            if difficulty == 'Easy':
                mean = seq_len * (0.7 if method == 'Oracle' else np.random.uniform(0.6, 0.8))
                std = seq_len * (0.05 if method == 'Oracle' else 0.1)
            elif difficulty == 'Medium':
                mean = seq_len * (0.5 if method == 'Oracle' else np.random.uniform(0.4, 0.6))
                std = seq_len * (0.1 if method == 'Oracle' else 0.15)
            else: # Hard
                mean = seq_len * 0.5
                std = seq_len * 0.3
            
            dummy_indices = torch.normal(mean=mean, std=std, size=(k_absolute,)).long()
            indices[method] = torch.clamp(dummy_indices, 0, seq_len - 1)
        
        all_plot_data[difficulty] = {
            'name': config['name'],
            'indices': indices,
            'seq_len': seq_len,
            'accuracies': {
                'FastKV': np.random.uniform(0.7, 0.9),
                'GemFilter': np.random.uniform(0.65, 0.85),
                'Speculative Prefill': np.random.uniform(0.6, 0.8)
            }
        }
    return all_plot_data

# ==============================================================================
# CORE PLOTTING LOGIC
# ==============================================================================

def plot_single_density(
    ax: plt.Axes,
    all_indices: Dict[str, torch.Tensor],
    sequence_length: int,
    plot_order_map: Dict,
    accuracies: Dict[str, float]
):
    """Plots the density for one sample and its accuracies on a given axis."""
    x_grid = np.linspace(0, sequence_length, 1000)

    for key, props in plot_order_map.items():
        if key in all_indices:
            indices_np = all_indices[key].cpu().numpy()
            if len(indices_np) > 1:
                try:
                    kde = gaussian_kde(indices_np, bw_method=0.03)
                    density = kde(x_grid)
                    ax.plot(x_grid, density, color=props['color'], label=props['label'])
                    ax.fill_between(x_grid, density, color=props['color'], alpha=0.2)
                except (np.linalg.LinAlgError, ValueError):
                    print(f"Warning: KDE failed for '{key}'. Plotting a histogram.")
                    ax.hist(indices_np, bins=50, density=True, color=props['color'], alpha=0.5, label=f"{props['label']} (hist)")

    # Add the accuracy text box
    if accuracies:
        text_lines = ["Oracle Overlap (Top-10%):"]
        display_map = {'FastKV': 'FastKV', 'GemFilter': 'GemFilter', 'Speculative Prefill': 'Spec. Prefill'}
        for key, label in display_map.items():
            if key in accuracies:
                text_lines.append(f"{label}: {accuracies[key]:.1%}")
        text_string = "\n".join(text_lines)
        
        ax.text(0.97, 0.97, text_string, transform=ax.transAxes, fontsize=14,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.7))

    ax.set_xlim(0, sequence_length)
    ax.set_yticklabels([])
    ax.tick_params(axis='y', length=0)
    ax.spines[['right', 'top', 'left']].set_visible(False)

def plot_density_grid(
    all_plot_data: Dict[str, Any],
    k_percentage: float,
    output_pdf_file: str,
    output_png_file: str
):
    """Creates the 1x3 grid plot showing token selection density by task difficulty."""
    set_publication_style()
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    fig.suptitle(f'Positional Distribution of Top-{int(100.*k_percentage)}% Ranked Tokens Varies with Task Difficulty', fontsize=32, weight='bold')

    plot_order_map = {
        'Oracle': {'label': 'Oracle', 'color': METHOD_COLORS['Oracle']},
        'FastKV (Layer 15)': {'label': 'FastKV (Layer 15)', 'color': METHOD_COLORS['FastKV']},
        'GemFilter (Layer 13)': {'label': 'GemFilter (Layer 13)', 'color': METHOD_COLORS['GemFilter']},
        'Speculative Prefill': {'label': 'Speculative Prefill', 'color': METHOD_COLORS['Speculative']},
    }

    difficulty_order = ['Easy', 'Medium', 'Hard']
    for i, difficulty in enumerate(difficulty_order):
        ax = axes[i]
        if difficulty not in all_plot_data:
            ax.text(0.5, 0.5, f"Data not found for\n{difficulty} task", ha='center', va='center', style='italic', fontsize=20)
            ax.set_title(f"{difficulty} Task", fontsize=28)
            ax.set_xticks([]); ax.set_yticks([])
            continue

        data = all_plot_data[difficulty]
        dataset_name = data['name'].replace("_", " ").title()
        seq_len = data['seq_len']
        accuracies = data.get('accuracies', {})

        plot_single_density(ax, data['indices'], seq_len, plot_order_map, accuracies)
        ax.set_title(f"{difficulty} Task ({dataset_name})", fontsize=28)
        ax.set_xlabel('Token Position in Prompt', fontsize=24)

    axes[0].set_ylabel('Density of Selected Tokens', fontsize=28)
    handles = [plt.Line2D([0], [0], color=props['color'], lw=4, label=props['label']) for props in plot_order_map.values()]
    fig.legend(handles, [h.get_label() for h in handles], loc='lower center',
               bbox_to_anchor=(0.5, 0.02), ncol=4, frameon=False)

    plt.tight_layout(rect=[0, 0.1, 1, 0.99])
    
    plt.savefig(output_pdf_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved PDF to: {output_pdf_file}")
    plt.savefig(output_png_file, dpi=300, bbox_inches='tight')
    print(f"Saved PNG to: {output_png_file}")
    plt.close(fig)

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate a grid of token selection density plots by task difficulty.")
    parser.add_argument("--k_percentage", type=float, default=0.1, help="Top-k percentage for token selection.")
    parser.add_argument("--debug", action="store_true", help="Use dummy data for fast layout iteration.")
    args = parser.parse_args()

    output_dir = "figures"
    os.makedirs(output_dir, exist_ok=True)
    output_pdf = os.path.join(output_dir, "token_selection.pdf")
    output_png = os.path.join(output_dir, "token_selection.png")

    if args.debug:
        all_plot_data = generate_dummy_data_for_grid(args.k_percentage)
        plot_density_grid(all_plot_data, args.k_percentage, output_pdf, output_png)
        return

    # --- Real Data Loading and Processing ---
    TARGET_MODEL = 'meta-llama/Llama-3.1-8B-Instruct'
    DRAFT_MODEL = 'meta-llama/Llama-3.2-1B-Instruct'
    MODELS = {
        'oracle': {'sanitized_name': TARGET_MODEL.replace('/', '_'), 'base_path': 'analysis_results/oracles'},
        'approx_target': {'sanitized_name': TARGET_MODEL.replace('/', '_'), 'base_path': 'analysis_results/approx_rankings'},
        'approx_draft': {'sanitized_name': DRAFT_MODEL.replace('/', '_'), 'base_path': 'analysis_results/approx_rankings'},
    }
    
    DATASETS_BY_DIFFICULTY = {'Easy': '2wikimqa', 'Medium': 'gov_report', 'Hard': 'trec'}
    all_plot_data = {}
    print("Loading and processing data for the grid plot...")
    
    for difficulty, dataset_name in tqdm(DATASETS_BY_DIFFICULTY.items(), desc="Processing datasets"):
        oracle_data = load_npz_data_for_dataset(MODELS['oracle']['base_path'], MODELS['oracle']['sanitized_name'], dataset_name)
        approx_target_data = load_npz_data_for_dataset(MODELS['approx_target']['base_path'], MODELS['approx_target']['sanitized_name'], dataset_name)
        approx_draft_data = load_npz_data_for_dataset(MODELS['approx_draft']['base_path'], MODELS['approx_draft']['sanitized_name'], dataset_name)
        
        common_keys = sorted(list(set(oracle_data.keys()) & set(approx_target_data.keys()) & set(approx_draft_data.keys())))
        if not common_keys: continue
        
        TARGET_TOKEN_COUNT = 2048
        best_sample_key, best_seq_len, min_distance = None, -1, float('inf')
        for key in common_keys:
            current_seq_len = len(oracle_data.get(key, {}).get('ranking', []))
            if current_seq_len > 0:
                distance = abs(current_seq_len - TARGET_TOKEN_COUNT)
                if distance < min_distance:
                    min_distance, best_sample_key, best_seq_len = distance, key, current_seq_len
        
        if best_sample_key is None: continue
        target_sample_key = best_sample_key
        print(f"\n  -> For '{dataset_name}', selected sample '{target_sample_key}' (Seq Len: {best_seq_len})")

        oracle_sample = oracle_data[target_sample_key]
        approx_target_sample = approx_target_data.get(target_sample_key, {})
        approx_draft_sample = approx_draft_data.get(target_sample_key, {})
        
        deserialize_rankings_in_sample(approx_target_sample)
        deserialize_rankings_in_sample(approx_draft_sample)

        fastkv_layer, gemfilter_layer, spec_k_candidates = 15, 13, [8, 4, 1]
        seq_len = best_seq_len
        k_absolute = int(seq_len * args.k_percentage)
        all_indices, accuracies_for_plot = {}, {}
        
        if 'ranking' not in oracle_sample: continue
        oracle_ranking_tensor = torch.from_numpy(oracle_sample['ranking']).float()
        all_indices['Oracle'] = get_top_k_indices(oracle_ranking_tensor, k_absolute, seq_len)
        
        # Calculate and store indices and accuracies
        fk_rankings = approx_target_sample.get('fastkv_rankings', {})
        if fastkv_layer in fk_rankings:
            all_indices['FastKV (Layer 15)'] = get_top_k_indices(torch.from_numpy(fk_rankings[fastkv_layer]).float(), k_absolute, seq_len)
            fk_accs = calculate_oracle_overlap(fk_rankings, oracle_ranking_tensor, 0.1)
            if fastkv_layer in fk_accs: accuracies_for_plot['FastKV'] = fk_accs[fastkv_layer]

        gf_rankings = approx_target_sample.get('gemfilter_rankings', {})
        if gemfilter_layer in gf_rankings:
            all_indices['GemFilter (Layer 13)'] = get_top_k_indices(torch.from_numpy(gf_rankings[gemfilter_layer]).float(), k_absolute, seq_len)
            gf_accs = calculate_oracle_overlap(gf_rankings, oracle_ranking_tensor, 0.1)
            if gemfilter_layer in gf_accs: accuracies_for_plot['GemFilter'] = gf_accs[gemfilter_layer]

        spec_rankings = approx_draft_sample.get('speculative_rankings', {})
        found_spec_k = next((k for k in spec_k_candidates if k in spec_rankings), None)
        if found_spec_k:
            all_indices['Speculative Prefill'] = get_top_k_indices(torch.from_numpy(spec_rankings[found_spec_k]).float(), k_absolute, seq_len)
            sp_accs = calculate_oracle_overlap(spec_rankings, oracle_ranking_tensor, 0.1)
            if found_spec_k in sp_accs: accuracies_for_plot['Speculative Prefill'] = sp_accs[found_spec_k]
        
        all_plot_data[difficulty] = {
            'name': dataset_name,
            'indices': all_indices,
            'seq_len': seq_len,
            'accuracies': accuracies_for_plot,
        }

    plot_density_grid(all_plot_data, args.k_percentage, output_pdf, output_png)

if __name__ == "__main__":
    main()
