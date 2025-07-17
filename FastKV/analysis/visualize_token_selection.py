import argparse
import os
import pickle
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import gaussian_kde
from tqdm import tqdm

from .viz_utils import set_publication_style, METHOD_COLORS

# ==============================================================================
# DATA LOADING AND PROCESSING
# ==============================================================================

def load_npz_data_for_dataset(base_path: str, model_name: str, dataset_name: str) -> Dict[str, Any]:
    """Loads and deserializes data from a single NPZ file for a given dataset."""
    file_path = os.path.join(base_path, model_name, f"{dataset_name}.npz")
    if not os.path.exists(file_path):
        return {}
    try:
        npz_file = np.load(file_path, allow_pickle=True)
        return {key: npz_file[key].item() for key in npz_file.files}
    except Exception as e:
        print(f"Warning: Could not load or process {file_path}. Error: {e}")
        return {}

def get_top_k_indices(scores: torch.Tensor, k: int, max_index: int) -> torch.Tensor:
    """Safely gets the indices of the top-k scores from a tensor."""
    scores = scores[:max_index]
    if scores.numel() == 0:
        return torch.tensor([], dtype=torch.long)
    
    actual_k = min(k, scores.numel())
    if actual_k < k:
        print(f"Warning: Can only select top {actual_k} tokens, not {k}. Using {actual_k}.")
        
    _, top_k_indices = torch.topk(scores, k=actual_k)
    return top_k_indices

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
            'seq_len': seq_len
        }
    return all_plot_data

# ==============================================================================
# CORE PLOTTING LOGIC
# ==============================================================================

def plot_single_density(ax: plt.Axes, all_indices: Dict[str, torch.Tensor], sequence_length: int, plot_order_map: Dict):
    """Plots the density for one sample on a given Matplotlib axis."""
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
                    print(f"Warning: KDE failed for '{key}'. Plotting a histogram as a fallback.")
                    ax.hist(indices_np, bins=50, density=True, color=props['color'], alpha=0.5, label=f"{props['label']} (hist)")

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
    fig.suptitle(f'Token Selection Density by Task Difficulty (Top {k_percentage:.0%} Kept)', fontsize=32, weight='bold')

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
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        data = all_plot_data[difficulty]
        dataset_name = data['name'].replace("_", " ").title()
        seq_len = data['seq_len']

        plot_single_density(ax, data['indices'], seq_len, plot_order_map)
        ax.set_title(f"{difficulty} Task ({dataset_name})\nSeq Len: {seq_len}", fontsize=28)
        ax.set_xlabel('Token Position in Prompt', fontsize=24)

    axes[0].set_ylabel('Density of Selected Tokens', fontsize=28)
    handles = [plt.Line2D([0], [0], color=props['color'], lw=4, label=props['label']) for props in plot_order_map.values()]
    fig.legend(
        handles, 
        [h.get_label() for h in handles],
        loc='lower center',
        bbox_to_anchor=(0.5, 0.02), # Position legend at the bottom center
        ncol=4,                   # Make it horizontal
        frameon=False             # Remove the legend box frame
    )

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
    
    DATASETS_BY_DIFFICULTY = {
        'Easy': '2wikimqa',
        'Medium': 'passage_retrieval_en',
        'Hard': 'gov_report',
    }

    all_plot_data = {}
    print("Loading and processing data for the grid plot...")
    
    for difficulty, dataset_name in tqdm(DATASETS_BY_DIFFICULTY.items(), desc="Processing datasets"):
        oracle_data = load_npz_data_for_dataset(MODELS['oracle']['base_path'], MODELS['oracle']['sanitized_name'], dataset_name)
        approx_target_data = load_npz_data_for_dataset(MODELS['approx_target']['base_path'], MODELS['approx_target']['sanitized_name'], dataset_name)
        approx_draft_data = load_npz_data_for_dataset(MODELS['approx_draft']['base_path'], MODELS['approx_draft']['sanitized_name'], dataset_name)
        
        common_keys = sorted(list(set(oracle_data.keys()) & set(approx_target_data.keys()) & set(approx_draft_data.keys())))
        if not common_keys:
            print(f"Warning: No common samples found for dataset '{dataset_name}'. Skipping this plot.")
            continue
        
        # --- MODIFICATION: Find the shortest valid sample instead of the first one ---
        shortest_sample_key = None
        min_seq_len = float('inf')

        for key in common_keys:
            # Use the oracle ranking as the ground truth for sequence length
            current_seq_len = len(oracle_data.get(key, {}).get('ranking', []))
            if 0 < current_seq_len < min_seq_len:
                min_seq_len = current_seq_len
                shortest_sample_key = key
        
        if shortest_sample_key is None:
            print(f"\nWarning: No valid common samples with sequence length > 0 found for '{dataset_name}'. Skipping.")
            continue

        target_sample_key = shortest_sample_key
        print(f"\n  -> For '{dataset_name}', selected shortest sample '{target_sample_key}' (Seq Len: {min_seq_len})")
        # --- END MODIFICATION ---

        oracle_sample = oracle_data[target_sample_key]
        approx_target_sample = approx_target_data[target_sample_key]
        approx_draft_sample = approx_draft_data[target_sample_key]
        
        for data_dict in [approx_target_sample, approx_draft_sample]:
            for rank_type in ['fastkv_rankings', 'gemfilter_rankings', 'speculative_rankings']:
                if rank_type in data_dict and isinstance(data_dict[rank_type], bytes):
                    data_dict[rank_type] = pickle.loads(data_dict[rank_type])

        fastkv_layer, gemfilter_layer = 15, 13
        spec_k_candidates = [8, 4, 1]
        
        seq_len = len(oracle_sample.get('ranking', []))
        if seq_len == 0:
            # This check is now mostly redundant due to the selection logic above, but kept for safety
            print(f"Warning: Sequence length is 0 for {dataset_name}. Skipping.")
            continue
            
        k_absolute = int(seq_len * args.k_percentage)
        all_indices = {}

        all_indices['Oracle'] = get_top_k_indices(torch.from_numpy(oracle_sample['ranking']).float(), k_absolute, seq_len)
        
        if 'fastkv_rankings' in approx_target_sample and fastkv_layer in approx_target_sample['fastkv_rankings']:
            all_indices['FastKV (Layer 15)'] = get_top_k_indices(torch.from_numpy(approx_target_sample['fastkv_rankings'][fastkv_layer]).float(), k_absolute, seq_len)
        if 'gemfilter_rankings' in approx_target_sample and gemfilter_layer in approx_target_sample['gemfilter_rankings']:
            all_indices['GemFilter (Layer 13)'] = get_top_k_indices(torch.from_numpy(approx_target_sample['gemfilter_rankings'][gemfilter_layer]).float(), k_absolute, seq_len)

        spec_rankings = approx_draft_sample.get('speculative_rankings', {})
        found_spec_k = next((k for k in spec_k_candidates if k in spec_rankings), None)
        if found_spec_k:
            scores = torch.from_numpy(spec_rankings[found_spec_k]).float()
            all_indices['Speculative Prefill'] = get_top_k_indices(scores, k_absolute, seq_len)
        
        all_plot_data[difficulty] = {
            'name': dataset_name,
            'indices': all_indices,
            'seq_len': seq_len
        }

    plot_density_grid(all_plot_data, args.k_percentage, output_pdf, output_png)

if __name__ == "__main__":
    main()
