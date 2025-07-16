# analysis/visualize_token_selection.py

import argparse
import os
import pickle
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from scipy.stats import gaussian_kde
from transformers import AutoTokenizer

# --- Plotting Style ---
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 14,
})

def load_npz_data_for_dataset(base_path: str, model_name_sanitized: str, dataset_name: str) -> Dict[str, Any]:
    """Loads all samples from a single .npz file for a given dataset. Exits on failure."""
    file_path = os.path.join(base_path, model_name_sanitized, f"{dataset_name}.npz")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"FATAL: Data file not found at: {file_path}")

    try:
        npz_file = np.load(file_path, allow_pickle=True)
        return {key: npz_file[key].item() for key in npz_file.files}
    except Exception as e:
        raise IOError(f"FATAL: Could not load or parse {file_path}. Error: {e}")

def get_top_k_indices(scores: torch.Tensor, k: int, max_index: Optional[int] = None) -> torch.Tensor:
    """
    Returns the indices of the top-k scores, strictly from within the valid index range.
    This ensures we select from a pool of valid indices, making comparisons fair.
    """
    if max_index is not None:
        if len(scores) < max_index:
            raise ValueError(f"Score tensor length ({len(scores)}) is less than max_index ({max_index}). This should not happen for valid methods.")
        # First, limit the scores to the valid range BEFORE finding the top-k.
        scores = scores[:max_index]

    if scores.numel() < k:
        raise ValueError(f"FATAL: Cannot select top {k} tokens because only {scores.numel()} valid scores are available.")
        
    _, top_k_indices = torch.topk(scores, k=k)
    return top_k_indices

def plot_selection_density(
    all_indices: Dict[str, torch.Tensor],
    sequence_length: int,
    k_percentage: float,
    dataset_name: str,
    sample_key: str,
    output_file: str
):
    """Creates a density plot showing the concentration of selected tokens."""
    fig, ax = plt.subplots(figsize=(20, 6))
    methods = {
        'Oracle (Answer-Informed)': {'color': '#2ca02c'},
        'FastKV-style (Layer 15)': {'color': '#d62728'},
        'GemFilter-style (Layer 13)': {'color': '#ff7f0e'},
        'Speculative-style (k=8)': {'color': '#1f77b4'},
    }
    x_grid = np.linspace(0, sequence_length, 1000)
    for name, props in methods.items():
        if name in all_indices and len(all_indices[name]) > 1:
            indices = all_indices[name].cpu().numpy()
            try:
                kde = gaussian_kde(indices, bw_method=0.03)
                density = kde(x_grid)
                ax.plot(x_grid, density, color=props['color'], label=name, linewidth=2.5)
                ax.fill_between(x_grid, density, color=props['color'], alpha=0.2)
            except np.linalg.LinAlgError:
                ax.hist(indices, bins=min(100, sequence_length//10), density=True, color=props['color'], alpha=0.5, label=f"{name} (hist)")

    ax.set_xlabel('Token Position in Prompt')
    ax.set_ylabel('Density of Selected Tokens')
    ax.grid(axis='y', linestyle=':', linewidth=0.7)
    ax.spines[['right', 'top']].set_visible(False)
    ax.legend(title="Ranking Method")
    ax.set_xlim(0, sequence_length)
    ax.set_yticklabels([]) 
    ax.tick_params(axis='y', length=0)
    title = f'Density of Top-{k_percentage:.0%} Selected Tokens\n'
    title += f'Dataset: {dataset_name.replace("_", " ").title()} (Sequence Length: {sequence_length})'
    ax.set_title(title, pad=20)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved to: {output_file}")

def print_token_text(
    tokenizer: AutoTokenizer, 
    all_indices: Dict[str, torch.Tensor],
    all_tokens: torch.Tensor,
    num_tokens_to_print: int = 300
):
    """Prints the text of the selected tokens compactly for qualitative analysis."""
    print("\n" + "="*80)
    print("Qualitative Deep Dive: Text of Top-Selected Tokens (Compact)")
    print("="*80 + "\n")

    for name, indices in all_indices.items():
        # Updated header to show total selected tokens vs. printed tokens
        header = f"--- {name} (Showing top {min(len(indices), num_tokens_to_print)} of {len(indices)} selections, sorted by position) ---"
        print(header)

        sorted_indices = torch.sort(indices).values
        tokens_to_show = sorted_indices[:num_tokens_to_print]
        groups_of_indices: List[List[int]] = []
        if tokens_to_show.numel() > 0:
            current_group = [tokens_to_show[0].item()]
            for i in range(1, len(tokens_to_show)):
                if tokens_to_show[i] == tokens_to_show[i-1] + 1:
                    current_group.append(tokens_to_show[i].item())
                else:
                    groups_of_indices.append(current_group)
                    current_group = [tokens_to_show[i].item()]
            groups_of_indices.append(current_group)

        decoded_parts = []
        for group in groups_of_indices:
            group_tensor = torch.tensor(group, dtype=torch.long, device=all_tokens.device)
            token_ids = all_tokens[group_tensor]
            decoded_text = tokenizer.decode(token_ids)
            clean_text = repr(decoded_text).strip("'")
            decoded_parts.append(f"...'{clean_text}'...")

        print("  " + " | ".join(decoded_parts) + "\n")

def main():
    parser = argparse.ArgumentParser(
        description="Generates a deep dive density plot comparing token selection positions for a single sample.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # --- Model Configuration ---
    TARGET_MODEL = 'meta-llama/Llama-3.1-8B-Instruct'
    DRAFT_MODEL = 'meta-llama/Llama-3.2-1B-Instruct'

    MODELS = {
        'oracle': {'sanitized_name': TARGET_MODEL.replace('/', '_'), 'base_path': 'analysis_results/oracles'},
        'approx_target': {'sanitized_name': TARGET_MODEL.replace('/', '_'), 'base_path': 'analysis_results/approx_rankings'},
        'approx_draft': {'sanitized_name': DRAFT_MODEL.replace('/', '_'), 'base_path': 'analysis_results/approx_rankings'},
    }
    
    parser.add_argument("--tokenizer_path", type=str, default=TARGET_MODEL, help="Path to the tokenizer for decoding.")
    parser.add_argument("--dataset", type=str, default='qasper', help="Dataset to analyze.")
    parser.add_argument("--sample_idx_in_file", type=int, default=0, help="The index of the sample to use from the list of common samples.")
    parser.add_argument("--k_percentage", type=float, default=0.1, help="Percentage for top-k analysis (e.g., 0.1 for top 10%).")
    parser.add_argument("--output_file", type=str, default="token_selection_density_deep_dive.pdf", help="Path to save the output PDF plot.")
    args = parser.parse_args()
    
    if not (0 < args.k_percentage <= 1):
        raise ValueError("--k_percentage must be between 0 and 1.")

    # --- Strict Data Loading ---
    print("Loading data with strict, fail-fast settings...")
    oracle_data = load_npz_data_for_dataset(MODELS['oracle']['base_path'], MODELS['oracle']['sanitized_name'], args.dataset)
    # FastKV/GemFilter come from the TARGET (8B) model's approx results
    approx_target_data = load_npz_data_for_dataset(MODELS['approx_target']['base_path'], MODELS['approx_target']['sanitized_name'], args.dataset)
    # Speculative comes from the DRAFT (1B) model's approx results
    approx_draft_data = load_npz_data_for_dataset(MODELS['approx_draft']['base_path'], MODELS['approx_draft']['sanitized_name'], args.dataset)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    # Find common sample keys across all three necessary files
    common_keys = sorted(list(set(oracle_data.keys()) & set(approx_target_data.keys()) & set(approx_draft_data.keys())))
    if not common_keys or args.sample_idx_in_file >= len(common_keys):
        raise IndexError(f"FATAL: Could not find a common sample at index {args.sample_idx_in_file} for dataset '{args.dataset}'.")

    target_sample_key = common_keys[args.sample_idx_in_file]
    print(f"\nAnalyzing common sample with key: '{target_sample_key}' for dataset {args.dataset}")

    # --- Data Extraction and Processing ---
    oracle_sample = oracle_data[target_sample_key]
    approx_target_sample = approx_target_data[target_sample_key]
    approx_draft_sample = approx_draft_data[target_sample_key]
    
    # Unpickle relevant ranking dicts
    for data_dict in [approx_target_sample, approx_draft_sample]:
        for rank_type in ['fastkv_rankings', 'gemfilter_rankings', 'speculative_rankings']:
            if rank_type in data_dict and isinstance(data_dict[rank_type], bytes):
                data_dict[rank_type] = pickle.loads(data_dict[rank_type])

    # --- Define specific layers/k-values to analyze ---
    fastkv_layer, gemfilter_layer, spec_k = 15, 13, 8

    input_ids = torch.from_numpy(oracle_sample['input_ids'])
    seq_len = len(input_ids)
    k_absolute = int(seq_len * args.k_percentage)
    
    print(f"Sequence length: {seq_len}. Strictly selecting top {k_absolute} tokens ({args.k_percentage:.0%}) for comparison.")
    
    all_indices = {}

    # --- Get Top-K Indices (Strict) ---
    # 1. Oracle Ranking (from TARGET model)
    all_indices['Oracle (Answer-Informed)'] = get_top_k_indices(torch.from_numpy(oracle_sample['ranking']).float(), k_absolute, max_index=seq_len)

    # 2. FastKV-style Ranking (from TARGET model)
    fastkv_rankings = approx_target_sample.get('fastkv_rankings', {})
    if fastkv_layer not in fastkv_rankings: raise KeyError(f"FATAL: Layer {fastkv_layer} not found in FastKV rankings for TARGET model.")
    scores = torch.from_numpy(fastkv_rankings[fastkv_layer]).float()
    all_indices['FastKV-style (Layer 15)'] = get_top_k_indices(scores, k_absolute, max_index=seq_len)
    
    # 3. GemFilter-style Ranking (from TARGET model)
    gemfilter_rankings = approx_target_sample.get('gemfilter_rankings', {})
    if gemfilter_layer not in gemfilter_rankings: raise KeyError(f"FATAL: Layer {gemfilter_layer} not found in GemFilter rankings for TARGET model.")
    scores = torch.from_numpy(gemfilter_rankings[gemfilter_layer]).float()
    all_indices['GemFilter-style (Layer 13)'] = get_top_k_indices(scores, k_absolute, max_index=seq_len)
        
    # 4. Speculative-style Ranking (from DRAFT model)
    spec_rankings = approx_draft_sample.get('speculative_rankings', {})
    if spec_k not in spec_rankings: raise KeyError(f"FATAL: Lookahead k={spec_k} not found in Speculative rankings for DRAFT model.")
    scores = torch.from_numpy(spec_rankings[spec_k]).float()
    all_indices['Speculative-style (k=8)'] = get_top_k_indices(scores, k_absolute, max_index=seq_len)
    
    # --- Generate Outputs ---
    plot_selection_density(all_indices, seq_len, args.k_percentage, args.dataset, target_sample_key, args.output_file)
    print_token_text(tokenizer, all_indices, input_ids)

if __name__ == "__main__":
    main()
