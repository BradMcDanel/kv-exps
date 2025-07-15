# analysis/visualize_token_selection.py

import argparse
import os
import pickle
from typing import Any, Dict, List

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
    """Loads all samples from a single .npz file for a given dataset."""
    file_path = os.path.join(base_path, model_name_sanitized, f"{dataset_name}.npz")
    if not os.path.exists(file_path):
        print(f"Warning: Data file not found at: {file_path}")
        return {}

    try:
        npz_file = np.load(file_path, allow_pickle=True)
        # Unpack the 0-d array to get the dictionary inside
        return {key: npz_file[key].item() for key in npz_file.files}
    except Exception as e:
        print(f"Warning: Could not load or parse {file_path}. Error: {e}")
        return {}

def get_top_k_indices(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Returns the indices of the top-k scores."""
    if scores.numel() == 0:
        return torch.tensor([], dtype=torch.long)
    
    # Ensure k is not larger than the number of available scores
    k = min(k, len(scores))
    if k == 0:
        return torch.tensor([], dtype=torch.long)
        
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

    # Define methods and their visual properties
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
            # Use a small bandwidth to highlight sharp peaks
            try:
                kde = gaussian_kde(indices, bw_method=0.03)
                density = kde(x_grid)
                ax.plot(x_grid, density, color=props['color'], label=name, linewidth=2.5)
                ax.fill_between(x_grid, density, color=props['color'], alpha=0.2)
            except np.linalg.LinAlgError:
                print(f"Warning: Could not compute KDE for '{name}' due to singular matrix. Plotting a histogram instead.")
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
    num_tokens_to_print: int = 200
):
    """Prints the text of the selected tokens compactly for qualitative analysis."""
    print("\n" + "="*80)
    print("Qualitative Deep Dive: Text of Top-Selected Tokens (Compact)")
    print("="*80 + "\n")

    for name, indices in all_indices.items():
        header = f"--- {name} (Top {min(len(indices), num_tokens_to_print)} selections, sorted by position) ---"
        print(header)

        if not indices.numel():
            print("  (No tokens selected)\n")
            continue

        # Sort indices by position to make the output readable
        sorted_indices = torch.sort(indices).values
        tokens_to_show = sorted_indices[:num_tokens_to_print]

        # Group consecutive indices to decode them as chunks
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

        # Decode and print the chunks
        decoded_parts = []
        for group in groups_of_indices:
            group_tensor = torch.tensor(group, dtype=torch.long, device=all_tokens.device)
            token_ids = all_tokens[group_tensor]
            decoded_text = tokenizer.decode(token_ids)
            # Use repr() to escape special characters like newlines
            clean_text = repr(decoded_text).strip("'")
            decoded_parts.append(f"...'{clean_text}'...")

        # Join the parts with a separator for readability
        print("  " + " | ".join(decoded_parts) + "\n")

def main():
    parser = argparse.ArgumentParser(
        description="Generates a deep dive density plot comparing token selection positions for a single sample.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Define models and their paths, assuming 8B model for all
    MODEL_FULL_NAME = 'meta-llama/Llama-3.1-8B-Instruct'
    MODEL_SANITIZED_NAME = MODEL_FULL_NAME.replace('/', '_')
    MODELS = {
        'oracle': {
            'sanitized_name': MODEL_SANITIZED_NAME,
            'base_path': 'analysis_results/oracles'
        },
        'approx': {
            'sanitized_name': MODEL_SANITIZED_NAME,
            'base_path': 'analysis_results/approx_rankings'
        },
    }

    parser.add_argument("--tokenizer_path", type=str, default=MODEL_FULL_NAME, help="Path to the tokenizer for decoding.")
    parser.add_argument("--dataset", type=str, default='qasper', help="Dataset to analyze.")
    parser.add_argument("--sample_idx_in_file", type=int, default=0, help="The index of the sample to use from the list of common samples (0 for the first).")
    parser.add_argument("--k_percentage", type=float, default=0.1, help="Percentage for top-k analysis (e.g., 0.1 for top 10%).")
    parser.add_argument("--output_file", type=str, default="token_selection_density_deep_dive.pdf", help="Path to save the output PDF plot.")
    args = parser.parse_args()
    
    if not (0 < args.k_percentage <= 1):
        raise ValueError("--k_percentage must be between 0 and 1.")

    try:
        # Load data for the specified dataset from all sources
        oracle_data = load_npz_data_for_dataset(MODELS['oracle']['base_path'], MODELS['oracle']['sanitized_name'], args.dataset)
        approx_data = load_npz_data_for_dataset(MODELS['approx']['base_path'], MODELS['approx']['sanitized_name'], args.dataset)
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    except OSError as e:
        print(f"Error loading models or tokenizer: {e}")
        return

    # Find common sample keys (e.g., 'sample_0', 'sample_12')
    common_keys = sorted(list(set(oracle_data.keys()) & set(approx_data.keys())))
    
    if not common_keys or args.sample_idx_in_file >= len(common_keys):
        print(f"Error: Could not find a common sample at index {args.sample_idx_in_file} for dataset '{args.dataset}'.")
        print(f"Found {len(common_keys)} common samples. Available indices: 0 to {len(common_keys)-1}.")
        return

    target_sample_key = common_keys[args.sample_idx_in_file]
    print(f"\nAnalyzing common sample with key: {target_sample_key} for dataset {args.dataset}")

    # --- Data Extraction and Processing ---
    oracle_sample = oracle_data[target_sample_key]
    approx_sample = approx_data[target_sample_key]
    
    # Unpickle ranking dicts stored as bytes
    for rank_type in ['fastkv_rankings', 'gemfilter_rankings', 'speculative_rankings']:
        if isinstance(approx_sample.get(rank_type), bytes):
            approx_sample[rank_type] = pickle.loads(approx_sample[rank_type])

    # --- Define specific layers/k-values to analyze ---
    fastkv_layer, gemfilter_layer, spec_k = 15, 13, 8

    input_ids = torch.from_numpy(oracle_sample['input_ids'])
    seq_len = len(input_ids)
    k_absolute = int(seq_len * args.k_percentage)
    
    print(f"Sequence length: {seq_len}. Using top {k_absolute} tokens ({args.k_percentage:.0%}) for comparison.")
    
    all_indices = {}

    # 1. Oracle Ranking
    all_indices['Oracle (Answer-Informed)'] = get_top_k_indices(torch.from_numpy(oracle_sample['ranking']).float(), k_absolute)

    # 2. FastKV-style Ranking
    fastkv_rankings = approx_sample.get('fastkv_rankings', {})
    if fastkv_layer in fastkv_rankings:
        scores = torch.from_numpy(fastkv_rankings[fastkv_layer]).float()
        all_indices['FastKV-style (Layer 15)'] = get_top_k_indices(scores, k_absolute)
    else:
        print(f"Warning: Layer {fastkv_layer} not found in FastKV rankings. Available: {list(fastkv_rankings.keys())}")

    # 3. GemFilter-style Ranking
    gemfilter_rankings = approx_sample.get('gemfilter_rankings', {})
    if gemfilter_layer in gemfilter_rankings:
        scores = torch.from_numpy(gemfilter_rankings[gemfilter_layer]).float()
        all_indices['GemFilter-style (Layer 13)'] = get_top_k_indices(scores, k_absolute)
    else:
        print(f"Warning: Layer {gemfilter_layer} not found in GemFilter rankings. Available: {list(gemfilter_rankings.keys())}")
        
    # 4. Speculative-style Ranking
    spec_rankings = approx_sample.get('speculative_rankings', {})
    if spec_k in spec_rankings:
        scores = torch.from_numpy(spec_rankings[spec_k]).float()
        all_indices['Speculative-style (k=8)'] = get_top_k_indices(scores, k_absolute)
    else:
        print(f"Warning: Lookahead k={spec_k} not found in Speculative rankings. Available: {list(spec_rankings.keys())}")

    # --- Generate Outputs ---
    plot_selection_density(
        all_indices,
        seq_len,
        args.k_percentage,
        args.dataset,
        target_sample_key,
        args.output_file
    )
    
    print_token_text(tokenizer, all_indices, input_ids)

if __name__ == "__main__":
    main()
