import os
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from transformers import AutoTokenizer
from scipy.stats import gaussian_kde
from typing import Dict, Any, List

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
        return {key: npz_file[key].item() for key in npz_file.files}
    except Exception as e:
        print(f"Warning: Could not load or parse {file_path}. Error: {e}")
        return {}

def get_top_k_indices(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Returns the indices of the top-k scores."""
    if scores.numel() == 0:
        return torch.tensor([], dtype=torch.long)
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

    methods = {
        'Oracle (8B Answer-Informed)': {'color': '#2ca02c'},
        'FastKV-style (8B Layer 15)': {'color': '#ff7f0e'},
        'Speculative Prefill-style (1B Layer 15 Cumulative)': {'color': '#1f77b4'},
    }
    
    x_grid = np.linspace(0, sequence_length, 1000)
    for name, props in methods.items():
        if name in all_indices and len(all_indices[name]) > 1:
            indices = all_indices[name].cpu().numpy()
            kde = gaussian_kde(indices, bw_method=0.03)
            density = kde(x_grid)
            ax.plot(x_grid, density, color=props['color'], label=name, linewidth=2.5)
            ax.fill_between(x_grid, density, color=props['color'], alpha=0.2)
    
    ax.set_xlabel('Token Position in Prompt')
    ax.set_ylabel('Density of Selected Tokens')
    ax.grid(axis='y', linestyle=':', linewidth=0.7)
    ax.spines[['right', 'top']].set_visible(False)
    ax.legend()
    ax.set_xlim(0, sequence_length)
    ax.set_yticklabels([]) 
    ax.tick_params(axis='y', length=0)

    title = f'Deep Dive: Density of Top-{k_percentage:.0%} Selected Tokens\n'
    title += f'Dataset: {dataset_name.replace("_", " ").title()} | Sample Key: {sample_key} | Sequence Length: {sequence_length}'
    ax.set_title(title, pad=20)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\nDeep dive visualization saved to: {output_file}")

def print_token_text(
    tokenizer : AutoTokenizer, 
    all_indices: Dict[str, torch.Tensor],
    all_tokens: torch.Tensor,
    num_tokens_to_print: int = 200
):
    """Prints the text of the selected tokens compactly for qualitative analysis."""
    print("\n" + "="*80)
    print("Qualitative Deep Dive: Text of Top-Selected Tokens (Compact)")
    print("="*80 + "\n")

    for name, indices in all_indices.items():
        header = f"--- {name} (Top {min(len(indices), num_tokens_to_print)} selections) ---"
        print(header)

        if not indices.numel():
            print("  (No tokens selected)\n")
            continue

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
            decoded_parts.append(f"{clean_text}")

        print("  " + " | ".join(decoded_parts) + "\n")

def main():
    parser = argparse.ArgumentParser(
        description="Generates a deep dive density plot comparing token selection positions for a single sample.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Define models and their paths
    MODELS = {
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

    parser.add_argument("--tokenizer_path", type=str, default='meta-llama/Llama-3.1-8B-Instruct', help="Path to the tokenizer for decoding.")
    parser.add_argument("--dataset", type=str, default='qasper', help="Dataset to analyze.")
    parser.add_argument("--sample_idx_in_file", type=int, default=0, help="The index of the sample to use from the list of common samples (0 for the first).")
    parser.add_argument("--k_percentage", type=float, default=0.1, help="Percentage for top-k analysis (e.g., 0.1 for top 10%).")
    parser.add_argument("--output_file", type=str, default="token_selection_density_deep_dive.pdf", help="Path to save the output PDF plot.")
    args = parser.parse_args()
    
    try:
        # Load data for the specified dataset from all models
        oracle_data = load_npz_data_for_dataset(MODELS['oracle']['base_path'], MODELS['oracle']['sanitized_name'], args.dataset)
        approx_8b_data = load_npz_data_for_dataset(MODELS['8B']['base_path'], MODELS['8B']['sanitized_name'], args.dataset)
        approx_1b_data = load_npz_data_for_dataset(MODELS['1B']['base_path'], MODELS['1B']['sanitized_name'], args.dataset)
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    except OSError as e:
        print(f"Error loading models or tokenizer: {e}")
        return

    # Find common sample keys (e.g., 'sample_0', 'sample_12')
    common_keys = sorted(list(
        set(oracle_data.keys()) & set(approx_8b_data.keys()) & set(approx_1b_data.keys())
    ))
    
    if not common_keys or args.sample_idx_in_file >= len(common_keys):
        print(f"Error: Could not find a common sample at index {args.sample_idx_in_file} for dataset '{args.dataset}'.")
        print(f"Found {len(common_keys)} common samples.")
        return

    target_sample_key = common_keys[args.sample_idx_in_file]
    print(f"\nAnalyzing common sample with key: {target_sample_key}")

    # Retrieve the specific sample data using the key
    oracle_sample = oracle_data[target_sample_key]
    approx_8b_sample = approx_8b_data[target_sample_key]
    approx_1b_sample = approx_1b_data[target_sample_key]
    
    # Unpickle ranking dicts if necessary
    if isinstance(approx_8b_sample['layerwise_rankings'], bytes):
        approx_8b_sample['layerwise_rankings'] = pickle.loads(approx_8b_sample['layerwise_rankings'])
    if isinstance(approx_1b_sample['cumulative_rankings'], bytes):
        approx_1b_sample['cumulative_rankings'] = pickle.loads(approx_1b_sample['cumulative_rankings'])

    layer_to_use = 15

    input_ids = torch.from_numpy(oracle_sample['input_ids'])
    k = int(len(input_ids) * args.k_percentage)
    all_indices = {}

    # 1. Oracle
    all_indices['Oracle (8B Answer-Informed)'] = get_top_k_indices(torch.from_numpy(oracle_sample['ranking']), k)

    # 2. FastKV-style (8B Single Layer)
    layerwise_rankings_8b = approx_8b_sample.get('layerwise_rankings', {})
    if layer_to_use in layerwise_rankings_8b:
        fastkv_ranking = torch.from_numpy(layerwise_rankings_8b[layer_to_use])
        all_indices['FastKV-style (8B Layer 15)'] = get_top_k_indices(fastkv_ranking, k)
    else:
        print(f"Warning: Layer {layer_to_use} not found in 8B layerwise rankings. Available: {list(layerwise_rankings_8b.keys())}")

    # 3. Speculative Prefill-style (1B Cumulative)
    cumulative_rankings_1b = approx_1b_sample.get('cumulative_rankings', {})
    if layer_to_use in cumulative_rankings_1b:
        spec_prefill_ranking = torch.from_numpy(cumulative_rankings_1b[layer_to_use])
        all_indices['Speculative Prefill-style (1B Layer 15 Cumulative)'] = get_top_k_indices(spec_prefill_ranking, k)
    else:
        print(f"Warning: Layer {layer_to_use} not found in 1B cumulative rankings. Available: {list(cumulative_rankings_1b.keys())}")

    plot_selection_density(
        all_indices,
        len(input_ids),
        args.k_percentage,
        args.dataset,
        target_sample_key,
        args.output_file
    )
    
    print_token_text(tokenizer, all_indices, input_ids)

if __name__ == "__main__":
    main()
