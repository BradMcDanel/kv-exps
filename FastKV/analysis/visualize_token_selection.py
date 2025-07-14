import os
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from transformers import AutoTokenizer
from scipy.stats import gaussian_kde
from typing import Dict

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

def load_data(path: str) -> Dict:
    """Safely loads a pickle file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found at: {path}")
    print(f"Loading data from: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)

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
    sample_idx: int,
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
            # bw_method controls the "smoothness" of the density curve
            kde = gaussian_kde(indices, bw_method=0.03)
            density = kde(x_grid)
            ax.plot(x_grid, density, color=props['color'], label=name, linewidth=2.5)
            ax.fill_between(x_grid, density, color=props['color'], alpha=0.2)
    
    # --- Formatting the plot ---
    ax.set_xlabel('Token Position in Prompt')
    ax.set_ylabel('Density of Selected Tokens')
    ax.grid(axis='y', linestyle=':', linewidth=0.7)
    ax.spines[['right', 'top']].set_visible(False)
    ax.legend()
    ax.set_xlim(0, sequence_length)
    ax.set_yticklabels([]) # Hide y-axis numerical labels as density values are relative
    ax.tick_params(axis='y', length=0) # remove y-axis ticks

    title = f'Deep Dive: Density of Top-{k_percentage:.0%} Selected Tokens\n'
    title += f'Dataset: {dataset_name.replace("_", " ").title()} | Sample Index: {sample_idx} | Sequence Length: {sequence_length}'
    ax.set_title(title, pad=20)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\nDeep dive visualization saved to: {output_file}")


def print_token_text(
    tokenizer: AutoTokenizer,
    all_indices: Dict[str, torch.Tensor],
    all_tokens: torch.Tensor,
    num_tokens_to_print: int = 20
):
    """Prints the text of the selected tokens for qualitative analysis."""
    print("\n" + "="*80)
    print("Qualitative Deep Dive: Text of Top-Selected Tokens")
    print("="*80 + "\n")
    
    for name, indices in all_indices.items():
        print(f"--- {name} (Top {min(len(indices), num_tokens_to_print)} selections, sorted by position) ---")
        
        sorted_indices = torch.sort(indices).values
        tokens_to_show = sorted_indices[:num_tokens_to_print]
        token_ids = all_tokens[tokens_to_show]
        decoded_tokens = [tokenizer.decode([tid]) for tid in token_ids]
        
        formatted_output = []
        for i, token_text in zip(tokens_to_show, decoded_tokens):
            clean_text = repr(token_text)[1:-1]
            formatted_output.append(f"  Pos {i:<4}: '{clean_text}'")
        
        print("\n".join(formatted_output))
        print("\n" + "-"*40 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generates a deep dive density plot comparing token selection positions for a single sample.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    default_oracle_path = 'analysis_results/oracles/oracles_model_meta-llama_Llama-3.1-8B-Instruct.pkl'
    default_approx_8b_path = 'analysis_results/approx_rankings/approx_rankings_model_meta-llama_Llama-3.1-8B-Instruct.pkl'
    default_approx_1b_path = 'analysis_results/approx_rankings/approx_rankings_model_meta-llama_Llama-3.2-1B-Instruct.pkl'

    parser.add_argument("--oracle_file", type=str, default=default_oracle_path, help="Path to the 8B oracle rankings file.")
    parser.add_argument("--approx_8b_file", type=str, default=default_approx_8b_path, help="Path to the 8B approximate rankings file.")
    parser.add_argument("--approx_1b_file", type=str, default=default_approx_1b_path, help="Path to the 1B approximate rankings file.")
    parser.add_argument("--tokenizer_path", type=str, default='meta-llama/Llama-3.1-8B-Instruct', help="Path to the tokenizer for decoding.")
    parser.add_argument("--dataset", type=str, default='qasper', help="Dataset to analyze.")
    parser.add_argument("--sample_idx_in_file", type=int, default=0, help="The index of the sample to use from the list of common samples (0 for the first).")
    parser.add_argument("--k_percentage", type=float, default=0.1, help="Percentage for top-k analysis (e.g., 0.1 for top 10%).")
    parser.add_argument("--output_file", type=str, default="token_selection_density_deep_dive.pdf", help="Path to save the output PDF plot.")
    args = parser.parse_args()
    
    try:
        oracle_data = load_data(args.oracle_file)
        approx_8b_data = load_data(args.approx_8b_file)
        approx_1b_data = load_data(args.approx_1b_file)
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    except (FileNotFoundError, OSError) as e:
        print(f"Error: {e}\nPlease ensure paths are correct and you are running from the project root.")
        return

    oracle_sample_ids = {s['sample_idx'] for s in oracle_data.get(args.dataset, [])}
    approx_8b_sample_ids = {s['sample_idx'] for s in approx_8b_data.get(args.dataset, [])}
    approx_1b_sample_ids = {s['sample_idx'] for s in approx_1b_data.get(args.dataset, [])}
    
    common_ids = sorted(list(oracle_sample_ids & approx_8b_sample_ids & approx_1b_sample_ids))
    
    if not common_ids or args.sample_idx_in_file >= len(common_ids):
        print(f"Error: Could not find a common sample at index {args.sample_idx_in_file} for dataset '{args.dataset}'.")
        return

    target_sample_id = common_ids[args.sample_idx_in_file]
    print(f"\nAnalyzing common sample with original index: {target_sample_id}")

    oracle_sample = next(s for s in oracle_data[args.dataset] if s['sample_idx'] == target_sample_id)
    approx_8b_sample = next(s for s in approx_8b_data[args.dataset] if s['sample_idx'] == target_sample_id)
    approx_1b_sample = next(s for s in approx_1b_data[args.dataset] if s['sample_idx'] == target_sample_id)
    
    layer_to_use = 15

    input_ids = torch.tensor(oracle_sample['input_ids'])
    k = int(len(input_ids) * args.k_percentage)
    all_indices = {}

    # 1. Oracle
    all_indices['Oracle (8B Answer-Informed)'] = get_top_k_indices(torch.tensor(oracle_sample['ranking']), k)

    # 2. FastKV-style (8B Single Layer)
    layerwise_rankings_8b = approx_8b_sample['layerwise_rankings']
    if layer_to_use in layerwise_rankings_8b:
        fastkv_ranking = torch.tensor(layerwise_rankings_8b[layer_to_use])
        all_indices['FastKV-style (8B Layer 15)'] = get_top_k_indices(fastkv_ranking, k)
    else:
        print(f"Error: Layer {layer_to_use} not found in 8B layerwise rankings. Available: {list(layerwise_rankings_8b.keys())}")
        return

    # 3. Speculative Prefill-style (1B Cumulative)
    cumulative_rankings_1b = approx_1b_sample['cumulative_rankings']
    if layer_to_use in cumulative_rankings_1b:
        spec_prefill_ranking = torch.tensor(cumulative_rankings_1b[layer_to_use])
        all_indices['Speculative Prefill-style (1B Layer 15 Cumulative)'] = get_top_k_indices(spec_prefill_ranking, k)
    else:
        print(f"Error: Layer {layer_to_use} not found in 1B cumulative rankings. Available: {list(cumulative_rankings_1b.keys())}")
        return

    plot_selection_density(
        all_indices,
        len(input_ids),
        args.k_percentage,
        args.dataset,
        target_sample_id,
        args.output_file
    )
    
    print_token_text(tokenizer, all_indices, input_ids)

if __name__ == "__main__":
    main()
