# analysis/visualize_token_selection.py

import argparse
import os
import pickle
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import gaussian_kde
from transformers import AutoTokenizer

from .viz_utils import set_publication_style, METHOD_COLORS

def load_npz_data_for_dataset(base_path, model_name, dataset_name):
    file_path = os.path.join(base_path, model_name, f"{dataset_name}.npz")
    if not os.path.exists(file_path): raise FileNotFoundError(f"FATAL: Data file not found: {file_path}")
    try:
        npz_file = np.load(file_path, allow_pickle=True)
        return {key: npz_file[key].item() for key in npz_file.files}
    except Exception as e: raise IOError(f"FATAL: Could not load {file_path}. Error: {e}")

def get_top_k_indices(scores, k, max_index=None):
    if max_index: scores = scores[:max_index]
    if scores.numel() < k: raise ValueError(f"FATAL: Cannot select top {k}, only {scores.numel()} available.")
    _, top_k_indices = torch.topk(scores, k=k)
    return top_k_indices

def print_token_text(tokenizer, all_indices, all_tokens, num_tokens_to_print=512):
    print("\n" + "="*80 + "\nQualitative Deep Dive: Text of Top-Selected Tokens\n" + "="*80 + "\n")
    canonical_order = ['Oracle', 'FastKV (Layer 15)', 'GemFilter (Layer 13)']
    actual_speculative_key = next((key for key in all_indices if key.startswith('Speculative Prefill')), None)
    if actual_speculative_key: canonical_order.append(actual_speculative_key)
    methods_to_print = [name for name in canonical_order if name in all_indices]
    for name in methods_to_print:
        indices = all_indices[name]
        header = f"--- {name} (Top {min(len(indices), num_tokens_to_print)} of {len(indices)}) ---"
        print(header)
        sorted_indices = torch.sort(indices).values[:num_tokens_to_print]
        groups = []
        if sorted_indices.numel() > 0:
            current_group = [sorted_indices[0].item()]
            for i in range(1, len(sorted_indices)):
                if sorted_indices[i] == sorted_indices[i-1] + 1: current_group.append(sorted_indices[i].item())
                else:
                    groups.append(current_group)
                    current_group = [sorted_indices[i].item()]
            groups.append(current_group)
        decoded_parts = [f"...'{repr(tokenizer.decode(all_tokens[torch.tensor(g)]))}'..." for g in groups]
        print("  " + " | ".join(decoded_parts) + "\n")

def plot_selection_density(all_indices, sequence_length, k_percentage, dataset_name, output_pdf_file, output_png_file):
    """Creates a density plot using the unified color scheme."""
    fig, ax = plt.subplots(figsize=(20, 7))

    # Define a canonical order for plotting layers
    plot_order_map = {
        'Oracle': {'label': 'Oracle', 'color': METHOD_COLORS['Oracle']},
        'FastKV (Layer 15)': {'label': 'FastKV (Layer 15)', 'color': METHOD_COLORS['FastKV']},
        'GemFilter (Layer 13)': {'label': 'GemFilter (Layer 13)', 'color': METHOD_COLORS['GemFilter']},
    }
    # Find the dynamic speculative key to add it to the map
    spec_key = next((k for k in all_indices if k.startswith('Speculative Prefill')), None)
    if spec_key:
        plot_order_map[spec_key] = {'label': spec_key, 'color': METHOD_COLORS['Speculative']}

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
                    ax.hist(indices_np, bins=50, density=True, color=props['color'], alpha=0.5, label=f"{props['label']} (hist)")
    
    ax.set_xlabel('Token Position in Prompt')
    ax.set_ylabel('Density of Selected Tokens')
    ax.grid(axis='y', linestyle=':', linewidth=0.7)
    ax.spines[['right', 'top']].set_visible(False)
    ax.legend(title="Ranking Method")
    ax.set_xlim(0, sequence_length)
    ax.set_yticklabels([])
    ax.tick_params(axis='y', length=0)
    title = f'Density of Top-{k_percentage:.0%} Selected Tokens\n'
    title += f'Dataset: {dataset_name.replace("_", " ").title()} (Seq Len: {sequence_length})'
    ax.set_title(title, pad=20)
    plt.tight_layout()
    
    plt.savefig(output_pdf_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved PDF to: {output_pdf_file}")
    plt.savefig(output_png_file, dpi=300, bbox_inches='tight')
    print(f"Saved PNG to: {output_png_file}")
    plt.close(fig)

def main():
    set_publication_style() # Apply unified styling
    parser = argparse.ArgumentParser(description="Deep dive density plot for token selection.")
    TARGET_MODEL = 'meta-llama/Llama-3.1-8B-Instruct'
    DRAFT_MODEL = 'meta-llama/Llama-3.2-1B-Instruct'
    MODELS = {
        'oracle': {'sanitized_name': TARGET_MODEL.replace('/', '_'), 'base_path': 'analysis_results/oracles'},
        'approx_target': {'sanitized_name': TARGET_MODEL.replace('/', '_'), 'base_path': 'analysis_results/approx_rankings'},
        'approx_draft': {'sanitized_name': DRAFT_MODEL.replace('/', '_'), 'base_path': 'analysis_results/approx_rankings'},
    }
    parser.add_argument("--tokenizer_path", type=str, default=TARGET_MODEL)
    parser.add_argument("--dataset", type=str, default='qasper')
    parser.add_argument("--sample_idx_in_file", type=int, default=0)
    parser.add_argument("--k_percentage", type=float, default=0.1)
    parser.add_argument("--output_name", type=str, default="token_selection_density_deep_dive")
    args = parser.parse_args()
    
    output_dir = "figures"
    os.makedirs(output_dir, exist_ok=True)
    output_pdf_file = os.path.join(output_dir, f"{args.output_name}.pdf")
    output_png_file = os.path.join(output_dir, f"{args.output_name}.png")

    oracle_data = load_npz_data_for_dataset(MODELS['oracle']['base_path'], MODELS['oracle']['sanitized_name'], args.dataset)
    approx_target_data = load_npz_data_for_dataset(MODELS['approx_target']['base_path'], MODELS['approx_target']['sanitized_name'], args.dataset)
    approx_draft_data = load_npz_data_for_dataset(MODELS['approx_draft']['base_path'], MODELS['approx_draft']['sanitized_name'], args.dataset)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    common_keys = sorted(list(set(oracle_data.keys()) & set(approx_target_data.keys()) & set(approx_draft_data.keys())))
    target_sample_key = common_keys[args.sample_idx_in_file]

    oracle_sample = oracle_data[target_sample_key]
    approx_target_sample = approx_target_data[target_sample_key]
    approx_draft_sample = approx_draft_data[target_sample_key]
    
    for data_dict in [approx_target_sample, approx_draft_sample]:
        for rank_type in ['fastkv_rankings', 'gemfilter_rankings', 'speculative_rankings']:
            if rank_type in data_dict and isinstance(data_dict[rank_type], bytes):
                data_dict[rank_type] = pickle.loads(data_dict[rank_type])

    fastkv_layer, gemfilter_layer, spec_k_candidates = 15, 13, [8, 4, 2, 1]
    input_ids = torch.from_numpy(oracle_sample['input_ids'])
    seq_len = len(input_ids)
    k_absolute = int(seq_len * args.k_percentage)
    all_indices = {}

    all_indices['Oracle'] = get_top_k_indices(torch.from_numpy(oracle_sample['ranking']).float(), k_absolute, seq_len)
    all_indices['FastKV (Layer 15)'] = get_top_k_indices(torch.from_numpy(approx_target_sample['fastkv_rankings'][fastkv_layer]).float(), k_absolute, seq_len)
    all_indices['GemFilter (Layer 13)'] = get_top_k_indices(torch.from_numpy(approx_target_sample['gemfilter_rankings'][gemfilter_layer]).float(), k_absolute, seq_len)
    
    spec_rankings = approx_draft_sample.get('speculative_rankings', {})
    found_spec_k = next((k for k in spec_k_candidates if k in spec_rankings), None)
    if found_spec_k:
        scores = torch.from_numpy(spec_rankings[found_spec_k]).float()
        all_indices[f'Speculative Prefill (k={found_spec_k})'] = get_top_k_indices(scores, k_absolute, seq_len)

    plot_selection_density(all_indices, seq_len, args.k_percentage, args.dataset, output_pdf_file, output_png_file)
    print_token_text(tokenizer, all_indices, input_ids)

if __name__ == "__main__":
    main()
