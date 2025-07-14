import os
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from tqdm import tqdm
import torch
import math

# --- Plotting Style ---
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('mako')
plt.rcParams.update({'font.size': 11})

# --- Analysis Functions (remain the same) ---
def calculate_full_rank_correlation(sample_rankings: dict, oracle_layer_idx: int, layers_to_analyze: list):
    adj_corrs, oracle_corrs = [], []
    oracle_ranking = sample_rankings[oracle_layer_idx].to(torch.float32).numpy()
    for i in layers_to_analyze:
        corr_adj, _ = spearmanr(sample_rankings[i].to(torch.float32), sample_rankings[i-1].to(torch.float32))
        corr_oracle, _ = spearmanr(sample_rankings[i].to(torch.float32), oracle_ranking)
        adj_corrs.append(corr_adj)
        oracle_corrs.append(corr_oracle)
    return adj_corrs, oracle_corrs

def calculate_top_k_rank_correlation(sample_rankings: dict, oracle_layer_idx: int, layers_to_analyze: list, k_percentage: float):
    adj_corrs, oracle_corrs = [], []
    oracle_scores = sample_rankings[oracle_layer_idx]
    k = max(1, math.ceil(len(oracle_scores) * k_percentage))
    _, top_k_oracle_indices = torch.topk(oracle_scores, k=k)
    oracle_ranking_filtered = oracle_scores[top_k_oracle_indices].to(torch.float32).numpy()
    for i in layers_to_analyze:
        layer_ranking_filtered = sample_rankings[i][top_k_oracle_indices].to(torch.float32).numpy()
        prev_layer_ranking_filtered = sample_rankings[i-1][top_k_oracle_indices].to(torch.float32).numpy()
        corr_adj, _ = spearmanr(layer_ranking_filtered, prev_layer_ranking_filtered)
        corr_oracle, _ = spearmanr(layer_ranking_filtered, oracle_ranking_filtered)
        adj_corrs.append(corr_adj)
        oracle_corrs.append(corr_oracle)
    return adj_corrs, oracle_corrs

def calculate_retrieval_accuracy(sample_rankings: dict, oracle_layer_idx: int, layers_to_analyze: list, k_percentage: float):
    accuracies = []
    oracle_scores = sample_rankings[oracle_layer_idx]
    k = max(1, math.ceil(len(oracle_scores) * k_percentage))
    _, top_k_oracle_indices = torch.topk(oracle_scores, k=k)
    oracle_set = set(top_k_oracle_indices.tolist())
    for i in layers_to_analyze:
        layer_scores = sample_rankings[i]
        _, top_k_layer_indices = torch.topk(layer_scores, k=k)
        layer_set = set(top_k_layer_indices.tolist())
        intersection_size = len(oracle_set.intersection(layer_set))
        accuracy = intersection_size / k if k > 0 else 0
        accuracies.append(accuracy)
    return None, accuracies

# --- Main Plotting Orchestrator ---
def plot_stability_grid(multi_dataset_results: dict, args: argparse.Namespace):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey=True)
    axes_flat = axes.flatten()
    is_rank_corr = args.analysis_type in ['full_rank_corr', 'top_k_rank_corr']
    
    for i, dataset_name in enumerate(args.datasets_to_plot):
        ax = axes_flat[i]
        sample_rankings_list = multi_dataset_results.get(dataset_name)

        if not sample_rankings_list:
            ax.text(0.5, 0.5, f"No data for\n'{dataset_name}'", ha='center', va='center', color='red')
            ax.set_title(dataset_name.replace("_", " ").title())
            continue

        original_layer_indices = sorted(sample_rankings_list[0].keys())
        
        # --- NEW: Oracle Layer Selection ---
        if args.use_final_layer_as_oracle:
            oracle_layer_idx = original_layer_indices[-1]
            oracle_label = "Final Layer"
        else:
            oracle_layer_idx = original_layer_indices[-2]
            oracle_label = "Penultimate Layer"
        
        # We analyze up to the layer *before* the oracle
        layers_to_analyze = original_layer_indices[1 : original_layer_indices.index(oracle_layer_idx) + 1]
        
        all_series1, all_series2 = [], []
        
        for sample_rankings in sample_rankings_list:
            if len(sample_rankings) != len(original_layer_indices): continue
            
            if args.analysis_type == 'full_rank_corr':
                s1, s2 = calculate_full_rank_correlation(sample_rankings, oracle_layer_idx, layers_to_analyze)
            elif args.analysis_type == 'top_k_rank_corr':
                s1, s2 = calculate_top_k_rank_correlation(sample_rankings, oracle_layer_idx, layers_to_analyze, args.k_percentage)
            else: # retrieval_accuracy
                s1, s2 = calculate_retrieval_accuracy(sample_rankings, oracle_layer_idx, layers_to_analyze, args.k_percentage)
            
            if s1 is not None: all_series1.append(s1)
            if s2 is not None: all_series2.append(s2)

        ax.set_title(f"{dataset_name.replace('_', ' ').title()} (n={len(sample_rankings_list)})", fontsize=12)
        
        # --- NEW: Trim the final point for plotting ---
        # The x-axis for the oracle correlation should not include the oracle layer itself
        x_axis_oracle_plot = [i + 1 for i in layers_to_analyze[:-1]]
        x_axis_adj_plot = [i + 1 for i in layers_to_analyze]
        
        # Plot Series 2 (Correlation to Oracle / Accuracy)
        if all_series2:
            s2_matrix = np.array(all_series2)[:, :-1] # Exclude the final self-correlation point
            mean2 = np.mean(s2_matrix, axis=0)
            std2 = np.std(s2_matrix, axis=0)
            ax.plot(x_axis_oracle_plot, mean2, marker='s', linestyle='--', markersize=4)
            ax.fill_between(x_axis_oracle_plot, mean2 - std2, mean2 + std2, alpha=0.2, color=sns.color_palette('mako')[3])
            
        # Plot Series 1 (Adjacent Correlation)
        if all_series1:
            mean1 = np.mean(all_series1, axis=0)
            std1 = np.std(all_series1, axis=0)
            ax.plot(x_axis_adj_plot, mean1, marker='o', linestyle='-', markersize=4)
            ax.fill_between(x_axis_adj_plot, mean1 - std1, mean1 + std1, alpha=0.2, color=sns.color_palette('mako')[1])

        ax.grid(True, which='both', linestyle=':', alpha=0.7)

    # --- Final Figure Formatting ---
    for i in range(len(args.datasets_to_plot), len(axes_flat)):
        axes_flat[i].set_visible(False)
        
    fig.supxlabel('Model Layer Index', fontsize=16)
    
    if is_rank_corr:
        fig.supylabel("Spearman's Rank Correlation (ρ)", fontsize=16)
    else:
        fig.supylabel(f"Top-{args.k_percentage:.0%} Retrieval Accuracy (F1)", fontsize=16)
    axes[0, 0].set_ylim(bottom=0.2, top=1.05)

    # Legend
    handles = []
    if is_rank_corr:
        handles.append(plt.Line2D([0], [0], color=sns.color_palette('mako')[0], marker='o', linestyle='-', label='Adjacent Correlation (Mean)'))
        handles.append(plt.Line2D([0], [0], color=sns.color_palette('mako')[3], marker='s', linestyle='--', label=f'Correlation to {oracle_label} (Mean)'))
    else:
        handles.append(plt.Line2D([0], [0], color=sns.color_palette('mako')[3], marker='s', linestyle='--', label=f'Accuracy vs. {oracle_label} (Top-{args.k_percentage:.0%})'))
    
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.96), ncol=2, fontsize=12)
    
    model_name_str = args.input_pkl.split('model_')[-1].replace('.pkl', '').replace('_', '/')
    fig.suptitle(f'Token Importance Stability for {model_name_str}', fontsize=20, y=1.0)
    
    plt.subplots_adjust(left=0.07, right=0.98, top=0.90, bottom=0.08, wspace=0.15, hspace=0.25)
    plt.savefig(args.output_file, format='pdf', dpi=300)
    plt.close(fig)
    print(f"\nAnalysis plot saved to: {args.output_file}")

def main():
    parser = argparse.ArgumentParser(description="Advanced visualization of token ranking stability.")
    parser.add_argument("input_pkl", type=str, help="Path to the aggregated .pkl file.")
    parser.add_argument("--datasets_to_plot", nargs='+', required=True, help="Space-separated list of up to 6 datasets to plot.")
    parser.add_argument("--analysis_type", type=str, default='full_rank_corr',
                        choices=['full_rank_corr', 'top_k_rank_corr', 'retrieval_accuracy'],
                        help="The type of stability analysis to perform.")
    parser.add_argument("--k_percentage", type=float, default=0.1, help="The percentage of top tokens for filtered analysis.")
    # --- NEW FLAG ---
    parser.add_argument("--use_final_layer_as_oracle", action='store_true', help="If set, use the final layer (L-1) as the oracle instead of the penultimate (L-2).")
    parser.add_argument("--output_file", type=str, default="token_ranking_analysis.pdf", help="Path to save the output PDF plot.")
    args = parser.parse_args()
    
    if not (0 < args.k_percentage <= 1):
        raise ValueError("--k_percentage must be between 0 and 1.")
    
    try:
        with open(args.input_pkl, "rb") as f:
            multi_dataset_results = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at {args.input_pkl}"); return

    plot_stability_grid(multi_dataset_results, args)

if __name__ == "__main__":
    main()
