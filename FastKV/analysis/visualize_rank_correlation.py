import os
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from tqdm import tqdm
import torch

# --- Plotting Style ---
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('mako')
plt.rcParams.update({'font.size': 11})

def plot_correlation_grid(multi_dataset_results: dict, args: argparse.Namespace):
    """Creates the 2x3 grid of Spearman's Rank Correlation plots."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey=True)
    axes_flat = axes.flatten()
    
    for i, dataset_name in enumerate(args.datasets_to_plot):
        ax = axes_flat[i]
        sample_rankings_list = multi_dataset_results.get(dataset_name)

        if not sample_rankings_list:
            ax.text(0.5, 0.5, f"No data for\n'{dataset_name}'", ha='center', va='center', color='red')
            ax.set_title(dataset_name.replace("_", " ").title())
            continue

        original_layer_indices = sorted(sample_rankings_list[0].keys())
        if len(original_layer_indices) < 3:
            ax.text(0.5, 0.5, 'Not enough layers', ha='center', va='center')
            ax.set_title(dataset_name.replace("_", " ").title())
            continue
            
        oracle_layer_idx = original_layer_indices[-2] # Penultimate layer
        layers_to_analyze = original_layer_indices[1 : original_layer_indices.index(oracle_layer_idx) + 1]
        
        all_adj_corrs, all_oracle_corrs = [], []
        
        for sample_rankings in sample_rankings_list:
            if len(sample_rankings) != len(original_layer_indices): continue
            
            oracle_ranking = sample_rankings[oracle_layer_idx].to(torch.float32).numpy()
            adj_corrs, oracle_corrs = [], []
            for i in layers_to_analyze:
                corr_adj, _ = spearmanr(sample_rankings[i].to(torch.float32), sample_rankings[i-1].to(torch.float32))
                corr_oracle, _ = spearmanr(sample_rankings[i].to(torch.float32), oracle_ranking)
                adj_corrs.append(corr_adj)
                oracle_corrs.append(corr_oracle)
            all_adj_corrs.append(adj_corrs)
            all_oracle_corrs.append(oracle_corrs)

        ax.set_title(f"{dataset_name.replace('_', ' ').title()} (n={len(all_adj_corrs)})", fontsize=12)
        
        # Trim final point for plotting
        x_axis_plot = [i + 1 for i in layers_to_analyze[:-1]]
        
        # Plot Adjacent Correlation
        mean_adj = np.mean(all_adj_corrs, axis=0)
        std_adj = np.std(all_adj_corrs, axis=0)
        ax.plot(x_axis_plot, mean_adj[:-1], marker='o', linestyle='-', markersize=4)
        ax.fill_between(x_axis_plot, (mean_adj - std_adj)[:-1], (mean_adj + std_adj)[:-1], alpha=0.2, color=sns.color_palette('mako')[1])
        
        # Plot Correlation to Oracle
        oracle_corrs_trimmed = np.array(all_oracle_corrs)[:, :-1]
        mean_oracle = np.mean(oracle_corrs_trimmed, axis=0)
        std_oracle = np.std(oracle_corrs_trimmed, axis=0)
        ax.plot(x_axis_plot, mean_oracle, marker='s', linestyle='--', markersize=4)
        ax.fill_between(x_axis_plot, mean_oracle - std_oracle, mean_oracle + std_oracle, alpha=0.2, color=sns.color_palette('mako')[3])
        
        ax.grid(True, which='both', linestyle=':', alpha=0.7)

    for i in range(len(args.datasets_to_plot), len(axes_flat)):
        axes_flat[i].set_visible(False)
        
    fig.supxlabel('Model Layer Index', fontsize=16)
    fig.supylabel("Spearman's Rank Correlation (ρ)", fontsize=16)
    axes[0, 0].set_ylim(bottom=0.2, top=1.05)
    
    handles = [
        plt.Line2D([0], [0], color=sns.color_palette('mako')[0], marker='o', linestyle='-', label='Adjacent Layer Correlation (Mean)'),
        plt.Line2D([0], [0], color=sns.color_palette('mako')[3], marker='s', linestyle='--', label=f'Correlation to Penultimate Layer (Mean)')
    ]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.96), ncol=2, fontsize=12)
    
    model_name_str = args.input_pkl.split('model_')[-1].replace('.pkl', '').replace('_', '/')
    fig.suptitle(f'Token Ranking Correlation for {model_name_str}', fontsize=20, y=1.0)
    
    plt.subplots_adjust(left=0.07, right=0.98, top=0.90, bottom=0.08, wspace=0.15, hspace=0.25)
    plt.savefig(args.output_file, format='pdf', dpi=300)
    plt.close(fig)
    print(f"\nRank correlation plot saved to: {args.output_file}")

def main():
    parser = argparse.ArgumentParser(description="Visualize aggregated Spearman's Rank Correlation.")
    parser.add_argument("input_pkl", type=str, help="Path to the aggregated .pkl file.")
    parser.add_argument("--datasets_to_plot", nargs='+', required=True, help="A space-separated list of up to 6 datasets to plot.")
    parser.add_argument("--output_file", type=str, default="rank_correlation.pdf", help="Path to save the output PDF plot.")
    args = parser.parse_args()
    
    try:
        with open(args.input_pkl, "rb") as f:
            multi_dataset_results = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at {args.input_pkl}"); return

    plot_correlation_grid(multi_dataset_results, args)

if __name__ == "__main__":
    main()
