import os
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch
import math

# --- Plotting Style ---
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('flare')
plt.rcParams.update({'font.size': 11})

def plot_accuracy_grid(multi_dataset_results: dict, args: argparse.Namespace):
    """Creates the 2x3 grid of F1 Retrieval Accuracy plots using the final layer as the oracle."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey=True)
    axes_flat = axes.flatten()
    
    palette = sns.color_palette('flare', n_colors=len(args.k_percentages))
    legend_handles = None
    
    for i, dataset_name in enumerate(args.datasets_to_plot):
        ax = axes_flat[i]
        sample_rankings_list = multi_dataset_results.get(dataset_name)

        if not sample_rankings_list:
            # Handle empty data
            continue

        original_layer_indices = sorted(sample_rankings_list[0].keys())
        num_layers = len(original_layer_indices)
        if num_layers < 2:
            # Handle insufficient layers
            continue
            
        # --- MODIFIED: Use the final layer (L-1) as the oracle ---
        oracle_layer_idx = original_layer_indices[-2]
        
        # We analyze all layers up to the one before the oracle
        layers_to_analyze = original_layer_indices[:-2]
        x_axis_plot = [i + 1 for i in layers_to_analyze]
        
        current_handles = []
        for k_idx, k_perc in enumerate(args.k_percentages):
            all_accuracies = []
            for sample_rankings in sample_rankings_list:
                if len(sample_rankings) != num_layers: continue
                
                # --- This function calculates accuracy for all layers in layers_to_analyze ---
                oracle_scores = sample_rankings[oracle_layer_idx]
                k = max(1, math.ceil(len(oracle_scores) * k_perc))
                _, top_k_oracle_indices = torch.topk(oracle_scores, k=k)
                oracle_set = set(top_k_oracle_indices.tolist())
                
                accuracies = []
                for layer_idx in layers_to_analyze:
                    layer_scores = sample_rankings[layer_idx]
                    _, top_k_layer_indices = torch.topk(layer_scores, k=k)
                    layer_set = set(top_k_layer_indices.tolist())
                    intersection_size = len(oracle_set.intersection(layer_set))
                    accuracy = intersection_size / k if k > 0 else 0
                    accuracies.append(accuracy)
                
                all_accuracies.append(accuracies)

            if not all_accuracies: continue
            
            mean_acc = np.mean(all_accuracies, axis=0)
            std_acc = np.std(all_accuracies, axis=0)

            line, = ax.plot(x_axis_plot, mean_acc, marker='s', linestyle='--', markersize=4, color=palette[k_idx], label=f'Top-{k_perc:.0%}')
            ax.fill_between(x_axis_plot, mean_acc - std_acc, mean_acc + std_acc, alpha=0.15, color=palette[k_idx])
            current_handles.append(line)
        
        if current_handles and not legend_handles:
             legend_handles = current_handles

        ax.set_title(f"{dataset_name.replace('_', ' ').title()} (n={len(sample_rankings_list)})", fontsize=12)
        ax.grid(True, which='both', linestyle=':', alpha=0.7)
    
    for i in range(len(args.datasets_to_plot), len(axes_flat)):
        axes_flat[i].set_visible(False)
        
    fig.supxlabel('Model Layer Index', fontsize=16)
    fig.supylabel("Retrieval Accuracy (F1-Score)", fontsize=16)
    axes[0, 0].set_ylim(bottom=0.2, top=1.05)
    
    if legend_handles:
        last_ax = axes_flat[len(args.datasets_to_plot) - 1]
        labels = [h.get_label() for h in legend_handles]
        last_ax.legend(handles=legend_handles, labels=labels, title='Oracle Set Size', loc='lower right', fontsize=10)
    
    model_name_str = args.input_pkl.split('model_')[-1].replace('.pkl', '').replace('_', '/')
    fig.suptitle(f'Token Importance Retrieval Accuracy for {model_name_str}', fontsize=20, y=0.98)
    
    plt.subplots_adjust(left=0.07, right=0.98, top=0.92, bottom=0.08, wspace=0.15, hspace=0.25)
    plt.savefig(args.output_file, format='pdf', dpi=300)
    plt.close(fig)
    print(f"\nRetrieval accuracy plot saved to: {args.output_file}")

def main():
    parser = argparse.ArgumentParser(description="Visualize aggregated F1 Retrieval Accuracy for multiple k-percentages.")
    parser.add_argument("input_pkl", type=str, help="Path to the aggregated .pkl file.")
    parser.add_argument("--datasets_to_plot", nargs='+', required=True, help="Space-separated list of up to 6 datasets to plot.")
    # --- NEW: Takes multiple percentages ---
    parser.add_argument("--k_percentages", type=float, nargs='+', default=[0.1, 0.2], help="Space-separated list of percentages for top-k analysis (e.g., 0.05 0.1 0.2).")
    parser.add_argument("--output_file", type=str, default="retrieval_accuracy.pdf", help="Path to save the output PDF plot.")
    args = parser.parse_args()
    
    for k_perc in args.k_percentages:
        if not (0 < k_perc <= 1):
            raise ValueError("--k_percentages must be between 0 and 1.")
    
    try:
        with open(args.input_pkl, "rb") as f:
            multi_dataset_results = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at {args.input_pkl}"); return

    plot_accuracy_grid(multi_dataset_results, args)

if __name__ == "__main__":
    main()
