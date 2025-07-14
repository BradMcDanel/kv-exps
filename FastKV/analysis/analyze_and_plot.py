import os
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from tqdm import tqdm
import torch

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 11})

def main():
    parser = argparse.ArgumentParser(description="Compare layer-wise model rankings against the oracle.")
    parser.add_argument("--oracle_file", type=str, required=True)
    parser.add_argument("--method_rankings_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="final_comparison.pdf")
    args = parser.parse_args()

    print("Loading data...")
    with open(args.oracle_file, "rb") as f:
        oracle_data = pickle.load(f)
    with open(args.method_rankings_file, "rb") as f:
        method_data = pickle.load(f)

    datasets_to_plot = sorted(list(oracle_data.keys()))
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey=True)
    axes_flat = axes.flatten()

    for i, dataset_name in enumerate(datasets_to_plot):
        if i >= len(axes_flat): break
        ax = axes_flat[i]
        
        oracle_samples = oracle_data.get(dataset_name, {})
        method_samples = method_data.get(dataset_name, {})
        
        common_indices = sorted(list(set(oracle_samples.keys()) & set(method_samples.keys())))
        
        if not common_indices:
            ax.text(0.5, 0.5, f"No common samples for\n'{dataset_name}'", ha='center', va='center', color='red')
            ax.set_title(dataset_name.replace("_", " ").title())
            continue

        all_base_model_corrs = []
        all_spec_model_corrs = []

        for j, sample_idx in enumerate(common_indices):
            print(i, j)
            oracle_ranking_tensor = oracle_samples[sample_idx]["oracle_ranking"]
            oracle_ranking_full = oracle_ranking_tensor.to(torch.float32).numpy()
            
            # --- Base Model Convergence ---
            base_rankings_list = method_samples[sample_idx].get("base_model_rankings")
            if base_rankings_list and all(t is not None for t in base_rankings_list):
                
                # --- ADDED ASSERTION ---
                method_len = len(base_rankings_list[0])
                oracle_len = len(oracle_ranking_full)
                assert method_len == oracle_len, \
                    f"Length mismatch for {dataset_name} sample {sample_idx} (Base Model)! Method: {method_len}, Oracle: {oracle_len}"
                # -----------------------

                base_corrs = [spearmanr(t.to(torch.float32).numpy(), oracle_ranking_full)[0] for t in base_rankings_list if t is not None]
                all_base_model_corrs.append(base_corrs)
            
            # --- Speculator Model Convergence ---
            spec_rankings_list = method_samples[sample_idx].get("spec_model_rankings")
            if spec_rankings_list and all(t is not None for t in spec_rankings_list):

                # --- ADDED ASSERTION ---
                method_len = len(spec_rankings_list[0])
                oracle_len = len(oracle_ranking_full)
                assert method_len == oracle_len, \
                    f"Length mismatch for {dataset_name} sample {sample_idx} (Spec Model)! Method: {method_len}, Oracle: {oracle_len}"
                # -----------------------

                spec_corrs = [spearmanr(t.to(torch.float32).numpy(), oracle_ranking_full)[0] for t in spec_rankings_list if t is not None]
                all_spec_model_corrs.append(spec_corrs)

        ax.set_title(f"{dataset_name.replace('_', ' ').title()} (n={len(common_indices)})", fontsize=12)

        # Plotting logic remains the same...
        if all_base_model_corrs:
            base_matrix = np.array(all_base_model_corrs)
            mean_base = np.nanmean(base_matrix, axis=0)
            std_base = np.nanstd(base_matrix, axis=0)
            x_axis_base = np.arange(1, len(mean_base) + 1)
            ax.plot(x_axis_base, mean_base, marker='o', linestyle='-', markersize=4, color='darkslateblue', label='8B Model vs. 8B Oracle')
            ax.fill_between(x_axis_base, mean_base - std_base, mean_base + std_base, alpha=0.2, color='darkslateblue')

        if all_spec_model_corrs:
            max_layers_spec = max(len(c) for c in all_spec_model_corrs)
            padded_spec = np.array([c + [np.nan]*(max_layers_spec-len(c)) for c in all_spec_model_corrs])
            mean_spec = np.nanmean(padded_spec, axis=0)
            std_spec = np.nanstd(padded_spec, axis=0)
            x_axis_spec = np.arange(1, len(mean_spec) + 1)
            ax.plot(x_axis_spec, mean_spec, marker='s', linestyle='--', markersize=4, color='crimson', label='1B Model vs. 8B Oracle')
            ax.fill_between(x_axis_spec, mean_spec - std_spec, mean_spec + std_spec, alpha=0.2, color='crimson')

        ax.grid(True, which='both', linestyle=':', alpha=0.7)

    for i in range(len(datasets_to_plot), len(axes_flat)):
        axes_flat[i].set_visible(False)
    
    fig.supxlabel('Model Layer Index', fontsize=16)
    fig.supylabel("Correlation (ρ) to Answer-Informed Oracle", fontsize=16)
    axes[0, 0].set_ylim(bottom=0.0, top=1.05)
    
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.96), ncol=2, fontsize=12)
    
    base_model_str = args.method_rankings_file.split('basemodel_')[-1].split('_specmodel_')[0].replace('_', '/')
    fig.suptitle(f'Layer-wise Ranking Convergence to Oracle ({base_model_str})', fontsize=20, y=1.0)
    
    plt.subplots_adjust(left=0.07, right=0.98, top=0.90, bottom=0.08, wspace=0.15, hspace=0.25)
    plt.savefig(args.output_file, format='pdf', dpi=300)
    print(f"\nFinal comparison plot saved to: {args.output_file}")


if __name__ == "__main__":
    main()
