import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from .viz_utils import (
    set_publication_style,
    METHOD_COLORS,
    TASKS_AND_DATASETS,
    ALL_DATASETS_TO_PLOT,
    DATASET_TO_TASK_MAP
)
# We need the new data loading function
from .retrieval_metrics import load_npz_data_for_dataset, get_per_sample_accuracies_long_form

# --- Debug Data Generation ---

def generate_dummy_data_long_form() -> pd.DataFrame:
    """Generates random data in the long-form DataFrame format for debugging."""
    print("--- Using --debug mode: Generating random dummy data ---")
    records = []
    num_samples = 10
    num_layers = 32

    for task_category, datasets in TASKS_AND_DATASETS.items():
        for dataset in datasets:
            for sample_idx in range(num_samples):
                # Simulate trends: FastKV > GemFilter, higher correlation for QA
                base_corr_fastkv = 0.7 if 'QA' in task_category else 0.5
                base_corr_gemfilter = base_corr_fastkv - 0.1
                base_corr_spec = 0.6 if 'QA' in task_category else 0.45

                for layer in range(num_layers):
                    # Simulate layer-wise trend (improves then plateaus)
                    layer_factor = min(1, layer / 15.0)
                    
                    # Add noise/variance
                    noise = np.random.normal(0, 0.1)

                    records.append({'task_category': task_category, 'method': 'FastKV', 'layer': layer, 
                                    'correlation': np.clip(base_corr_fastkv * layer_factor + noise, -0.1, 0.9)})
                    records.append({'task_category': task_category, 'method': 'GemFilter', 'layer': layer, 
                                    'correlation': np.clip(base_corr_gemfilter * layer_factor + noise - 0.05, -0.1, 0.9)})
                    # CLAA should perform slightly above FastKV but smoother (less noise)
                    base_corr_claa = base_corr_fastkv + 0.03
                    claa_noise = np.random.normal(0, 0.05)  # Half the noise of FastKV
                    records.append({'task_category': task_category, 'method': 'CLAA', 'layer': layer,
                                    'correlation': np.clip(base_corr_claa * layer_factor + claa_noise, -0.1, 0.9)})
                    # SpecPrefill is constant per sample but has variance across samples
                    records.append({'task_category': task_category, 'method': 'SpecPrefill', 'layer': layer, 
                                    'correlation': np.clip(base_corr_spec + np.random.normal(0, 0.05), -0.1, 0.9)})

    return pd.DataFrame(records)


# --- Main Plotting Function ---

def plot_paper_version_with_variance(full_df: pd.DataFrame, output_prefix: str):
    """Creates the 2x3 grid with confidence intervals averaged by task."""
    set_publication_style()
    plt.rcParams.update({'lines.linewidth': 2.5, 'lines.markersize': 0}) # No markers needed

    n_rows, n_cols = 2, 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 10), sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0.25, wspace=0.05)
    
    plt.ylim(bottom=-0.1, top=0.9) # Set a fixed, intuitive range for correlation
    plt.xlim(left=-1, right=32)

    task_items = list(TASKS_AND_DATASETS.keys())
    for i, ax in enumerate(axes.flat):
        task_name = task_items[i]
        task_df = full_df[full_df['task_category'] == task_name]

        if not task_df.empty:
            sns.lineplot(
                data=task_df,
                x='layer',
                y='correlation',
                hue='method',
                palette={
                    'GemFilter': METHOD_COLORS['GemFilter'],
                    'FastKV': METHOD_COLORS['FastKV'],
                    'CLAA': METHOD_COLORS['CLAA'],
                    'SpecPrefill': METHOD_COLORS['Speculative'],
                },
                hue_order=['GemFilter', 'FastKV', 'CLAA', 'SpecPrefill'],
                style='method',
                dashes={'GemFilter': '', 'FastKV': '', 'CLAA': '', 'SpecPrefill': (2, 2)},
                errorbar='se',  # Use standard error instead of 95% CI
                ax=ax,
                legend=False # We will create a single legend later
            )

        ax.set_title(task_name, fontsize=28)
        ax.set_xticks([0, 8, 16, 24, 31])
        ax.grid(True, which='major', linestyle=':', linewidth=0.6)
        ax.set_xlabel('') # Remove individual x-labels
        ax.set_ylabel('') # Remove individual y-labels
    
    # Create a single, shared legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=METHOD_COLORS['GemFilter'], lw=3, label='GemFilter'),
        Line2D([0], [0], color=METHOD_COLORS['FastKV'], lw=3, label='FastKV'),
        Line2D([0], [0], color=METHOD_COLORS['CLAA'], lw=3, label='CLAA'),
        Line2D([0], [0], color=METHOD_COLORS['Speculative'], lw=3, linestyle='--', label='Spec. Prefill')
    ]
    axes.flat[-1].legend(handles=legend_elements, loc='lower right',
                         frameon=True, facecolor='white', framealpha=0.9,
                         fontsize=18)

    fig.text(0.5, 0.04, 'Model Layer Index', ha='center', va='center', fontsize=34)
    fig.text(0.06, 0.5, 'Spearman Rank Correlation with Oracle', ha='center', va='center', rotation='vertical', fontsize=30)
    fig.suptitle(f'Comparing Token Ranking Heuristics against Oracle', y=0.99, fontsize=32, weight='bold')

    plt.savefig(f"{output_prefix}.pdf", format='pdf', dpi=300, bbox_inches='tight')
    print(f"\nPaper version plot with variance saved to: {output_prefix}.pdf")
    plt.savefig(f"{output_prefix}.png", format='png', dpi=300, bbox_inches='tight')
    print(f"Paper version plot with variance saved to: {output_prefix}.png")
    plt.close(fig)

def main():
    """Main execution block."""
    parser = argparse.ArgumentParser(description="Visualize ranking methods with variance.")
    parser.add_argument("--debug", action="store_true", help="Use dummy data for fast layout iteration.")
    args = parser.parse_args()

    if args.debug:
        full_df = generate_dummy_data_long_form()
    else:
        # Import data loading infrastructure only when needed
        from .visualize_rankings_comparison import load_all_data, MODELS_TO_LOAD
        results = load_all_data(MODELS_TO_LOAD, ALL_DATASETS_TO_PLOT)
        
        all_dfs = []
        for ds in tqdm(ALL_DATASETS_TO_PLOT, desc="Calculating Per-Sample Accuracies"):
            # Remember to apply the .item() fix in retrieval_metrics.py
            df = get_per_sample_accuracies_long_form(ds, results)
            if not df.empty:
                df['task_category'] = DATASET_TO_TASK_MAP[ds]
                all_dfs.append(df)
                
        if not all_dfs:
            print("No valid data found to plot. Exiting.")
            return

        full_df = pd.concat(all_dfs, ignore_index=True)
    
    output_dir = "figures"
    os.makedirs(output_dir, exist_ok=True)
    plot_paper_version_with_variance(full_df, os.path.join(output_dir, "ranking_comparison_with_variance"))

if __name__ == "__main__":
    main()
