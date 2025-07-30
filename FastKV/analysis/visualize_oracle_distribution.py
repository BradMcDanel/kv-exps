# analysis/visualize_oracle_distribution.py

import argparse
import os
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import gaussian_kde
from tqdm import tqdm

# Assuming these utilities are in a sibling file or an installed package.
# You will need to ensure TASKS_AND_DATASETS from viz_utils is correctly defined
# to map all 16 datasets to their 6 task categories.
from .viz_utils import (
    set_publication_style,
    TASKS_AND_DATASETS,
    DATASET_NAME_MAP,
)
from .retrieval_metrics import (
    load_npz_data_for_dataset,
    get_top_k_indices
)

ALL_LONG_BENCH_DATASETS = [
    'narrativeqa',
    'qasper',
    'multifieldqa_en',
    'hotpotqa',
    '2wikimqa',
    'musique',
    'gov_report',
    'qmsum',
    'multi_news',
    'trec',
    'triviaqa',
    'samsum',
    'passage_count',
    'passage_retrieval_en',
    'lcc',
    'repobench-p'
]

# ==============================================================================
# SCRIPT DESCRIPTION
# This script generates a 2x3 grid plot visualizing the positional distribution
# of top-k% oracle-ranked tokens, aggregated by task category. This reveals the
# characteristic "importance fingerprint" for different types of long-context tasks.
#
# For example, does QA consistently require tokens from the middle?
# Does summarization follow a U-shape, prioritizing the beginning and end?
#
# This analysis provides a strong, data-driven foundation for discussing why
# different heuristics might succeed or fail.
# ==============================================================================

def generate_dummy_data(k_percentage: float) -> Dict[str, np.ndarray]:
    """
    Generates plausible-looking dummy data for each task category to allow
    for rapid layout and visualization debugging.
    """
    print("--- Using --debug mode: Generating random dummy data ---")
    aggregated_data = {}
    num_tokens = 200000  # Total number of top-k tokens to generate per category

    # --- Define plausible patterns for each task type ---
    # CORRECTED: Keys now exactly match TASKS_AND_DATASETS from viz_utils.py
    patterns = {
        'Single-Doc QA': [
            {'mean': 0.3, 'std': 0.08, 'weight': 0.5},
            {'mean': 0.7, 'std': 0.05, 'weight': 0.5}
        ],
        'Multi-Doc QA': [
            {'mean': 0.2, 'std': 0.05, 'weight': 0.3},
            {'mean': 0.5, 'std': 0.05, 'weight': 0.4},
            {'mean': 0.8, 'std': 0.05, 'weight': 0.3}
        ],
        'Summarization': [
            {'mean': 0.1, 'std': 0.1, 'weight': 0.45},
            {'mean': 0.9, 'std': 0.1, 'weight': 0.45},
            {'mean': 0.5, 'std': 0.2, 'weight': 0.1} # some middle importance
        ],
        'Few-shot Learning': [
            {'mean': 0.05, 'std': 0.04, 'weight': 1.0} # Heavily front-loaded
        ],
        'Code Completion': [ # CORRECTED from 'Code'
            {'mean': 0.5, 'std': 0.25, 'weight': 0.4}, # Context from middle
            {'mean': 0.95, 'std': 0.05, 'weight': 0.6} # And right at the end
        ],
        'Synthetic Task': [ # CORRECTED from 'Synthetic'
            {'mean': 0.5, 'std': 0.3, 'weight': 1.0} # Uniformly distributed
        ],
    }

    # Iterate using the official task names to ensure consistency
    for task_name in TASKS_AND_DATASETS.keys():
        if task_name in patterns:
            task_patterns = patterns[task_name]
            indices = []
            for p in task_patterns:
                count = int(num_tokens * p['weight'])
                indices.append(np.random.normal(loc=p['mean'], scale=p['std'], size=count))
            aggregated_data[task_name] = np.clip(np.concatenate(indices), 0, 1)
        
    return aggregated_data

def aggregate_oracle_indices_by_task(k_percentage: float, model_config: Dict) -> Dict[str, np.ndarray]:
    """
    Loads oracle data for all datasets, computes top-k token positions,
    and aggregates them by their task category.
    """
    # Initialize a dictionary to hold lists of normalized indices for each task category
    aggregated_indices_by_task: Dict[str, List[np.ndarray]] = {task_name: [] for task_name in TASKS_AND_DATASETS}

    pbar = tqdm(ALL_LONG_BENCH_DATASETS, desc="Loading and Processing Datasets")
    for dataset_id in pbar:
        pbar.set_postfix_str(f"Processing {DATASET_NAME_MAP.get(dataset_id, dataset_id)}")
        
        # Find which task category this dataset belongs to
        current_task_category = None
        for task_cat, datasets_in_cat in TASKS_AND_DATASETS.items():
            if dataset_id in datasets_in_cat:
                current_task_category = task_cat
                break
        if not current_task_category:
            continue

        # Load the oracle data for the current dataset
        oracle_data = load_npz_data_for_dataset(
            model_config['base_path'], model_config['sanitized_name'], dataset_id
        )
        if not oracle_data:
            tqdm.write(f"Warning: No oracle data found for {dataset_id}")
            continue

        # Process each sample in the dataset
        for sample in oracle_data.values():
            if 'ranking' not in sample or len(sample['ranking']) == 0:
                continue

            ranking = torch.from_numpy(sample['ranking']).float()
            seq_len = len(ranking)
            if seq_len == 0: continue
            
            k_absolute = int(seq_len * k_percentage)
            if k_absolute == 0: continue

            # Get the indices of the top-k tokens and normalize them
            top_indices = get_top_k_indices(ranking, k_absolute, seq_len)
            normalized_indices = top_indices.float() / (seq_len - 1)
            
            aggregated_indices_by_task[current_task_category].append(normalized_indices.cpu().numpy())

    # Concatenate all indices for each task category into a single NumPy array
    final_aggregated_data = {
        task: np.concatenate(indices)
        for task, indices in aggregated_indices_by_task.items() if indices
    }
    return final_aggregated_data


def plot_oracle_distributions(aggregated_data: Dict[str, np.ndarray], k_percentage: float, output_file_prefix: str):
    """
    Generates and saves the 2x3 grid plot of oracle token distributions.
    """
    set_publication_style()
    plt.rcParams.update({'axes.labelsize': 28, 'xtick.labelsize': 22, 'ytick.labelsize': 22})

    n_rows, n_cols = 2, 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 10), sharex=True, sharey=True)
    fig.suptitle(f'Positional Distribution of Top-{int(k_percentage*100)}% Oracle-Ranked Tokens', fontsize=32, weight='bold')

    x_grid = np.linspace(0, 1, 1000)

    task_order = list(TASKS_AND_DATASETS.keys())
    for i, ax in enumerate(axes.flat):
        if i >= len(task_order):
            ax.axis('off')
            continue
        
        task_name = task_order[i]
        ax.set_title(task_name, fontsize=28)
        
        indices = aggregated_data.get(task_name)
        if indices is None or len(indices) < 2:
            ax.text(0.5, 0.5, "No data", ha='center', va='center')
            continue

        # Calculate and plot Kernel Density Estimate
        try:
            kde = gaussian_kde(indices, bw_method=0.03)
            density = kde(x_grid)
            ax.plot(x_grid, density, color='#0077B6', linewidth=2.5)
            ax.fill_between(x_grid, density, color='#90E0EF', alpha=0.6)
        except (np.linalg.LinAlgError, ValueError) as e:
            ax.text(0.5, 0.5, "KDE Error", ha='center', va='center')
            print(f"KDE failed for {task_name}: {e}")
        
        ax.set_xlim(0, 1)
        ax.grid(True, which='major', linestyle=':', linewidth=0.7)

    # Set common labels
    fig.text(0.5, 0.02, 'Normalized Token Position in Prompt', ha='center', va='center', fontsize=30)
    fig.text(0.06, 0.5, 'Density of Important Tokens', ha='center', va='center', rotation='vertical', fontsize=30)

    plt.tight_layout(rect=[0.08, 0.05, 1, 0.95])
    
    output_pdf = f"{output_file_prefix}.pdf"
    output_png = f"{output_file_prefix}.png"
    plt.savefig(output_pdf, dpi=300, bbox_inches='tight')
    print(f"\nSaved distribution plot to: {output_pdf}")
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f"Saved distribution plot to: {output_png}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize oracle token importance distribution across task categories.")
    parser.add_argument("--k_percentage", type=float, default=0.1, help="Top-k percentage of tokens to consider.")
    parser.add_argument("--debug", action="store_true", help="Use dummy data for fast layout testing.")
    args = parser.parse_args()

    output_dir = "figures"
    os.makedirs(output_dir, exist_ok=True)
    output_prefix = os.path.join(output_dir, "oracle_token_distribution_by_task")

    if args.debug:
        aggregated_data = generate_dummy_data(args.k_percentage)
    else:
        # Configuration for the model whose oracle data we want to analyze
        model_config = {
            'sanitized_name': 'meta-llama_Llama-3.1-8B-Instruct',
            'base_path': 'analysis_results/oracles'
        }
        aggregated_data = aggregate_oracle_indices_by_task(args.k_percentage, model_config)

    if not aggregated_data:
        print("No data was aggregated. Exiting.")
        return

    plot_oracle_distributions(aggregated_data, args.k_percentage, output_prefix)


if __name__ == "__main__":
    main()
