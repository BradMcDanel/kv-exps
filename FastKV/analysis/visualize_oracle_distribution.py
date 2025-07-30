# analysis/visualize_oracle_distribution.py

import argparse
import os
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

# Assuming these utilities are in a sibling file or an installed package.
# Make sure TASKS_AND_DATASETS and DATASET_NAME_MAP are correctly defined in viz_utils.
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
    'narrativeqa', 'qasper', 'multifieldqa_en', 'hotpotqa', '2wikimqa', 'musique',
    'gov_report', 'qmsum', 'multi_news', 'trec', 'triviaqa', 'samsum',
    'passage_count', 'passage_retrieval_en', 'lcc', 'repobench-p'
]


def generate_dummy_data() -> Dict[str, np.ndarray]:
    """
    Generates plausible-looking dummy data for each task category to allow
    for rapid layout and visualization debugging. The patterns are designed to
    be distinct and representative of expected real-world distributions.
    """
    print("--- Using --debug mode: Generating synthetic data ---")
    aggregated_data = {}
    num_tokens = 200000  # Total number of important tokens to simulate per category

    # --- Define plausible patterns for each task type ---
    patterns = {
        'Single-Doc QA': [
            {'mean': 0.5, 'std': 0.2, 'weight': 1.0} # Centered
        ],
        'Multi-Doc QA': [
            {'mean': 0.2, 'std': 0.05, 'weight': 0.3},
            {'mean': 0.5, 'std': 0.05, 'weight': 0.4},
            {'mean': 0.8, 'std': 0.05, 'weight': 0.3} # 3 distinct peaks
        ],
        'Summarization': [
            {'mean': 0.1, 'std': 0.08, 'weight': 0.5},
            {'mean': 0.9, 'std': 0.08, 'weight': 0.5} # Classic U-shape
        ],
        'Few-shot Learning': [
            {'mean': 0.05, 'std': 0.04, 'weight': 1.0} # Heavily front-loaded
        ],
        'Code Completion': [
            {'mean': 0.5, 'std': 0.25, 'weight': 0.4},
            {'mean': 0.95, 'std': 0.05, 'weight': 0.6} # Context from middle and end
        ],
        'Synthetic Task': [
            # Uniform distribution
            {'dist': 'uniform', 'weight': 1.0}
        ],
    }

    # Iterate using the official task names to ensure consistency
    for task_name in TASKS_AND_DATASETS.keys():
        if task_name in patterns:
            task_patterns = patterns[task_name]
            indices = []
            for p in task_patterns:
                count = int(num_tokens * p['weight'])
                if p.get('dist') == 'uniform':
                    indices.append(np.random.uniform(low=0.0, high=1.0, size=count))
                else: # Default to normal distribution
                    indices.append(np.random.normal(loc=p['mean'], scale=p['std'], size=count))
            aggregated_data[task_name] = np.clip(np.concatenate(indices), 0, 1)
        else: # Add a default uniform pattern for any missing task categories
            aggregated_data[task_name] = np.random.uniform(low=0.0, high=1.0, size=num_tokens)


    return aggregated_data


def aggregate_oracle_indices_by_task(k_percentage: float, model_config: Dict) -> Dict[str, np.ndarray]:
    """
    Loads oracle data for all datasets, computes top-k token positions,
    and aggregates them by their task category. (REAL DATA)
    """
    aggregated_indices_by_task: Dict[str, List[np.ndarray]] = {task_name: [] for task_name in TASKS_AND_DATASETS}

    pbar = tqdm(ALL_LONG_BENCH_DATASETS, desc="Loading and Processing Datasets")
    for dataset_id in pbar:
        pbar.set_postfix_str(f"Processing {DATASET_NAME_MAP.get(dataset_id, dataset_id)}")

        current_task_category = next((task_cat for task_cat, datasets in TASKS_AND_DATASETS.items() if dataset_id in datasets), None)
        if not current_task_category: continue

        oracle_data = load_npz_data_for_dataset(model_config['base_path'], model_config['sanitized_name'], dataset_id)
        if not oracle_data:
            tqdm.write(f"Warning: No oracle data found for {dataset_id}")
            continue

        for sample in oracle_data.values():
            if 'ranking' not in sample or len(sample['ranking']) == 0: continue
            ranking = torch.from_numpy(sample['ranking']).float()
            seq_len = len(ranking)
            if seq_len < 2: continue

            k_absolute = max(1, int(seq_len * k_percentage))
            top_indices = get_top_k_indices(ranking, k_absolute, seq_len)
            normalized_indices = top_indices.float() / (seq_len - 1)
            aggregated_indices_by_task[current_task_category].append(normalized_indices.cpu().numpy())

    final_aggregated_data = {
        task: np.concatenate(indices) for task, indices in aggregated_indices_by_task.items() if indices
    }
    return final_aggregated_data


def plot_binned_distributions(aggregated_data: Dict[str, np.ndarray], k_percentage: float, output_file_prefix: str, num_bins: int = 40):
    """
    Generates and saves a 2x3 grid plot of binned positional histograms.
    """
    set_publication_style()
    plt.rcParams.update({'axes.labelsize': 22, 'xtick.labelsize': 18, 'ytick.labelsize': 18, 'axes.titlesize': 24})

    n_rows, n_cols = 2, 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, 11), sharey=True) # sharey=True for easier comparison
    fig.suptitle(f'Aggregated Positional Distribution of Top-{int(k_percentage*100)}% Oracle-Ranked Tokens', fontsize=28, weight='bold')

    bins = np.linspace(0, 1, num_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = bins[1] - bins[0]

    task_order = list(TASKS_AND_DATASETS.keys())
    max_y_val = 0

    # First pass to find the max y-value for consistent scaling
    for task_name in task_order:
        indices = aggregated_data.get(task_name)
        if indices is not None and len(indices) > 0:
            counts, _ = np.histogram(indices, bins=bins)
            percentages = (counts / counts.sum()) * 100 if counts.sum() > 0 else counts
            if percentages.max() > max_y_val:
                max_y_val = percentages.max()

    for i, ax in enumerate(axes.flat):
        if i >= len(task_order):
            ax.axis('off')
            continue

        task_name = task_order[i]
        ax.set_title(task_name)

        indices = aggregated_data.get(task_name)
        if indices is None or len(indices) == 0:
            ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
            continue

        counts, _ = np.histogram(indices, bins=bins)
        percentages = (counts / counts.sum()) * 100 if counts.sum() > 0 else counts

        ax.bar(bin_centers, percentages, width=bin_width * 0.9, color='#0077B6', edgecolor='black', linewidth=0.5, alpha=0.8)

        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(0, max_y_val * 1.1)
        ax.grid(True, which='major', linestyle=':', linewidth=0.7, axis='y')
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
        if i % n_cols != 0:
            ax.set_yticklabels([]) # Hide y-labels for inner plots

    fig.text(0.5, 0.02, 'Normalized Token Position in Prompt', ha='center', va='center', fontsize=26, weight='bold')
    fig.text(0.06, 0.5, 'Share of Important Tokens (%)', ha='center', va='center', rotation='vertical', fontsize=26, weight='bold')

    plt.tight_layout(rect=[0.08, 0.05, 1, 0.95])
    
    output_filename = f"{output_file_prefix}_binned_debug" if 'debug' in str(aggregated_data) else f"{output_file_prefix}_binned"
    output_pdf = f"{output_filename}.pdf"
    output_png = f"{output_filename}.png"

    plt.savefig(output_pdf, dpi=300)
    print(f"\nSaved binned distribution plot to: {output_pdf}")
    plt.savefig(output_png, dpi=300)
    print(f"Saved binned distribution plot to: {output_png}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize oracle token importance distribution across task categories.")
    parser.add_argument("--k_percentage", type=float, default=0.1, help="Top-k percentage of tokens to consider.")
    parser.add_argument("--bins", type=int, default=40, help="Number of bins for the histogram.")
    parser.add_argument("--debug", action="store_true", help="Use synthetic dummy data for fast layout testing.")
    args = parser.parse_args()

    output_dir = "figures"
    os.makedirs(output_dir, exist_ok=True)
    output_prefix = os.path.join(output_dir, "oracle_token_distribution_by_task")

    if args.debug:
        # Generate dummy data for a quick plot preview
        aggregated_data = generate_dummy_data()
        # A little trick to help the saving function know it was a debug run
        aggregated_data['__name__'] = 'debug' 
    else:
        # Load and process the real oracle data (this can be slow)
        model_config = {
            'sanitized_name': 'meta-llama_Llama-3.1-8B-Instruct',
            'base_path': 'analysis_results/oracles'
        }
        aggregated_data = aggregate_oracle_indices_by_task(args.k_percentage, model_config)

    if not aggregated_data:
        print("No data was aggregated. Exiting.")
        return

    plot_binned_distributions(aggregated_data, args.k_percentage, output_prefix, num_bins=args.bins)


if __name__ == "__main__":
    main()
