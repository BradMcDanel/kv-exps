# analysis/visualize_rankings_comparison.py

import os
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch
import math
from typing import Dict, List, Any

# Import from common visualization utility
from .viz_utils import set_publication_style

# --- Hardcoded Datasets and Names for the Publication Figure ---
# Ordered by task as shown in the paper's table.
TASK_ORDERED_DATASETS = [
    # Single-Document QA
    'narrativeqa', 'qasper', 'multifieldqa_en',
    # Multi-Document QA
    'hotpotqa', '2wikimqa', 'musique',
    # Summarization
    'gov_report', 'qmsum', 'multi_news',
    # Few-shot Learning
    'trec', 'triviaqa', 'samsum',
    # Synthetic Task
    'passage_count', 'passage_retrieval_en',
    # Code Completion
    'lcc', 'repobench-p'
]

# Map for creating publication-ready names
DATASET_NAME_MAP = {
    'narrativeqa': 'NarrativeQA',
    'qasper': 'QASPER',
    'multifieldqa_en': 'MultiFieldQA',
    'hotpotqa': 'HotpotQA',
    '2wikimqa': '2WikiMQA',
    'musique': 'Musique',
    'gov_report': 'GovReport',
    'qmsum': 'QMSum',
    'multi_news': 'Multi-News',
    'trec': 'TREC',
    'triviaqa': 'TriviaQA',
    'samsum': 'SAMSum',
    'passage_count': 'PassageCount',
    'passage_retrieval_en': 'PassageRetrieval',
    'lcc': 'LCC',
    'repobench-p': 'RepoBench-P'
}


def calculate_retrieval_accuracy(
    approx_rankings: Dict[int, np.ndarray],
    oracle_ranking: torch.Tensor,
    k_percentage: float
) -> Dict[int, float]:
    """Calculates the retrieval accuracy for each item's ranking against the oracle."""
    if not approx_rankings or k_percentage <= 0 or oracle_ranking.numel() == 0:
        return {}

    prompt_len = len(oracle_ranking)
    k = max(1, math.ceil(prompt_len * k_percentage))
    _, top_k_oracle_indices = torch.topk(oracle_ranking, k=k)
    oracle_set = set(top_k_oracle_indices.tolist())
    
    accuracies = {}
    for key, scores_np in approx_rankings.items():
        scores_tensor = torch.from_numpy(scores_np).float()
        
        if len(scores_tensor) > prompt_len:
            scores_tensor = scores_tensor[:prompt_len]

        if scores_tensor.numel() < k:
            continue

        _, top_k_approx_indices = torch.topk(scores_tensor, k=k)
        approx_set = set(top_k_approx_indices.tolist())
        
        intersection_size = len(oracle_set.intersection(approx_set))
        accuracy = intersection_size / k if k > 0 else 0.0
        accuracies[key] = accuracy
        
    return accuracies

def load_npz_data_for_dataset(base_path: str, model_name_sanitized: str, dataset_name: str) -> Dict[str, Any]:
    """Loads all samples from a single .npz file for a given dataset."""
    file_path = os.path.join(base_path, model_name_sanitized, f"{dataset_name}.npz")
    if not os.path.exists(file_path):
        print(f"Warning: Data file not found at: {file_path}. Skipping.")
        return {}
    try:
        npz_file = np.load(file_path, allow_pickle=True)
        return {key: npz_file[key].item() for key in npz_file.files}
    except Exception as e:
        print(f"Warning: Could not load or parse {file_path}. Error: {e}. Skipping.")
        return {}

def load_all_data(models_to_load: Dict, datasets_to_plot: List[str]) -> Dict:
    """Loads all data from .npz files for the required models and datasets."""
    all_data = {}
    for model_key, model_info in tqdm(models_to_load.items(), desc="Loading Models"):
        model_data_for_all_datasets = {}
        for dataset_name in datasets_to_plot:
            dataset_samples = load_npz_data_for_dataset(
                model_info['base_path'], model_info['sanitized_name'], dataset_name
            )
            if dataset_samples:
                model_data_for_all_datasets[dataset_name] = dataset_samples
        if model_data_for_all_datasets:
            all_data[model_key] = model_data_for_all_datasets
    return all_data

def get_mean_accuracies(dataset_name: str, all_results: Dict, k_percentage: float) -> Dict:
    """Helper function to calculate mean accuracies for a single dataset."""
    oracle_samples = all_results.get('oracle', {}).get(dataset_name, {})
    approx_8b_samples = all_results.get('8B', {}).get(dataset_name, {})
    approx_1b_samples = all_results.get('1B', {}).get(dataset_name, {})
    
    common_keys = sorted(list(set(oracle_samples.keys()) & 
                              set(approx_8b_samples.keys()) & 
                              set(approx_1b_samples.keys())))
    if not common_keys:
        return {'common_keys_count': 0}

    # --- Aggregate accuracies ---
    accs_8b = {'fastkv': {}, 'gemfilter': {}}
    accs_1b_spec = {}
    
    for sample_key in common_keys:
        o_data = oracle_samples[sample_key]
        oracle_ranking = torch.from_numpy(o_data['ranking']).float()
        
        # 8B data
        a_data_8b = approx_8b_samples[sample_key]
        for rank_type in ['fastkv_rankings', 'gemfilter_rankings']:
            if rank_type in a_data_8b and isinstance(a_data_8b[rank_type], bytes):
                a_data_8b[rank_type] = pickle.loads(a_data_8b[rank_type])
        
        fk_acc = calculate_retrieval_accuracy(a_data_8b.get('fastkv_rankings', {}), oracle_ranking, k_percentage)
        gf_acc = calculate_retrieval_accuracy(a_data_8b.get('gemfilter_rankings', {}), oracle_ranking, k_percentage)
        for key, acc in fk_acc.items(): accs_8b['fastkv'].setdefault(key, []).append(acc)
        for key, acc in gf_acc.items(): accs_8b['gemfilter'].setdefault(key, []).append(acc)
        
        # 1B data
        a_data_1b = approx_1b_samples[sample_key]
        if 'speculative_rankings' in a_data_1b and isinstance(a_data_1b['speculative_rankings'], bytes):
            a_data_1b['speculative_rankings'] = pickle.loads(a_data_1b['speculative_rankings'])
        sp_acc = calculate_retrieval_accuracy(a_data_1b.get('speculative_rankings', {}), oracle_ranking, k_percentage)
        for key, acc in sp_acc.items(): accs_1b_spec.setdefault(key, []).append(acc)

    # --- Calculate means ---
    results = {'common_keys_count': len(common_keys)}
    for method, data in accs_8b.items():
        if data:
            results[method] = {k: np.mean(v) for k, v in data.items()}
    
    if accs_1b_spec:
        results['speculative'] = {k: np.mean(v) for k, v in accs_1b_spec.items()}
        
    return results

def plot_grid_comparison(
    all_results: Dict, 
    k_percentage: float, 
    output_pdf_file: str, 
    output_png_file: str
):
    """Creates a high-quality, compact grid comparison plot for publication."""
    
    print("Pre-calculating accuracies for all datasets...")
    all_mean_accuracies = {
        ds: get_mean_accuracies(ds, all_results, k_percentage) for ds in TASK_ORDERED_DATASETS
    }
    
    n_rows, n_cols = 4, 4
    fig, axes = plt.subplots(n_rows, n_cols, 
                             figsize=(20, 20),
                             sharex=True, sharey=True,
                             gridspec_kw={'hspace': 0.45, 'wspace': 0.1})
    axes = axes.flatten()

    SPEC_K_POINTS = [1, 8, 32]
    # Updated labels for clarity and consistency
    styles = {
        '8B_fastkv':    {'color': '#d62728', 'linestyle': '-', 'marker': 'o', 'label': 'FastKV (8B)'},
        '8B_gemfilter':   {'color': '#ff7f0e', 'linestyle': '-', 'marker': 's', 'label': 'GemFilter (8B)'},
        '1B_spec_32':     {'color': '#2ca02c', 'linestyle': (0, (5, 2, 1, 2)), 'label': 'Speculative (1B, k=32)'},
        '1B_spec_8':      {'color': '#1f77b4', 'linestyle': '--', 'label': 'Speculative (1B, k=8)'},
        '1B_spec_1':      {'color': '#9467bd', 'linestyle': ':', 'label': 'Speculative (1B, k=1)'},
    }

    for i, dataset_name in enumerate(tqdm(TASK_ORDERED_DATASETS, desc="Plotting Grid")):
        ax = axes[i]
        dataset_accuracies = all_mean_accuracies.get(dataset_name, {})
        pretty_name = DATASET_NAME_MAP.get(dataset_name, dataset_name)
        
        if not dataset_accuracies or dataset_accuracies.get('common_keys_count', 0) == 0:
            ax.text(0.5, 0.5, f"{pretty_name}\n(No Data)", ha='center', va='center', style='italic', fontsize=16)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        if 'fastkv' in dataset_accuracies:
            sorted_keys = sorted(dataset_accuracies['fastkv'].keys())
            mean_acc = [dataset_accuracies['fastkv'][k] for k in sorted_keys]
            ax.plot(sorted_keys, mean_acc, **styles['8B_fastkv'])
        if 'gemfilter' in dataset_accuracies:
            sorted_keys = sorted(dataset_accuracies['gemfilter'].keys())
            mean_acc = [dataset_accuracies['gemfilter'][k] for k in sorted_keys]
            ax.plot(sorted_keys, mean_acc, **styles['8B_gemfilter'])

        spec_data = dataset_accuracies.get('speculative', {})
        for k_point in sorted(SPEC_K_POINTS, reverse=True): 
            if k_point in spec_data:
                mean_acc_spec = spec_data[k_point]
                style_key = f'1B_spec_{k_point}'
                ax.axhline(y=mean_acc_spec, **styles[style_key])
        
        # Removed n=... from the title
        ax.set_title(f"{pretty_name}")

    # Use larger font size for axis labels, set by rcParams
    fig.text(0.5, 0.04, 'Model Layer Index', ha='center', va='center')
    fig.text(0.04, 0.5, f'Top-{k_percentage:.0%} Retrieval Accuracy', ha='center', va='center', rotation='vertical')
    
    for ax in axes:
        ax.set_xlim(left=-1, right=32)
        ax.set_ylim(bottom=0.0, top=1.0)
        ax.set_xticks([0, 8, 16, 24, 31])

    legend_order_keys = ['8B_fastkv', '8B_gemfilter', '1B_spec_32', '1B_spec_8', '1B_spec_1']
    handles = [plt.Line2D([0], [0], **{k:v for k,v in styles[key].items() if k != 'label'}) for key in legend_order_keys]
    labels = [styles[key]['label'] for key in legend_order_keys]
    
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=3, title='Ranking Method', frameon=False)
        
    fig.suptitle(f'Comparison of Ranking Methods vs. Oracle (Top-{k_percentage:.0%} Kept)', y=0.97)
    
    plt.tight_layout(rect=[0.06, 0.08, 0.98, 0.95]) 

    plt.savefig(output_pdf_file, format='pdf', dpi=300, bbox_inches='tight')
    print(f"\nGrid plot saved to: {output_pdf_file}")
    plt.savefig(output_png_file, format='png', dpi=300, bbox_inches='tight')
    print(f"Grid plot saved to: {output_png_file}")
    
    plt.close(fig)

def main():
    # Set the publication style at the beginning
    set_publication_style()

    parser = argparse.ArgumentParser(description="Visualize and compare ranking methods.")
    MODELS = {
        'oracle': {'sanitized_name': 'meta-llama_Llama-3.1-8B-Instruct', 'base_path': 'analysis_results/oracles'},
        '8B': {'sanitized_name': 'meta-llama_Llama-3.1-8B-Instruct', 'base_path': 'analysis_results/approx_rankings'},
        '1B': {'sanitized_name': 'meta-llama_Llama-3.2-1B-Instruct', 'base_path': 'analysis_results/approx_rankings'},
    }
    parser.add_argument("--k_percentage", type=float, default=0.2, help="Percentage for top-k analysis.")
    args = parser.parse_args()
    
    if not (0 < args.k_percentage <= 1): raise ValueError("--k_percentage must be between 0 and 1.")

    output_dir = "figures"
    os.makedirs(output_dir, exist_ok=True)
    output_pdf_file = os.path.join(output_dir, "ranking_comparison.pdf")
    output_png_file = os.path.join(output_dir, "ranking_comparison.png")

    all_results = load_all_data(MODELS, TASK_ORDERED_DATASETS)
    if not all_results or 'oracle' not in all_results or '8B' not in all_results or '1B' not in all_results:
        print("Error: Missing data. Please ensure data files exist for all models.")
        return

    plot_grid_comparison(
        all_results, 
        args.k_percentage, 
        output_pdf_file, 
        output_png_file
    )

if __name__ == "__main__":
    main()
