# analysis/retrieval_metrics.py
"""
This module provides functions for statistical analysis of token ranking methods,
including data loading, processing, retrieval accuracy calculations, and aggregation.
"""

import os
import math
import pickle
from typing import Dict, List, Any
from collections import defaultdict

import numpy as np
import torch


# --- DATA LOADING AND PROCESSING UTILITIES ---

def load_npz_data_for_dataset(base_path: str, model_name: str, dataset_name: str) -> Dict[str, Any]:
    """Loads and deserializes data from a single NPZ file for a given dataset."""
    file_path = os.path.join(base_path, model_name, f"{dataset_name}.npz")
    if not os.path.exists(file_path):
        return {}
    try:
        npz_file = np.load(file_path, allow_pickle=True)
        return {key: npz_file[key].item() for key in npz_file.files}
    except Exception as e:
        print(f"Warning: Could not load or process {file_path}. Error: {e}")
        return {}

def deserialize_rankings_in_sample(sample_data: Dict[str, Any]):
    """Deserializes pickled ranking data within a sample dictionary in-place."""
    for rank_type in ['fastkv_rankings', 'gemfilter_rankings', 'speculative_rankings']:
        if rank_type in sample_data and isinstance(sample_data[rank_type], bytes):
            sample_data[rank_type] = pickle.loads(sample_data[rank_type])

def get_top_k_indices(scores: torch.Tensor, k: int, max_index: int) -> torch.Tensor:
    """Safely gets the indices of the top-k scores from a tensor."""
    scores = scores[:max_index]
    if scores.numel() == 0:
        return torch.tensor([], dtype=torch.long)
    
    actual_k = min(k, scores.numel())
    if actual_k < k:
        print(f"Warning: Can only select top {actual_k} tokens, not {k}. Using {actual_k}.")
        
    _, top_k_indices = torch.topk(scores, k=actual_k)
    return top_k_indices


# --- METRIC CALCULATION ---

def calculate_oracle_overlap(
    approx_rankings: Dict, oracle_ranking: torch.Tensor, k_percentage: float
) -> Dict[Any, float]:
    """Calculates the retrieval accuracy for approximate rankings against an oracle."""
    if not approx_rankings: return {}
    prompt_len = len(oracle_ranking)
    k = max(1, math.ceil(prompt_len * k_percentage))
    _, top_k_oracle_indices = torch.topk(oracle_ranking, k=k)
    oracle_set = set(top_k_oracle_indices.tolist())
    accuracies = {}
    for key, scores_np in approx_rankings.items():
        scores_tensor = torch.from_numpy(scores_np).float()[:prompt_len]
        if scores_tensor.numel() < k: continue
        _, top_k_approx_indices = torch.topk(scores_tensor, k=k)
        approx_set = set(top_k_approx_indices.tolist())
        accuracies[key] = len(oracle_set.intersection(approx_set)) / k
    return accuracies

def get_mean_accuracies(
    dataset_name: str, all_results: Dict, k_percentage: float
) -> Dict[str, Any]:
    """Computes mean retrieval accuracies for a single dataset."""
    oracle_samples = all_results.get('oracle', {}).get(dataset_name, {})
    approx_8b = all_results.get('8B', {}).get(dataset_name, {})
    approx_1b = all_results.get('1B', {}).get(dataset_name, {})
    common_keys = sorted(list(set(oracle_samples.keys()) & set(approx_8b.keys()) & set(approx_1b.keys())))
    if not common_keys: return {}

    accs = {'fastkv': defaultdict(list), 'gemfilter': defaultdict(list), 'spec_prefill': []}
    k_priority = [8, 32, 1]

    for sample_key in common_keys:
        oracle_ranking = torch.from_numpy(oracle_samples[sample_key]['ranking']).float()
        
        # Deserialize data in-place for this sample
        deserialize_rankings_in_sample(approx_8b[sample_key])
        deserialize_rankings_in_sample(approx_1b[sample_key])

        fk_acc = calculate_oracle_overlap(approx_8b[sample_key].get('fastkv_rankings', {}), oracle_ranking, k_percentage)
        gf_acc = calculate_oracle_overlap(approx_8b[sample_key].get('gemfilter_rankings', {}), oracle_ranking, k_percentage)
        sp_acc = calculate_oracle_overlap(approx_1b[sample_key].get('speculative_rankings', {}), oracle_ranking, k_percentage)

        for k, v in fk_acc.items(): accs['fastkv'][k].append(v)
        for k, v in gf_acc.items(): accs['gemfilter'][k].append(v)
        
        chosen_k_acc = next((sp_acc[k_val] for k_val in k_priority if k_val in sp_acc), None)
        if chosen_k_acc is not None:
            accs['spec_prefill'].append(chosen_k_acc)

    mean_accs = {}
    if accs['fastkv']: mean_accs['fastkv'] = {k: np.mean(v) for k, v in accs['fastkv'].items()}
    if accs['gemfilter']: mean_accs['gemfilter'] = {k: np.mean(v) for k, v in accs['gemfilter'].items()}
    if accs['spec_prefill']: mean_accs['spec_prefill'] = np.mean(accs['spec_prefill'])
    return mean_accs

def aggregate_accuracies_by_task(all_mean_accuracies: Dict, tasks_and_datasets: Dict) -> Dict:
    """Averages per-dataset accuracies into per-task-category accuracies."""
    agg = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for task, datasets in tasks_and_datasets.items():
        for ds in datasets:
            if ds in all_mean_accuracies:
                for method, data in all_mean_accuracies[ds].items():
                    if method == 'spec_prefill':
                        agg[task][method]['values'].append(data)
                    else:
                        for k, v in data.items(): agg[task][method][k].append(v)
    final_agg = defaultdict(dict)
    for task, methods in agg.items():
        for method, data in methods.items():
            if method == 'spec_prefill':
                final_agg[task][method] = np.mean(data['values'])
            else:
                final_agg[task][method] = {k: np.mean(v) for k, v in data.items()}
    return final_agg

def find_global_max_accuracy(all_accuracies: Dict) -> float:
    """Finds the maximum accuracy value for consistent y-axis scaling."""
    max_val = 0.0
    for acc_group in all_accuracies.values():
        for method, method_accs in acc_group.items():
            if method == 'spec_prefill':
                max_val = max(max_val, method_accs)
            elif method_accs:
                max_val = max(max_val, max(method_accs.values()))
    return max_val if max_val > 0 else 1.0
