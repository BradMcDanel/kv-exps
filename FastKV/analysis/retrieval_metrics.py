# analysis/retrieval_metrics.py
"""
This module provides functions for statistical analysis of token ranking methods,
including data loading, processing, retrieval accuracy calculations, and aggregation.
"""

import os
import math
import pickle
from typing import Dict, List, Any, Set
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


# --- STABILITY EVOLUTION METRICS ---

def get_top_k_indices_np(ranking: np.ndarray, k: int) -> Set[int]:
    """
    Efficiently returns the set of indices for the top-k scores from a NumPy array.
    Uses np.argpartition for performance.
    """
    k = min(k, ranking.size)
    if k <= 0:
        return set()
    
    # argpartition is faster than argsort for finding top-k. It partitions the
    # array such that the k-th largest element is in its sorted place.
    indices = np.argpartition(ranking, -k)[-k:]
    return set(indices)

def calculate_jaccard_similarity(set1: Set[int], set2: Set[int]) -> float:
    """Calculates Jaccard similarity (intersection over union) between two sets."""
    if not isinstance(set1, set) or not isinstance(set2, set):
        raise TypeError("Inputs must be sets.")
        
    intersection_size = len(set1.intersection(set2))
    union_size = len(set1.union(set2))
    
    if union_size == 0:
        return 1.0  # Jaccard of two empty sets is 1.0
        
    return intersection_size / union_size

def calculate_stability_evolution_metrics(
    oracle_ranking: np.ndarray,
    fastkv_rankings: Dict[int, np.ndarray],
    k_percentage: float,
    stability_threshold: float
) -> Dict[str, Any]:
    """
    Calculates metrics for ranking stability and quality evolution across layers.

    Args:
        oracle_ranking: A NumPy array of oracle scores for each token.
        fastkv_rankings: A dictionary mapping layer index to a NumPy array of FastKV scores.
        k_percentage: The percentage of top tokens to consider for accuracy and stability.
        stability_threshold: The Jaccard similarity threshold for determining the early-exit layer.

    Returns:
        A dictionary containing calculated metrics:
        - 'valid_layers': Sorted list of layers with rankings.
        - 'accuracies': List of Jaccard similarities against the oracle for each layer.
        - 'pairwise_jaccard': List of Jaccard similarities between consecutive layers.
        - 'exit_layer': The layer index where stability threshold was first met or exceeded.
        - 'k_absolute': The absolute number of top tokens considered.
    """
    if not fastkv_rankings:
        return {
            'valid_layers': [], 'accuracies': [], 'pairwise_jaccard': [], 
            'exit_layer': -1, 'k_absolute': 0
        }

    prompt_len = len(oracle_ranking)
    k_absolute = max(1, math.ceil(prompt_len * k_percentage))

    oracle_top_k_set = get_top_k_indices_np(oracle_ranking, k_absolute)
    valid_layers = sorted(fastkv_rankings.keys())
        
    # 1. Accuracy vs. Oracle (Jaccard similarity of top-k sets)
    accuracies = [
        calculate_jaccard_similarity(
            get_top_k_indices_np(fastkv_rankings[layer], k_absolute), 
            oracle_top_k_set
        ) for layer in valid_layers
    ]

    # 2. Pairwise Top-K Jaccard Similarity between consecutive layers
    pairwise_jaccard = []
    if len(valid_layers) >= 2:
        for i in range(1, len(valid_layers)):
            prev_layer_idx, curr_layer_idx = valid_layers[i-1], valid_layers[i]
            prev_top_k = get_top_k_indices_np(fastkv_rankings[prev_layer_idx], k_absolute)
            curr_top_k = get_top_k_indices_np(fastkv_rankings[curr_layer_idx], k_absolute)
            similarity = calculate_jaccard_similarity(prev_top_k, curr_top_k)
            pairwise_jaccard.append(similarity)

    # 3. Determine Early Exit Layer
    exit_layer = -1
    if pairwise_jaccard:
        for i, similarity in enumerate(pairwise_jaccard):
            if similarity >= stability_threshold:
                # Exit layer is the current layer in the pair (i.e., the second one)
                exit_layer = valid_layers[i+1] 
                break
    
    return {
        'valid_layers': valid_layers,
        'accuracies': accuracies,
        'pairwise_jaccard': pairwise_jaccard,
        'exit_layer': exit_layer,
        'k_absolute': k_absolute
    }

def compute_adaptive_exit_metrics_for_dataset(
    oracle_data_for_ds: Dict,
    fastkv_data_for_ds: Dict,
    k_percentage: float,
    stability_threshold: float,
    fixed_layer: int,
) -> Dict[str, float]:
    """
    Computes and aggregates adaptive vs. fixed layer metrics across all samples in a dataset.

    For each sample, this function:
    1. Calculates the adaptive exit layer based on stability (`stability_threshold`).
    2. Records the retrieval accuracy at that adaptive layer.
    3. Records the retrieval accuracy at a specified `fixed_layer`.
    4. Averages these metrics across all common samples in the dataset.

    Returns:
        A dictionary with aggregated results, or an empty dict if no valid samples are found.
        Example: {'avg_adaptive_layer': 12.3, 'avg_adaptive_accuracy': 0.85, 'avg_fixed_accuracy': 0.88}
    """
    common_keys = sorted(list(set(oracle_data_for_ds.keys()) & set(fastkv_data_for_ds.keys())))
    if not common_keys:
        return {}

    adaptive_layers, adaptive_accuracies, fixed_accuracies = [], [], []

    for sample_key in common_keys:
        oracle_sample = oracle_data_for_ds[sample_key]
        fastkv_sample = fastkv_data_for_ds[sample_key]

        # Ensure rankings data is deserialized from bytes if needed
        deserialize_rankings_in_sample(fastkv_sample)

        oracle_ranking = oracle_sample.get('ranking')
        fastkv_rankings = fastkv_sample.get('fastkv_rankings', {})

        if oracle_ranking is None or not fastkv_rankings:
            continue
        
        # Calculate evolution metrics for this single sample
        metrics = calculate_stability_evolution_metrics(
            oracle_ranking=oracle_ranking,
            fastkv_rankings=fastkv_rankings,
            k_percentage=k_percentage,
            stability_threshold=stability_threshold,
        )

        valid_layers = metrics['valid_layers']
        per_layer_accuracies = metrics['accuracies']
        if not valid_layers or not per_layer_accuracies:
            continue

        # 1. Determine Adaptive Exit Layer and its Accuracy
        exit_layer = metrics['exit_layer']
        # Fallback: if stability is never reached, use the last available layer
        layer_to_use = exit_layer if exit_layer != -1 else valid_layers[-1]
        
        try:
            adaptive_idx = valid_layers.index(layer_to_use)
            adaptive_layers.append(layer_to_use)
            adaptive_accuracies.append(per_layer_accuracies[adaptive_idx])
        except (ValueError, IndexError):
            continue # Skip sample if the layer isn't found for some reason

        # 2. Determine Fixed Layer Accuracy
        # Find the closest available layer to the target fixed_layer
        closest_layer = min(valid_layers, key=lambda l: abs(l - fixed_layer))
        try:
            fixed_idx = valid_layers.index(closest_layer)
            fixed_accuracies.append(per_layer_accuracies[fixed_idx])
        except (ValueError, IndexError):
            # This can happen if adaptive logic ran but fixed didn't have a match
            # To keep lists aligned, we might skip or use a nan. Skipping is safer.
            adaptive_layers.pop()
            adaptive_accuracies.pop()
            continue
            
    if not adaptive_accuracies: # Check if any samples were successfully processed
        return {}
        
    return {
        'avg_adaptive_layer': np.mean(adaptive_layers),
        'avg_adaptive_accuracy': np.mean(adaptive_accuracies),
        'avg_fixed_accuracy': np.mean(fixed_accuracies),
    }

