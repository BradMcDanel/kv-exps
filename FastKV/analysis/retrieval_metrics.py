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
import pandas as pd
from scipy.stats import spearmanr

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
    for rank_type in ['fastkv_rankings', 'gemfilter_rankings', 'claa_rankings', 'speculative_rankings']:
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
    """Calculates Spearman's rank correlation between approximate rankings and oracle."""
    if not approx_rankings: return {}
    prompt_len = len(oracle_ranking)
    oracle_np = oracle_ranking.numpy() if hasattr(oracle_ranking, 'numpy') else oracle_ranking
    correlations = {}
    for key, scores_np in approx_rankings.items():
        scores_truncated = scores_np[:prompt_len]
        if len(scores_truncated) == 0: continue
        min_len = min(len(oracle_np), len(scores_truncated))
        if min_len < 2: continue  # Need at least 2 points for correlation
        corr, _ = spearmanr(oracle_np[:min_len], scores_truncated[:min_len])
        correlations[key] = corr if np.isfinite(corr) else 0.0
    return correlations

def get_mean_accuracies(
    dataset_name: str, all_results: Dict, k_percentage: float
) -> Dict[str, Any]:
    """Computes mean retrieval accuracies for a single dataset."""
    oracle_samples = all_results.get('oracle', {}).get(dataset_name, {})
    approx_8b = all_results.get('8B', {}).get(dataset_name, {})
    approx_1b = all_results.get('1B', {}).get(dataset_name, {})
    common_keys = sorted(list(set(oracle_samples.keys()) & set(approx_8b.keys()) & set(approx_1b.keys())))
    if not common_keys: return {}

    accs = {'fastkv': defaultdict(list), 'gemfilter': defaultdict(list), 'claa': defaultdict(list), 'spec_prefill': []}
    k_priority = [8, 32, 1]

    for sample_key in common_keys:
        oracle_ranking = torch.from_numpy(oracle_samples[sample_key]['ranking']).float()
        
        # Deserialize data in-place for this sample
        deserialize_rankings_in_sample(approx_8b[sample_key])
        deserialize_rankings_in_sample(approx_1b[sample_key])

        # Check if ALL ranking methods are present for this sample
        has_fastkv = 'fastkv_rankings' in approx_8b[sample_key] and approx_8b[sample_key]['fastkv_rankings']
        has_gemfilter = 'gemfilter_rankings' in approx_8b[sample_key] and approx_8b[sample_key]['gemfilter_rankings']
        has_claa = 'claa_rankings' in approx_8b[sample_key] and approx_8b[sample_key]['claa_rankings']
        has_speculative = 'speculative_rankings' in approx_1b[sample_key] and approx_1b[sample_key]['speculative_rankings']
        
        # Skip samples that don't have all ranking methods
        if not (has_fastkv and has_gemfilter and has_claa and has_speculative):
            continue

        fk_acc = calculate_oracle_overlap(approx_8b[sample_key]['fastkv_rankings'], oracle_ranking, k_percentage)
        gf_acc = calculate_oracle_overlap(approx_8b[sample_key]['gemfilter_rankings'], oracle_ranking, k_percentage)
        claa_acc = calculate_oracle_overlap(approx_8b[sample_key]['claa_rankings'], oracle_ranking, k_percentage)
        sp_acc = calculate_oracle_overlap(approx_1b[sample_key]['speculative_rankings'], oracle_ranking, k_percentage)

        for k, v in fk_acc.items(): accs['fastkv'][k].append(v)
        for k, v in gf_acc.items(): accs['gemfilter'][k].append(v)
        for k, v in claa_acc.items(): accs['claa'][k].append(v)
        
        chosen_k_acc = next((sp_acc[k_val] for k_val in k_priority if k_val in sp_acc), None)
        if chosen_k_acc is not None:
            accs['spec_prefill'].append(chosen_k_acc)

    mean_accs = {}
    if accs['fastkv']: mean_accs['fastkv'] = {k: np.mean(v) for k, v in accs['fastkv'].items()}
    if accs['gemfilter']: mean_accs['gemfilter'] = {k: np.mean(v) for k, v in accs['gemfilter'].items()}
    if accs['claa']: mean_accs['claa'] = {k: np.mean(v) for k, v in accs['claa'].items()}
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
    """Efficiently returns the set of indices for the top-k scores from a NumPy array."""
    k = min(k, ranking.size)
    if k <= 0: return set()
    indices = np.argpartition(ranking, -k)[-k:]
    return set(indices)

def calculate_jaccard_similarity(set1: Set[int], set2: Set[int]) -> float:
    """Calculates Jaccard similarity (intersection over union) between two sets."""
    if not isinstance(set1, set) or not isinstance(set2, set):
        raise TypeError("Inputs must be sets.")
    intersection_size = len(set1.intersection(set2))
    union_size = len(set1.union(set2))
    if union_size == 0:
        return 1.0
    return intersection_size / union_size

def calculate_stability_evolution_metrics( 
    oracle_ranking: np.ndarray,
    fastkv_rankings: Dict[int, np.ndarray],
    k_percentage: float, # Note: this is now only for the accuracy EVALUATION
    stability_threshold: float,
    stability_window: int = 3,
) -> Dict[str, Any]:
    """
    Calculates metrics for ranking stability and quality evolution across layers
    using Spearman's Rank Correlation with a sliding window.

    Args:
        oracle_ranking: A NumPy array of oracle scores for each token.
        fastkv_rankings: A dictionary mapping layer index to a NumPy array of FastKV scores.
        k_percentage: The percentage of top tokens to consider for ACCURACY evaluation.
        stability_threshold: The Spearman's ρ threshold for determining the early-exit layer.
        stability_window: The number of consecutive layers that must average above the threshold.

    Returns:
        A dictionary containing calculated metrics. The 'pairwise_stability' key
        now contains Spearman correlations.
    """
    if not fastkv_rankings:
        return {'valid_layers': [], 'accuracies': [], 'pairwise_stability': [], 'exit_layer': -1}

    prompt_len = len(oracle_ranking)
    k_absolute = max(1, math.ceil(prompt_len * k_percentage))
    oracle_top_k_set = get_top_k_indices_np(oracle_ranking, k_absolute)

    valid_layers = sorted(fastkv_rankings.keys())
    
    # 1. Accuracy vs. Oracle (Jaccard similarity of top-k sets) - This is for EVALUATION only.
    accuracies = [
        calculate_jaccard_similarity(
            get_top_k_indices_np(fastkv_rankings[layer], k_absolute),
            oracle_top_k_set
        ) for layer in valid_layers
    ]

    # 2. Pairwise Spearman Correlation between consecutive layers - This is the new STABILITY SIGNAL.
    pairwise_stability = []
    if len(valid_layers) >= 2:
        for i in range(1, len(valid_layers)):
            prev_layer_idx, curr_layer_idx = valid_layers[i-1], valid_layers[i]
            len_prev = len(fastkv_rankings[prev_layer_idx])
            len_curr = len(fastkv_rankings[curr_layer_idx])
            min_len = min(len_prev, len_curr)
            
            # Spearman correlation is the new stability signal
            corr, _ = spearmanr(
                fastkv_rankings[prev_layer_idx][:min_len],
                fastkv_rankings[curr_layer_idx][:min_len]
            )
            pairwise_stability.append(corr if np.isfinite(corr) else 0.0)

    # 3. Determine Early Exit Layer using a sliding window on Spearman correlation
    exit_layer = -1
    # We need at least `stability_window` correlations to make a decision
    if len(pairwise_stability) >= stability_window:
        correlations_np = np.array(pairwise_stability)
        # Create a view of the array with sliding windows
        # shape: (num_windows, window_size)
        windows = np.lib.stride_tricks.sliding_window_view(correlations_np, window_shape=stability_window)
        avg_correlations = np.mean(windows, axis=1)
        
        # Find the first window where the average exceeds the threshold
        stable_indices = np.where(avg_correlations >= stability_threshold)[0]
        if len(stable_indices) > 0:
            first_stable_window_idx = stable_indices[0]
            # The exit layer is the layer at the END of this first stable window.
            # The correlations start at valid_layers[1]. A window of size 3 at index 0
            # involves layers [1, 2, 3]. So the exit is at layer 3.
            # General formula: index_in_valid_layers = first_stable_window_idx + stability_window
            exit_layer = valid_layers[first_stable_window_idx + stability_window]
    
    return {
        'valid_layers': valid_layers,
        'accuracies': accuracies,
        'pairwise_stability': pairwise_stability, # Note the new name
        'exit_layer': exit_layer,
    }

def compute_adaptive_exit_metrics_for_dataset(
    oracle_data_for_ds: Dict,
    fastkv_data_for_ds: Dict,
    k_percentage: float,
    stability_threshold: float,
    stability_window: int,
    fixed_layer: int,
) -> Dict[str, float]:
    """
    Computes and aggregates adaptive (Spearman's ρ) vs. fixed layer metrics across all samples in a dataset.

    For each sample, this function:
    1. Calculates the adaptive exit layer based on Spearman's ρ with a sliding window.
    2. Records the retrieval accuracy at that adaptive layer.
    3. Records the retrieval accuracy at a specified `fixed_layer`.
    4. Averages these metrics across all common samples in the dataset.

    Returns:
        A dictionary with aggregated results, or an empty dict if no valid samples are found.
    """
    common_keys = sorted(list(set(oracle_data_for_ds.keys()) & set(fastkv_data_for_ds.keys())))
    if not common_keys:
        return {}

    adaptive_layers, adaptive_accuracies, fixed_accuracies = [], [], []

    for sample_key in common_keys:
        oracle_sample = oracle_data_for_ds.get(sample_key, {})
        fastkv_sample = fastkv_data_for_ds.get(sample_key, {})

        deserialize_rankings_in_sample(fastkv_sample)

        oracle_ranking = oracle_sample.get('ranking')
        fastkv_rankings = fastkv_sample.get('fastkv_rankings', {})

        if oracle_ranking is None or not fastkv_rankings:
            continue
        
        # Use our new stability metric function to get the exit layer and accuracies for this sample
        metrics = calculate_stability_evolution_metrics(
            oracle_ranking=oracle_ranking,
            fastkv_rankings=fastkv_rankings,
            k_percentage=k_percentage,
            stability_threshold=stability_threshold,
            stability_window=stability_window,
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
            continue # Skip sample if the layer isn't found

        # 2. Determine Fixed Layer Accuracy
        # Find the closest available layer to the target fixed_layer
        closest_layer = min(valid_layers, key=lambda l: abs(l - fixed_layer))
        try:
            fixed_idx = valid_layers.index(closest_layer)
            fixed_accuracies.append(per_layer_accuracies[fixed_idx])
        except (ValueError, IndexError):
            # If fixed layer calc fails, we must discard the adaptive results for this sample
            # to keep the lists parallel for accurate averaging.
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


def get_per_sample_accuracies_long_form(dataset_name: str, all_results: dict) -> pd.DataFrame:
    """
    Calculates per-sample Spearman correlations and returns them in a long-form DataFrame.
    This format is ideal for plotting with seaborn to show variance.
    """
    oracle_data = all_results.get('oracle', {}).get(dataset_name, {})
    approx_8b_data = all_results.get('8B', {}).get(dataset_name, {})
    approx_1b_data = all_results.get('1B', {}).get(dataset_name, {})

    if not oracle_data or (not approx_8b_data and not approx_1b_data):
        return pd.DataFrame()

    records = []
    
    # Get the intersection of sample keys available for all models
    common_keys = set(oracle_data.keys()) & set(approx_8b_data.keys()) & set(approx_1b_data.keys())
    
    for sample_key in common_keys:
        oracle_sample = oracle_data[sample_key]
        approx_sample_8b = approx_8b_data[sample_key]
        approx_sample_1b = approx_1b_data[sample_key]

        # Use the existing helper function to deserialize data in-place
        deserialize_rankings_in_sample(approx_sample_8b)
        deserialize_rankings_in_sample(approx_sample_1b)
        
        oracle_ranking = oracle_sample.get('ranking')
        if oracle_ranking is None: 
            continue

        # Process GemFilter and FastKV
        for method_key in ['gemfilter_rankings', 'fastkv_rankings']:
            method_name = method_key.split('_')[0].capitalize()
            rankings_dict = approx_sample_8b.get(method_key)
            
            if rankings_dict:
                for layer, approx_ranking in rankings_dict.items():
                    if oracle_ranking.shape == approx_ranking.shape:
                        corr, _ = spearmanr(oracle_ranking, approx_ranking)
                        records.append({
                            'dataset': dataset_name,
                            'sample': sample_key,
                            'method': method_name,
                            'layer': int(layer),
                            'correlation': corr if not np.isnan(corr) else 0.0
                        })

        # Process Speculative Prefill
        spec_rankings_dict = approx_sample_1b.get('speculative_rankings')
        if spec_rankings_dict:
            # We will use the k=8 result to be consistent with other analyses
            spec_ranking = spec_rankings_dict.get(8) 
            if spec_ranking is not None and oracle_ranking.shape == spec_ranking.shape:
                spec_corr, _ = spearmanr(oracle_ranking, spec_ranking)
                if not np.isnan(spec_corr):
                    for layer in range(32): # Duplicate the constant value for all layers for plotting
                         records.append({
                            'dataset': dataset_name,
                            'sample': sample_key,
                            'method': 'SpecPrefill',
                            'layer': layer,
                            'correlation': spec_corr
                        })

    return pd.DataFrame(records)
