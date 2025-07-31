import torch
import time
import torch.nn.functional as F
import torch.nn as nn
import math
import logging
import numpy as np

# Global variable to store original prompt length across all layers
_prompt_len = None
# Global variable to store precomputed oracle rankings
_oracle_rankings = None

def set_oracle_rankings(rankings):
    """Set precomputed oracle rankings globally."""
    global _oracle_rankings
    _oracle_rankings = rankings
    logging.info(f"Oracle: Set precomputed rankings with shape {rankings.shape if rankings is not None else None}")

def get_oracle_rankings():
    """Get precomputed oracle rankings."""
    global _oracle_rankings
    return _oracle_rankings

def compress(model, args): 
    layers = len(model.model.layers)
    
    logging.info(f"Oracle compress: layers={layers}, tsp_idx={args.tsp_idx}, max_capacity_prompt={args.max_capacity_prompt}, max_capacity_prompt_percentage={args.max_capacity_prompt_percentage}, tsp_len={args.tsp_len}, tsp_len_percentage={args.tsp_len_percentage}")

    for i in range(layers):
        model.model.layers[i].self_attn.kv_cluster.window_size = args.window_size
        model.model.layers[i].self_attn.kv_cluster.kernel_size = args.kernel_size
        model.model.layers[i].self_attn.kv_cluster.pooling = args.pooling
        model.model.layers[i].self_attn.kv_cluster.max_capacity_prompt = args.max_capacity_prompt
        model.model.layers[i].self_attn.kv_cluster.max_capacity_prompt_percentage = args.max_capacity_prompt_percentage
        model.model.layers[i].self_attn.kv_cluster.tsp_length = args.tsp_len
        model.model.layers[i].self_attn.kv_cluster.tsp_len_percentage = args.tsp_len_percentage
        model.model.layers[i].self_attn.kv_cluster.min_layer_idx = args.min_layer_idx
        if i == args.tsp_idx:
            model.model.layers[i].self_attn.kv_cluster.tsp_layer = True
            logging.info(f"Oracle: Set layer {i} as TSP layer")
        else:
            model.model.layers[i].self_attn.kv_cluster.tsp_layer = False

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class OracleCluster():
    def __init__(self, window_size=8, max_capacity_prompt=512, kernel_size=7, pooling='avgpool', tsp_layer=False, tsp_length=2048, max_capacity_prompt_percentage=None, tsp_len_percentage=None, min_layer_idx=0):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        self.max_capacity_prompt_percentage = max_capacity_prompt_percentage
        if self.max_capacity_prompt_percentage is None:
            assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.tsp_layer = tsp_layer
        self.tsp_length = tsp_length
        self.tsp_len_percentage = tsp_len_percentage
        self.min_layer_idx = min_layer_idx

    def reset(self, window_size=8, max_capacity_prompt=512, kernel_size=7, pooling='avgpool', tsp_layer=False, tsp_length=2048, max_capacity_prompt_percentage=None, tsp_len_percentage=None, min_layer_idx=0):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        self.max_capacity_prompt_percentage = max_capacity_prompt_percentage
        if self.max_capacity_prompt_percentage is None:
            assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.tsp_layer = tsp_layer
        self.tsp_length = tsp_length
        self.tsp_len_percentage = tsp_len_percentage
        self.min_layer_idx = min_layer_idx
        
    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups, layer_idx):
        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, _, q_len, head_dim = query_states.shape

        # Set original prompt length globally on first call
        # Reset for each new forward pass (when we hit layer 0)
        global _prompt_len
        if layer_idx == 0:
            _prompt_len = q_len
            logging.info(f"Oracle: Set global prompt length = {_prompt_len}")

        # Use global original prompt length
        self.original_prompt_len = _prompt_len

        if self.max_capacity_prompt_percentage is not None:
            max_capacity_prompt = int(self.original_prompt_len * self.max_capacity_prompt_percentage)
        else:
            max_capacity_prompt = self.max_capacity_prompt

        if self.tsp_len_percentage is not None:
            tsp_length = int(self.original_prompt_len * self.tsp_len_percentage)
        else:
            tsp_length = self.tsp_length


        if q_len < max_capacity_prompt:
            avg_indices = None
            return key_states, value_states, avg_indices
        else:
            # Use oracle rankings for KV cache compression (instead of attention heuristic)
            oracle_rankings = get_oracle_rankings()
            if oracle_rankings is None:
                raise ValueError(f"Oracle Layer {layer_idx}: Precomputed rankings are required for KV cache compression")
            if len(oracle_rankings) < q_len:
                raise ValueError(f"Oracle Layer {layer_idx}: Rankings too short ({len(oracle_rankings)} < {q_len}) for KV cache compression")
            
            # Convert oracle rankings to tensor and select top tokens for KV cache compression
            rankings_tensor = torch.tensor(oracle_rankings[:q_len], device=key_states.device, dtype=torch.float32)
            cache_indices = rankings_tensor[:-self.window_size].topk(max_capacity_prompt - self.window_size, dim=-1).indices
            
            # Expand indices for gathering from KV tensors
            cache_indices = cache_indices.unsqueeze(0).repeat(bsz, 1)  # Expand for batch
            cache_indices = cache_indices.unsqueeze(1).unsqueeze(-1).expand(-1, key_states.shape[1], -1, head_dim)
            
            # Compress KV cache using oracle-selected indices
            if layer_idx >= self.min_layer_idx:
                k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim=2, index=cache_indices)
                v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim=2, index=cache_indices)
                k_cur = key_states[:, :, -self.window_size:, :]
                v_cur = value_states[:, :, -self.window_size:, :]
                key_states = torch.cat([k_past_compress, k_cur], dim=2)
                value_states = torch.cat([v_past_compress, v_cur], dim=2)

            if self.tsp_layer and (q_len > tsp_length):
                # Use same oracle rankings for TSP sequence pruning
                tsp_indices = rankings_tensor[:-self.window_size].topk(tsp_length - self.window_size, dim=-1).indices
                tsp_indices = tsp_indices.unsqueeze(0).repeat(bsz, 1)  # Expand for batch
                window_indices = torch.arange(q_len - self.window_size, q_len, device=tsp_indices.device).unsqueeze(0).repeat(bsz,1)
                tsp_indices = torch.cat([tsp_indices, window_indices], dim=-1)
                tsp_indices, _ = torch.sort(tsp_indices, dim=1)
                logging.info(f"Oracle Layer {layer_idx}: TSP applied using precomputed rankings - selected {tsp_indices.shape[-1]} tokens")
            else:
                tsp_indices = None

            return key_states, value_states, tsp_indices

def init_oracle(self):
    self.kv_cluster = OracleCluster()
