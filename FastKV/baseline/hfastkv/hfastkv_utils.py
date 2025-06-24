import torch
import time
import torch.nn.functional as F
import torch.nn as nn
import math
import logging

def compress(model, args):
    layers = len(model.model.layers)
    tsp_schedule = {}
    if args.tsp_schedule:
        try:
            # Sort the schedule by layer index to process it in order
            schedule_items = sorted([item.split(':') for item in args.tsp_schedule.split(',')], key=lambda x: int(x[0]))
            tsp_schedule = {int(k): int(v) for k, v in schedule_items}
        except (ValueError, TypeError):
            raise ValueError("Invalid tsp_schedule format. Expected 'layer_idx1:len1,layer_idx2:len2,...'")

    for i in range(layers):
        model.model.layers[i].self_attn.kv_cluster.window_size = args.window_size
        model.model.layers[i].self_attn.kv_cluster.kernel_size = args.kernel_size
        model.model.layers[i].self_attn.kv_cluster.pooling = args.pooling
        model.model.layers[i].self_attn.kv_cluster.max_capacity_prompt = args.max_capacity_prompt
        model.model.layers[i].self_attn.kv_cluster.tsp_schedule = tsp_schedule
        # *** NEW: Explicitly tell the layer if it's a TSP layer ***
        model.model.layers[i].self_attn.kv_cluster.is_tsp_layer = i in tsp_schedule


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class HFastKVCluster():
    def __init__(self, window_size=8, max_capacity_prompt=512, kernel_size=7, pooling='avgpool', tsp_schedule=None, is_tsp_layer=False):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        if max_capacity_prompt > 0:
            assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.tsp_schedule = tsp_schedule if tsp_schedule is not None else {}
        self.is_tsp_layer = is_tsp_layer

    def reset(self, window_size=8, max_capacity_prompt=512, kernel_size=7, pooling='avgpool', tsp_schedule=None, is_tsp_layer=False):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        if max_capacity_prompt > 0:
            assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.tsp_schedule = tsp_schedule if tsp_schedule is not None else {}
        self.is_tsp_layer = is_tsp_layer

    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups, layer_idx):
        bsz, num_heads, q_len, head_dim = query_states.shape
        
        # --- Standard KV Cache Compression ---
        # This part runs regardless of TSP, to manage the KV cache size
        if q_len > self.max_capacity_prompt:
            # Note: We still calculate scores based on the incoming q_len, which is now correctly funneled.
            key_states_temp = repeat_kv(key_states, num_key_value_groups)
            # This matmul is now efficient because q_len and key_states.shape[-2] are both small after the first TSP.
            attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states_temp.transpose(2, 3)) / math.sqrt(head_dim)
            
            mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(attn_weights.device)
            attention_mask_local = mask[None, None, :, :]

            attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask_local
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights_sum = attn_weights[:, :, -self.window_size:, : -self.window_size].sum(dim=-2)

            if self.pooling == 'avgpool':
                attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size=self.kernel_size, padding=self.kernel_size//2, stride=1)
            else: # maxpool
                attn_cache = F.max_pool1d(attn_weights_sum, kernel_size=self.kernel_size, padding=self.kernel_size//2, stride=1)
            
            attn_cache = attn_cache.view(bsz, -1, num_key_value_groups, q_len - self.window_size).sum(dim=-2)
            
            # Compress for the KV cache
            num_to_keep_kv = self.max_capacity_prompt - self.window_size
            indices_kv = attn_cache.topk(num_to_keep_kv, dim=-1).indices
            indices_kv = indices_kv.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            
            k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim=2, index=indices_kv)
            v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim=2, index=indices_kv)
            k_cur = key_states[:, :, -self.window_size:, :]
            v_cur = value_states[:, :, -self.window_size:, :]
            
            key_states_compressed = torch.cat([k_past_compress, k_cur], dim=2)
            value_states_compressed = torch.cat([v_past_compress, v_cur], dim=2)
        else:
            # If not compressing, pass through the original states
            key_states_compressed = key_states
            value_states_compressed = value_states
            attn_cache = None # No scores calculated

        # --- Hierarchical Token-Selective Propagation Logic ---
        propagated_indices = None
        # *** THE FIX: Only perform TSP logic if it's a designated TSP layer ***
        if self.is_tsp_layer:
            current_tsp_len = self.tsp_schedule[layer_idx]
            
            # If scores weren't calculated for KV compression, we must calculate them now for TSP.
            if attn_cache is None:
                key_states_temp = repeat_kv(key_states, num_key_value_groups)
                attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states_temp.transpose(2, 3)) / math.sqrt(head_dim)
                attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                attn_weights_sum = attn_weights[:, :, -self.window_size:, : -self.window_size].sum(dim=-2)
                if self.pooling == 'avgpool':
                    attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size=self.kernel_size, padding=self.kernel_size//2, stride=1)
                else:
                    attn_cache = F.max_pool1d(attn_weights_sum, kernel_size=self.kernel_size, padding=self.kernel_size//2, stride=1)
                attn_cache = attn_cache.view(bsz, -1, num_key_value_groups, q_len - self.window_size).sum(dim=-2)

            num_available = q_len - self.window_size
            num_to_keep_tsp = min(current_tsp_len, num_available)

            tsp_scores = attn_cache.sum(dim=-2)
            top_k_indices = tsp_scores.topk(num_to_keep_tsp, dim=-1).indices

            window_indices = torch.arange(q_len - self.window_size, q_len, device=top_k_indices.device).unsqueeze(0).repeat(bsz, 1)
            
            propagated_indices = torch.cat([top_k_indices, window_indices], dim=-1)
            propagated_indices, _ = torch.sort(torch.unique(propagated_indices, dim=1), dim=1)

        # Return the compressed KV for this layer's cache, and the TSP indices for the next layer's funneling
        return key_states_compressed, value_states_compressed, propagated_indices

def init_hfastkv(self):
    self.kv_cluster = HFastKVCluster()
