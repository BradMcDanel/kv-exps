# baseline/taper/taper_utils.py
import torch
import time
import torch.nn.functional as F
import torch.nn as nn
import math
import logging
from typing import List

_prompt_len = None

def unpack_tsp_schedule(tsp_schedule):
    schedule_items = sorted([item.split(':') for item in tsp_schedule.split(',')], key=lambda x: int(x[0]))
    tsp_schedule = {int(k): float(v) for k, v in schedule_items}
    return tsp_schedule

def compress(model, args): 
    layers = len(model.model.layers)
    tsp_schedule = unpack_tsp_schedule(args.tsp_schedule)

    tsp_layers = [i for i in range(layers) if i in tsp_schedule]
    last_tsp_layer_idx = max(tsp_layers) if tsp_layers else -1
    model.model.last_tsp_layer_idx = last_tsp_layer_idx # Attach to model.model
    if last_tsp_layer_idx > -1:
        logging.info(f"Taper: Last TSP layer is {last_tsp_layer_idx}. Score computation will stop after this layer.")

    for i in range(layers):
        model.model.layers[i].self_attn.kv_cluster.window_size = args.window_size
        model.model.layers[i].self_attn.kv_cluster.kernel_size = args.kernel_size
        model.model.layers[i].self_attn.kv_cluster.pooling = args.pooling
        model.model.layers[i].self_attn.kv_cluster.max_capacity_prompt = args.max_capacity_prompt
        model.model.layers[i].self_attn.kv_cluster.max_capacity_prompt_percentage = args.max_capacity_prompt_percentage
        model.model.layers[i].self_attn.kv_cluster.tsp_schedule = tsp_schedule
        if i in tsp_schedule:
            model.model.layers[i].self_attn.kv_cluster.tsp_layer = True
            logging.info(f"Taper: Set layer {i} as a TSP layer")
        else:
            model.model.layers[i].self_attn.kv_cluster.tsp_layer = False

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class TaperCluster():
    def __init__(self, window_size=8, max_capacity_prompt=512, kernel_size=7, pooling='avgpool', tsp_layer=False, max_capacity_prompt_percentage=None, tsp_schedule=None):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        self.max_capacity_prompt_percentage = max_capacity_prompt_percentage
        if self.max_capacity_prompt_percentage is None:
            assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.tsp_layer = tsp_layer
        self.tsp_schedule = tsp_schedule if tsp_schedule is not None else {}

    def _aggregate_and_select_indices(self, scores_list: List[torch.Tensor], num_to_keep: int) -> torch.Tensor:
        if not scores_list:
            raise ValueError("Cannot select indices from an empty list of scores.")
        
        aggregated_scores = torch.stack(scores_list, dim=1)
        bsz, num_layers, num_heads, score_len = aggregated_scores.shape
        
        if self.pooling in ['avgpool', 'maxpool'] and self.kernel_size > 1:
            reshaped_for_pooling = aggregated_scores.view(bsz * num_layers * num_heads, 1, score_len)
            padding = (self.kernel_size - 1) // 2
            pool_fn = F.avg_pool1d if self.pooling == 'avgpool' else F.max_pool1d
            pooled_tensor = pool_fn(reshaped_for_pooling, kernel_size=self.kernel_size, stride=1, padding=padding)
            processed_scores = pooled_tensor.view(bsz, num_layers, num_heads, score_len)
        else:
            processed_scores = aggregated_scores
        
        max_scores_across_heads, _ = processed_scores.max(dim=2)
        final_token_importance, _ = max_scores_across_heads.max(dim=1)
        
        num_to_keep = min(num_to_keep, final_token_importance.shape[-1])
        _, top_indices = torch.topk(final_token_importance, k=num_to_keep, dim=-1)
        return torch.sort(top_indices, dim=-1)[0]

    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups, layer_idx, **kwargs):
        bsz, _, q_len, head_dim = query_states.shape
        _, num_kv_heads, _, _ = key_states.shape
        
        global _prompt_len
        if layer_idx == 0:
            _prompt_len = q_len
            logging.info(f"Taper: Set global prompt length = {_prompt_len}")
            
        self.original_prompt_len = _prompt_len

        if self.max_capacity_prompt_percentage is not None:
            max_capacity_prompt = int(self.original_prompt_len * self.max_capacity_prompt_percentage)
        else:
            max_capacity_prompt = self.max_capacity_prompt

        is_score_collection_needed = kwargs.get("is_score_collection_needed", True)
        taper_collected_scores = kwargs["taper_collected_scores"]
        
        if is_score_collection_needed and q_len > self.window_size:
            key_states_for_scores = repeat_kv(key_states, num_key_value_groups)
            attn_logits = torch.matmul(query_states[..., -self.window_size:, :], key_states_for_scores[..., :-self.window_size, :].transpose(2, 3)) / math.sqrt(head_dim)
            attn_probs = F.softmax(attn_logits, dim=-1, dtype=torch.float32).to(query_states.dtype)
            scores_for_this_layer = attn_probs.sum(dim=-2)
            taper_collected_scores.append(scores_for_this_layer)
        
        if self.tsp_layer and q_len > max_capacity_prompt:
            logging.info(f"Taper Layer {layer_idx}: Applying TSP. Original seq len: {q_len}, Max capacity: {max_capacity_prompt}")
            tsp_keep_ratio = self.tsp_schedule[layer_idx]
            tsp_length = int(q_len * tsp_keep_ratio)
            num_to_keep = max(tsp_length, max_capacity_prompt) - self.window_size

            if num_to_keep <= 0 or not taper_collected_scores:
                taper_collected_scores.clear()
                return key_states, value_states, None
            
            indices_to_keep_past = self._aggregate_and_select_indices(taper_collected_scores, num_to_keep)
            taper_collected_scores.clear()
            
            past_indices = indices_to_keep_past.unsqueeze(1).unsqueeze(-1).expand(bsz, num_kv_heads, num_to_keep, head_dim)
            k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim=2, index=past_indices)
            v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim=2, index=past_indices)
            k_cur = key_states[:, :, -self.window_size:, :]
            v_cur = value_states[:, :, -self.window_size:, :]
            key_states = torch.cat([k_past_compress, k_cur], dim=2)
            value_states = torch.cat([v_past_compress, v_cur], dim=2)

            window_indices = torch.arange(q_len - self.window_size, q_len, device=indices_to_keep_past.device)
            window_indices = window_indices.unsqueeze(0).expand(bsz, -1)
            tsp_indices = torch.cat([indices_to_keep_past, window_indices], dim=1)
            tsp_indices = torch.sort(tsp_indices, dim=1)[0]
            
            logging.info(f"Taper Layer {layer_idx}: Pruned. New K/V seq len: {key_states.shape[-2]}")
            return key_states, value_states, tsp_indices
        else:
            return key_states, value_states, None

def init_taper(self):
    self.kv_cluster = TaperCluster()
