import torch
import time
import torch.nn.functional as F
import torch.nn as nn
import math
import logging

_prompt_len = None

def unpack_tsp_schedule(tsp_schedule):
    schedule_items = sorted([item.split(':') for item in tsp_schedule.split(',')], key=lambda x: int(x[0]))
    tsp_schedule = {int(k): float(v) for k, v in schedule_items}
    return tsp_schedule

def compress(model, args): 
    layers = len(model.model.layers)
    tsp_schedule = unpack_tsp_schedule(args.tsp_schedule)

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
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
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
        if tsp_schedule is None:
            tsp_schedule = {}

    def reset(self, window_size=8, max_capacity_prompt=512, kernel_size=7, pooling='avgpool', tsp_layer=False, max_capacity_prompt_percentage=None, tsp_schedule=None):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        self.max_capacity_prompt_percentage = max_capacity_prompt_percentage
        if self.max_capacity_prompt_percentage is None:
            assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.tsp_layer = tsp_layer
        if tsp_schedule is None:
            tsp_schedule = {}
        
    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups, layer_idx):
        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, _, q_len, head_dim = query_states.shape

        global _prompt_len
        if layer_idx == 0:
            _prompt_len = q_len
            logging.info(f"TAPER: Set global prompt length = {_prompt_len}")

        # Use global original prompt length
        self.original_prompt_len = _prompt_len

        # IMPORTANT: Always use original prompt length for percentage calculation, not current q_len
        if self.max_capacity_prompt_percentage is not None:
            max_capacity_prompt = int(self.original_prompt_len * self.max_capacity_prompt_percentage)
        else:
            max_capacity_prompt = self.max_capacity_prompt

        if self.tsp_layer:
            tsp_length = int(self.tsp_schedule[layer_idx] * _prompt_len)

        if q_len < max_capacity_prompt:
            avg_indices = None
            return key_states, value_states, avg_indices
        else:
            key_states_temp = repeat_kv(key_states, num_key_value_groups)
            attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states_temp.transpose(2, 3)) / math.sqrt(head_dim)
            mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(attn_weights.device)
            attention_mask = mask[None, None, :, :]

            attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights_sum = attn_weights[:, :, -self.window_size:, : -self.window_size].sum(dim = -2)
            if self.pooling == 'avgpool':
                attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            elif self.pooling == 'maxpool':
                attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            else:
                raise ValueError('Pooling method not supported')

            attn_cache = attn_cache.view(bsz, -1, num_key_value_groups, q_len-self.window_size).sum(dim=-2)
            indices = attn_cache.topk(max_capacity_prompt - self.window_size, dim=-1).indices
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            
            k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            k_cur = key_states[:, :, -self.window_size:, :]
            v_cur = value_states[:, :, -self.window_size:, :]
            key_states = torch.cat([k_past_compress, k_cur], dim = 2)
            value_states = torch.cat([v_past_compress, v_cur], dim = 2)

            if self.tsp_layer and (q_len > tsp_length):
                tsp_indices = attn_cache.sum(dim=-2).topk(tsp_length - self.window_size, dim=-1).indices
                window_indices = torch.arange(q_len - self.window_size, q_len, device=tsp_indices.device).unsqueeze(0).repeat(bsz,1)
                tsp_indices = torch.cat([tsp_indices, window_indices], dim=-1)
                tsp_indices, _ = torch.sort(tsp_indices, dim=1)
                logging.info(f"TAPER Layer {layer_idx}: TSP applied - selected {tsp_indices.shape[-1]} tokens")
            else:
                tsp_indices = None

            return key_states, value_states, tsp_indices

def init_taper(self):
    self.kv_cluster = TaperCluster()
