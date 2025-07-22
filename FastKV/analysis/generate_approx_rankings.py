import argparse
import json
import math
import os
import pickle
import types
import zipfile
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          set_seed)
from transformers.cache_utils import Cache, DynamicCache
from transformers.models.llama.modeling_llama import (
    LlamaAttention, apply_rotary_pos_emb as hf_apply_rotary_pos_emb,
    repeat_kv as hf_repeat_kv)


# --- Base Class ---

class BaseRankingGenerator:
    """A base class for generating token rankings using different strategies."""
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, args: argparse.Namespace):
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.config = self.model.config
        self.device = self.model.device
        self.orig_fwds: Dict[int, Any] = {}

    def generate_rankings(self, inputs: Dict[str, torch.Tensor]) -> Dict:
        raise NotImplementedError

    def _unpatch_model(self):
        for i, layer in enumerate(self.model.model.layers):
            if hasattr(layer, 'self_attn') and i in self.orig_fwds:
                layer.self_attn.forward = self.orig_fwds[i]
        self.orig_fwds.clear()

# --- Strategy 1: FastKV ---

def _patched_attention_forward_fastkv(
    self_attn: LlamaAttention,
    generator_obj: 'FastKVRankingGenerator',
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
    print(f"DEBUG FastKV: Entered patched attention forward, layer {getattr(self_attn, 'layer_idx', 'unknown')}")
    batch_size, query_length, _ = hidden_states.size()
    print(f"DEBUG FastKV: batch_size={batch_size}, query_length={query_length}")
    num_heads = self_attn.config.num_attention_heads
    head_dim = self_attn.config.hidden_size // num_heads
    num_key_value_heads = self_attn.config.num_key_value_heads
    hidden_size = self_attn.config.hidden_size
    num_key_value_groups = num_heads // num_key_value_heads
    
    query_projection = self_attn.q_proj(hidden_states)
    key_projection = self_attn.k_proj(hidden_states)
    value_projection = self_attn.v_proj(hidden_states)
    
    query_states = query_projection.view(batch_size, query_length, num_heads, head_dim).transpose(1, 2)
    key_states_before_rope = key_projection.view(batch_size, query_length, num_key_value_heads, head_dim).transpose(1, 2)
    value_states_for_cache = value_projection.view(batch_size, query_length, num_key_value_heads, head_dim).transpose(1, 2)
    
    cos, sin = self_attn.rotary_emb(value_states_for_cache, position_ids=position_ids)
    query_states_rotated, key_states_rotated = hf_apply_rotary_pos_emb(query_states, key_states_before_rope, cos, sin, position_ids)

    key_states_for_sdpa, value_states_for_sdpa = past_key_value.update(
        key_states_rotated, value_states_for_cache, self_attn.layer_idx, {"cache_position": cache_position}
    )
    key_states_for_sdpa = hf_repeat_kv(key_states_for_sdpa, num_key_value_groups)
    value_states_for_sdpa = hf_repeat_kv(value_states_for_sdpa, num_key_value_groups)
    
    # Compute attention weights for FastKV TSP layer ranking
    if generator_obj.tsp_layer_idx == self_attn.layer_idx and query_length > generator_obj.window_size:
        # Compute attention scores similar to FastKV's update_kv method
        attn_weights = torch.matmul(query_states_rotated[..., -generator_obj.window_size:, :], key_states_for_sdpa.transpose(-1, -2)) / math.sqrt(head_dim)
        
        # Apply causal mask for the window
        mask = torch.full((generator_obj.window_size, generator_obj.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        attn_weights[:, :, -generator_obj.window_size:, -generator_obj.window_size:] += mask[None, None, :, :]
        
        # Softmax and sum over query positions (last window_size tokens)
        attn_weights_softmax = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states_rotated.dtype)
        attn_weights_sum = attn_weights_softmax[:, :, -generator_obj.window_size:, :-generator_obj.window_size].sum(dim=-2)
        
        # Apply pooling as in FastKV
        if generator_obj.pooling == 'avgpool':
            attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size=generator_obj.kernel_size, padding=generator_obj.kernel_size//2, stride=1)
        elif generator_obj.pooling == 'maxpool':
            attn_cache = F.max_pool1d(attn_weights_sum, kernel_size=generator_obj.kernel_size, padding=generator_obj.kernel_size//2, stride=1)
        else:
            attn_cache = attn_weights_sum
            
        # Sum across heads to get final importance scores
        if len(attn_cache.shape) == 3:  # [batch, heads, seq]
            final_scores = attn_cache.sum(dim=1)  # Sum over heads
            generator_obj.tsp_scores = final_scores.squeeze(0)  # Remove batch dimension
        else:
            generator_obj.tsp_scores = attn_cache.squeeze(0)
    
    attn_mask_input = attention_mask
    is_sdpa_causal = (query_length > 1) and (attn_mask_input is None)
    if attn_mask_input is not None:
        is_sdpa_causal = False
        actual_kv_sequence_length = key_states_for_sdpa.shape[-2]
        if attn_mask_input.shape[-1] > actual_kv_sequence_length:
            attn_mask_input = attn_mask_input[:, :, :, :actual_kv_sequence_length]
            
    attention_output = F.scaled_dot_product_attention(
        query_states_rotated, key_states_for_sdpa, value_states_for_sdpa, attn_mask=attn_mask_input, dropout_p=0.0, is_causal=is_sdpa_causal
    )
    attention_output = attention_output.transpose(1, 2).contiguous().reshape(batch_size, query_length, hidden_size)
    attention_output = self_attn.o_proj(attention_output)
    
    return attention_output, None, past_key_value


class FastKVRankingGenerator(BaseRankingGenerator):
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, args: argparse.Namespace):
        super().__init__(model, tokenizer, args)
        self.window_size = args.window_size
        self.kernel_size = args.kernel_size
        self.pooling = args.pooling
        self.tsp_layer_idx = 0  # Default to first layer, can be configurable
        self.tsp_scores = None
        
    def _patch_model(self):
        # Only patch the TSP layer
        layer = self.model.model.layers[self.tsp_layer_idx]
        if hasattr(layer, 'self_attn') and isinstance(layer.self_attn, LlamaAttention):
            attn_module = layer.self_attn
            if self.tsp_layer_idx not in self.orig_fwds:
                self.orig_fwds[self.tsp_layer_idx] = attn_module.forward
            setattr(attn_module, "layer_idx", self.tsp_layer_idx)
            attn_module.forward = types.MethodType(partial(_patched_attention_forward_fastkv, generator_obj=self), attn_module)

    @torch.no_grad()
    def generate_rankings(self, inputs: Dict[str, torch.Tensor]) -> Dict[int, torch.Tensor]:
        self.tsp_scores = None
        self._patch_model()
        
        try:
            input_ids = inputs['input_ids']
            prompt_length = input_ids.shape[1]
            
            # Run forward pass to compute TSP scores
            position_ids = torch.arange(prompt_length, device=self.device).unsqueeze(0)  # Add batch dimension
            with torch.no_grad():
                _ = self.model(input_ids=input_ids, position_ids=position_ids, use_cache=True)
            
            # Return the TSP scores if computed
            if self.tsp_scores is not None:
                # For FastKV, we simulate different layers being the TSP layer
                rankings = {}
                for layer_idx in range(self.config.num_hidden_layers):
                    rankings[layer_idx] = self.tsp_scores.clone()
                return rankings
            else:
                return {}
                
        finally:
            self._unpatch_model()
            # Clear GPU cache to free memory for next generator
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

# --- Strategy 2: GemFilter ---

def _patched_attention_forward_gemfilter(
    self_attn: LlamaAttention,
    generator_obj: 'GemFilterRankingGenerator',
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
    batch_size, query_length, _ = hidden_states.size()
    num_heads = self_attn.config.num_attention_heads
    head_dim = self_attn.config.hidden_size // num_heads
    num_key_value_heads = self_attn.config.num_key_value_heads
    hidden_size = self_attn.config.hidden_size
    num_key_value_groups = num_heads // num_key_value_heads
    
    query_projection = self_attn.q_proj(hidden_states)
    key_projection = self_attn.k_proj(hidden_states)
    value_projection = self_attn.v_proj(hidden_states)
    
    query_states = query_projection.view(batch_size, query_length, num_heads, head_dim).transpose(1, 2)
    key_states_before_rope = key_projection.view(batch_size, query_length, num_key_value_heads, head_dim).transpose(1, 2)
    value_states_for_cache = value_projection.view(batch_size, query_length, num_key_value_heads, head_dim).transpose(1, 2)
    
    cos, sin = self_attn.rotary_emb(value_states_for_cache, position_ids=position_ids)
    query_states_rotated, key_states_rotated = hf_apply_rotary_pos_emb(query_states, key_states_before_rope, cos, sin, position_ids)

    key_states_for_sdpa, value_states_for_sdpa = past_key_value.update(
        key_states_rotated, value_states_for_cache, self_attn.layer_idx, {"cache_position": cache_position}
    )
    key_states_for_sdpa = hf_repeat_kv(key_states_for_sdpa, num_key_value_groups)
    value_states_for_sdpa = hf_repeat_kv(value_states_for_sdpa, num_key_value_groups)
    
    # Compute GemFilter ranking scores
    if generator_obj.select_layer_idx == self_attn.layer_idx and query_length >= 1:
        # GemFilter uses the last query token to compute similarity with all keys
        query_last_states = query_states_rotated[:, :, -1:, :]  # [batch, heads, 1, head_dim]
        key_states_repeat = key_states_for_sdpa  # Already repeated
        
        # Compute inner product (similarity) between last query and all keys
        inner_product = torch.matmul(query_last_states, key_states_repeat.transpose(-1, -2))
        inner_product = inner_product[:, :, 0, :]  # Remove the singleton query dimension
        
        # Sum over heads as in GemFilter's standard_dis_index
        inner_product_summed = torch.sum(inner_product, dim=1, keepdim=True)
        
        # Apply pooling as in GemFilter (avgpool with kernel_size)
        pooled_scores = F.avg_pool1d(
            inner_product_summed, 
            kernel_size=generator_obj.kernel_size, 
            padding=generator_obj.kernel_size//2, 
            stride=1
        )
        
        # Store the importance scores (remove batch dimension)
        generator_obj.gemfilter_scores = pooled_scores.squeeze(0).squeeze(0)
    
    attn_mask_input = attention_mask
    is_sdpa_causal = (query_length > 1) and (attn_mask_input is None)
    if attn_mask_input is not None:
        is_sdpa_causal = False
        actual_kv_sequence_length = key_states_for_sdpa.shape[-2]
        if attn_mask_input.shape[-1] > actual_kv_sequence_length:
            attn_mask_input = attn_mask_input[:, :, :, :actual_kv_sequence_length]
            
    attention_output = F.scaled_dot_product_attention(
        query_states_rotated, key_states_for_sdpa, value_states_for_sdpa, attn_mask=attn_mask_input, dropout_p=0.0, is_causal=is_sdpa_causal
    )
    attention_output = attention_output.transpose(1, 2).contiguous().reshape(batch_size, query_length, hidden_size)
    attention_output = self_attn.o_proj(attention_output)
    
    return attention_output, None, past_key_value


class GemFilterRankingGenerator(BaseRankingGenerator):
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, args: argparse.Namespace):
        super().__init__(model, tokenizer, args)
        self.topk = getattr(args, 'topk', 1024)
        self.topk_percentage = getattr(args, 'topk_percentage', None)
        self.select_layer_idx = getattr(args, 'select_layer_idx', 13)  # Default middle layer
        self.kernel_size = args.kernel_size
        self.original_prompt_len = None
        self.gemfilter_scores = None
        
    def _patch_model(self):
        # Only patch the select layer
        layer = self.model.model.layers[self.select_layer_idx]
        if hasattr(layer, 'self_attn') and isinstance(layer.self_attn, LlamaAttention):
            attn_module = layer.self_attn
            if self.select_layer_idx not in self.orig_fwds:
                self.orig_fwds[self.select_layer_idx] = attn_module.forward
            setattr(attn_module, "layer_idx", self.select_layer_idx)
            attn_module.forward = types.MethodType(partial(_patched_attention_forward_gemfilter, generator_obj=self), attn_module)

    @torch.no_grad()
    def generate_rankings(self, inputs: Dict[str, torch.Tensor]) -> Dict[int, torch.Tensor]:
        self.gemfilter_scores = None
        self.original_prompt_len = None
        self._patch_model()
        
        try:
            input_ids = inputs['input_ids']
            prompt_length = input_ids.shape[1]
            self.original_prompt_len = prompt_length
            
            # Run forward pass to compute GemFilter scores
            position_ids = torch.arange(prompt_length, device=self.device).unsqueeze(0)  # Add batch dimension
            with torch.no_grad():
                _ = self.model(input_ids=input_ids, position_ids=position_ids, use_cache=True)
            
            # Return the GemFilter scores if computed
            if self.gemfilter_scores is not None:
                # For GemFilter, we simulate different layers being the select layer
                rankings = {}
                for layer_idx in range(self.config.num_hidden_layers):
                    rankings[layer_idx] = self.gemfilter_scores.clone()
                return rankings
            else:
                return {}
                
        finally:
            self._unpatch_model()
            # Clear GPU cache to free memory for next generator
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# --- Strategy 3: CLAA ---

def _patched_attention_forward_claa(
    self_attn: LlamaAttention,
    generator_obj: 'CLAARankingGenerator',
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
    batch_size, query_length, _ = hidden_states.size()
    num_heads = self_attn.config.num_attention_heads
    head_dim = self_attn.config.hidden_size // num_heads
    num_key_value_heads = self_attn.config.num_key_value_heads
    hidden_size = self_attn.config.hidden_size
    num_key_value_groups = num_heads // num_key_value_heads
    
    query_projection = self_attn.q_proj(hidden_states)
    key_projection = self_attn.k_proj(hidden_states)
    value_projection = self_attn.v_proj(hidden_states)
    
    query_states = query_projection.view(batch_size, query_length, num_heads, head_dim).transpose(1, 2)
    key_states_before_rope = key_projection.view(batch_size, query_length, num_key_value_heads, head_dim).transpose(1, 2)
    value_states_for_cache = value_projection.view(batch_size, query_length, num_key_value_heads, head_dim).transpose(1, 2)
    
    cos, sin = self_attn.rotary_emb(value_states_for_cache, position_ids=position_ids)
    query_states_rotated, key_states_rotated = hf_apply_rotary_pos_emb(query_states, key_states_before_rope, cos, sin, position_ids)

    key_states_for_sdpa, value_states_for_sdpa = past_key_value.update(
        key_states_rotated, value_states_for_cache, self_attn.layer_idx, {"cache_position": cache_position}
    )
    key_states_for_sdpa = hf_repeat_kv(key_states_for_sdpa, num_key_value_groups)
    value_states_for_sdpa = hf_repeat_kv(value_states_for_sdpa, num_key_value_groups)
    
    # Compute CLAA scores - similar to FastKV but maintain rolling buffer
    if query_length > generator_obj.window_size:
        # Reset buffer at layer 0
        if self_attn.layer_idx == 0:
            generator_obj.score_buffer.clear()
        
        # Compute attention scores like FastKV
        attn_weights = torch.matmul(query_states_rotated[..., -generator_obj.window_size:, :], key_states_for_sdpa.transpose(-1, -2)) / math.sqrt(head_dim)
        
        # Apply causal mask for the window
        mask = torch.full((generator_obj.window_size, generator_obj.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        attn_weights[:, :, -generator_obj.window_size:, -generator_obj.window_size:] += mask[None, None, :, :]
        
        # Softmax and sum over query positions (last window_size tokens)
        attn_weights_softmax = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states_rotated.dtype)
        attn_weights_sum = attn_weights_softmax[:, :, -generator_obj.window_size:, :-generator_obj.window_size].sum(dim=-2)
        
        # Apply pooling as in CLAA
        if generator_obj.pooling == 'avgpool':
            attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size=generator_obj.kernel_size, padding=generator_obj.kernel_size//2, stride=1)
        elif generator_obj.pooling == 'maxpool':
            attn_cache = F.max_pool1d(attn_weights_sum, kernel_size=generator_obj.kernel_size, padding=generator_obj.kernel_size//2, stride=1)
        else:
            attn_cache = attn_weights_sum
            
        # Sum across heads to get layer-level importance scores  
        if len(attn_cache.shape) == 3:  # [batch, heads, seq]
            attn_cache = attn_cache.sum(dim=1, keepdim=True)  # Sum over heads, keep dims for buffer
        
        # Update rolling score buffer
        generator_obj.score_buffer.append(attn_cache)
        if len(generator_obj.score_buffer) > generator_obj.last_n_layers:
            generator_obj.score_buffer.pop(0)
        
        # Compute final CLAA scores if we have enough layers and this is the TSP layer
        if (self_attn.layer_idx == generator_obj.tsp_layer_idx and 
            len(generator_obj.score_buffer) >= generator_obj.last_n_layers):
            # CLAA: Stack scores from last N layers and take max across layers and heads
            scores_tensor = torch.stack(generator_obj.score_buffer, dim=1)  # [batch, n_layers, heads, seq_len]
            flattened_scores = scores_tensor.flatten(start_dim=1, end_dim=2)  # [batch, n_layers*heads, seq_len]
            final_token_importance, _ = flattened_scores.max(dim=1)  # [batch, seq_len]
            generator_obj.claa_scores = final_token_importance.squeeze(0)  # Remove batch dimension
    
    attn_mask_input = attention_mask
    is_sdpa_causal = (query_length > 1) and (attn_mask_input is None)
    if attn_mask_input is not None:
        is_sdpa_causal = False
        actual_kv_sequence_length = key_states_for_sdpa.shape[-2]
        if attn_mask_input.shape[-1] > actual_kv_sequence_length:
            attn_mask_input = attn_mask_input[:, :, :, :actual_kv_sequence_length]
            
    attention_output = F.scaled_dot_product_attention(
        query_states_rotated, key_states_for_sdpa, value_states_for_sdpa, attn_mask=attn_mask_input, dropout_p=0.0, is_causal=is_sdpa_causal
    )
    attention_output = attention_output.transpose(1, 2).contiguous().reshape(batch_size, query_length, hidden_size)
    attention_output = self_attn.o_proj(attention_output)
    
    return attention_output, None, past_key_value


class CLAARankingGenerator(BaseRankingGenerator):
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, args: argparse.Namespace):
        super().__init__(model, tokenizer, args)
        self.window_size = args.window_size
        self.kernel_size = args.kernel_size
        self.pooling = args.pooling
        self.last_n_layers = 4  # Fixed as requested
        self.tsp_layer_idx = self.last_n_layers - 1  # TSP layer should be after we have enough layers in buffer
        self.score_buffer = []  # Rolling buffer for attention scores
        self.claa_scores = None
        
    def _patch_model(self):
        # Patch all layers up to and including TSP layer
        for i in range(self.tsp_layer_idx + 1):
            layer = self.model.model.layers[i]
            if hasattr(layer, 'self_attn') and isinstance(layer.self_attn, LlamaAttention):
                attn_module = layer.self_attn
                if i not in self.orig_fwds:
                    self.orig_fwds[i] = attn_module.forward
                setattr(attn_module, "layer_idx", i)
                attn_module.forward = types.MethodType(partial(_patched_attention_forward_claa, generator_obj=self), attn_module)

    @torch.no_grad()
    def generate_rankings(self, inputs: Dict[str, torch.Tensor]) -> Dict[int, torch.Tensor]:
        self.claa_scores = None
        self.score_buffer.clear()
        self._patch_model()
        
        try:
            input_ids = inputs['input_ids']
            prompt_length = input_ids.shape[1]
            
            # Run forward pass to compute CLAA scores
            position_ids = torch.arange(prompt_length, device=self.device).unsqueeze(0)  # Add batch dimension
            with torch.no_grad():
                _ = self.model(input_ids=input_ids, position_ids=position_ids, use_cache=True)
            
            # Return the CLAA scores if computed
            if self.claa_scores is not None:
                # For CLAA, we simulate different layers being the TSP layer
                rankings = {}
                for layer_idx in range(self.config.num_hidden_layers):
                    rankings[layer_idx] = self.claa_scores.clone()
                return rankings
            else:
                return {}
                
        finally:
            self._unpatch_model()
            # Clear GPU cache to free memory for next generator
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.score_buffer.clear()

# --- Strategy 4: Speculative ---

def _patched_attention_forward_speculative(
    self_attn: LlamaAttention,
    generator_obj: 'SpeculativeRankingGenerator',
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
    batch_size, query_length, _ = hidden_states.size()
    num_heads = self_attn.config.num_attention_heads
    head_dim = self_attn.config.hidden_size // num_heads
    num_key_value_heads = self_attn.config.num_key_value_heads
    hidden_size = self_attn.config.hidden_size
    num_key_value_groups = num_heads // num_key_value_heads
    
    query_projection = self_attn.q_proj(hidden_states)
    key_projection = self_attn.k_proj(hidden_states)
    value_projection = self_attn.v_proj(hidden_states)
    
    query_states = query_projection.view(batch_size, query_length, num_heads, head_dim).transpose(1, 2)
    key_states_before_rope = key_projection.view(batch_size, query_length, num_key_value_heads, head_dim).transpose(1, 2)
    value_states_for_cache = value_projection.view(batch_size, query_length, num_key_value_heads, head_dim).transpose(1, 2)
    
    cos, sin = self_attn.rotary_emb(value_states_for_cache, position_ids=position_ids)
    query_states_rotated, key_states_rotated = hf_apply_rotary_pos_emb(query_states, key_states_before_rope, cos, sin, position_ids)

    # Only capture Qs during the speculative decoding phase
    if generator_obj.is_generating_speculatively and query_length == 1:
        generator_obj.captured_qs[self_attn.layer_idx].append(query_states_rotated.detach().clone())

    # This past_key_value is updated IN-PLACE. The original `prompt_only_cache` is preserved outside.
    key_states_for_sdpa, value_states_for_sdpa = past_key_value.update(
        key_states_rotated, value_states_for_cache, self_attn.layer_idx, {"cache_position": cache_position}
    )
    key_states_for_sdpa = hf_repeat_kv(key_states_for_sdpa, num_key_value_groups)
    value_states_for_sdpa = hf_repeat_kv(value_states_for_sdpa, num_key_value_groups)
    
    attn_mask_input = attention_mask
    is_sdpa_causal = (query_length > 1) and (attn_mask_input is None)
    if attn_mask_input is not None:
        is_sdpa_causal = False
        actual_kv_sequence_length = key_states_for_sdpa.shape[-2]
        if attn_mask_input.shape[-1] > actual_kv_sequence_length:
            attn_mask_input = attn_mask_input[:, :, :, :actual_kv_sequence_length]
            
    attention_output = F.scaled_dot_product_attention(
        query_states_rotated, key_states_for_sdpa, value_states_for_sdpa, attn_mask=attn_mask_input, dropout_p=0.0, is_causal=is_sdpa_causal
    )
    attention_output = attention_output.transpose(1, 2).contiguous().reshape(batch_size, query_length, hidden_size)
    attention_output = self_attn.o_proj(attention_output)
    
    return attention_output, None, past_key_value


class SpeculativeRankingGenerator(BaseRankingGenerator):
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, args: argparse.Namespace):
        super().__init__(model, tokenizer, args)
        self.eos_token_ids = self._extract_eos_token_ids()
        self.lookahead_k_values = args.lookahead_k_values
        self.max_k = max(self.lookahead_k_values) if self.lookahead_k_values else 0
        self.pool_kernel_size = args.kernel_size if args.pooling != 'none' else None
        self.pool_type = args.pooling.lower()
        self._validate_pooling_config()
        self.captured_qs: List[List[torch.Tensor]] = []
        self.is_generating_speculatively = False
        self.rankings: Dict[int, torch.Tensor] = {} # Keyed by k, value is the ranking tensor

    def _validate_pooling_config(self):
        if self.pool_type not in ['avgpool', 'maxpool', 'none']:
            raise ValueError(f"pool_type must be 'avgpool', 'maxpool', or 'none', but got {self.pool_type}")
        if self.pool_kernel_size is not None:
            if self.pool_kernel_size <= 1: self.pool_kernel_size = None
            elif self.pool_kernel_size % 2 == 0: raise ValueError("pool_kernel_size must be an odd number.")
            if self.pool_type == 'none' and self.pool_kernel_size is not None: raise ValueError("pool_kernel_size is specified, but pool_type is 'none'.")
        if self.pool_type != 'none' and self.pool_kernel_size is None:
            raise ValueError(f"pool_type is '{self.pool_type}', but pool_kernel_size is not specified.")

    def _extract_eos_token_ids(self) -> List[int]:
        config_eos = self.config.eos_token_id
        if isinstance(config_eos, int): return [config_eos]
        if isinstance(config_eos, list): return list(config_eos)
        if self.tokenizer.eos_token_id: return [self.tokenizer.eos_token_id]
        raise ValueError("eos_token_id not defined in config or tokenizer.")

    def _patch_model(self):
        num_layers = self.config.num_hidden_layers
        self.captured_qs = [[] for _ in range(num_layers)]
        for i, layer in enumerate(self.model.model.layers):
            if hasattr(layer, 'self_attn') and isinstance(layer.self_attn, LlamaAttention):
                attn_module = layer.self_attn
                if i not in self.orig_fwds: self.orig_fwds[i] = attn_module.forward
                setattr(attn_module, "layer_idx", i)
                attn_module.forward = types.MethodType(partial(_patched_attention_forward_speculative, generator_obj=self), attn_module)

    def _compute_raw_qk_scores(self, prompt_only_cache_tuple: Tuple[Tuple[torch.Tensor, torch.Tensor], ...]) -> torch.Tensor:
        if not self.captured_qs or not any(self.captured_qs): return torch.empty(0)
        
        num_layers = self.config.num_hidden_layers
        num_q_heads = self.config.num_attention_heads
        num_kv_heads = self.config.num_key_value_heads
        num_kv_groups = num_q_heads // num_kv_heads
        head_dim = self.config.hidden_size // num_q_heads

        all_layer_scores = []
        for layer_idx in range(num_layers):
            if not self.captured_qs[layer_idx] or not prompt_only_cache_tuple[layer_idx][0].numel(): continue
            key_prompt_layer = prompt_only_cache_tuple[layer_idx][0].detach()
            key_prompt_layer_repeated = hf_repeat_kv(key_prompt_layer, num_kv_groups)
            all_q_for_layer = torch.cat(self.captured_qs[layer_idx], dim=2)
            attn_logits = torch.matmul(all_q_for_layer, key_prompt_layer_repeated.transpose(-1, -2)) / math.sqrt(head_dim)
            all_layer_scores.append(attn_logits)
        if not all_layer_scores: return torch.empty(0)

        # Shape: [batch_size, num_layers, num_heads, num_gen_tokens, prompt_len]
        return torch.stack(all_layer_scores, dim=1)

    def _token_importance_from_attn_scores(self, attn_scores: torch.Tensor):
        # This function now mirrors the logic from the corrected oracle script.
        if attn_scores.numel() == 0:
            return

        # Input shape: [batch_size, num_layers, num_heads, num_gen_tokens, prompt_len]
        bs, num_layers, num_heads, num_gen_tokens, prompt_len = attn_scores.shape

        if bs != 1:
            raise NotImplementedError("Batch size > 1 is not supported for speculative ranking.")
        
        # Softmax over the prompt length dimension
        probs = F.softmax(attn_scores, dim=-1, dtype=torch.bfloat16)

        # Permute to [B, N_gen, S_prompt, L, H] for easier aggregation
        probs = probs.permute(0, 3, 4, 1, 2)

        # Step 1: Max-aggregation over Layers and Heads
        peak_importance, _ = torch.max(torch.max(probs, dim=-1)[0], dim=-1)
        # Shape: [B, N_gen, S_prompt]

        # Now, calculate rankings for each 'k'
        for k in self.lookahead_k_values:
            if k > num_gen_tokens:
                continue
            
            # Step 2: Mean-aggregation over the first 'k' generated tokens
            scores_for_k = peak_importance[:, :k, :]
            mean_importance = torch.mean(scores_for_k, dim=1)
            # Shape: [B, S_prompt]

            # Step 3 (Optional): 1D pooling on the final aggregated scores
            final_scores = mean_importance
            kernel_size = self.pool_kernel_size
            if kernel_size and prompt_len >= kernel_size:
                scores_for_pooling = final_scores.unsqueeze(1) # Add channel dim
                
                pool_fn = F.avg_pool1d if self.pool_type == 'avgpool' else F.max_pool1d
                
                pooled_scores = pool_fn(
                    scores_for_pooling,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=kernel_size // 2
                )
                final_scores = pooled_scores.squeeze(1)
            
            self.rankings[k] = final_scores.squeeze(0).to(self.model.dtype)

    @torch.no_grad()
    def generate_rankings(self, inputs: Dict[str, torch.Tensor]) -> Dict[int, torch.Tensor]:
        self.rankings.clear()
        if self.max_k == 0: return {}
        self._patch_model()
        try:
            input_ids = inputs['input_ids']
            prompt_length = input_ids.shape[1]
            
            # --- Stage 1: Prompt Prefill ---
            self.is_generating_speculatively = False
            prefill_pos = torch.arange(prompt_length, device=self.device)
            prefill_out = self.model(input_ids=input_ids, use_cache=True, cache_position=prefill_pos)
            prefill_cache = prefill_out.past_key_values
            
            if isinstance(prefill_cache, tuple): prefill_cache = DynamicCache.from_legacy_cache(prefill_cache)
            prompt_only_cache_tuple = prefill_cache.to_legacy_cache()
            
            current_tokens = torch.argmax(prefill_out.logits[:, -1, :], dim=-1, keepdim=True)
            current_cache = prefill_cache

            # --- Stage 2: Speculative Token Generation ---
            self.is_generating_speculatively = True
            for _ in range(self.max_k):
                if current_tokens.item() in self.eos_token_ids: break
                
                cache_len = current_cache.get_seq_length(0)
                pos_ids = torch.tensor([[cache_len]], device=self.device)
                decode_cache_pos = torch.tensor([cache_len], device=self.device)
                
                decode_out = self.model(input_ids=current_tokens, position_ids=pos_ids, past_key_values=current_cache, use_cache=True, cache_position=decode_cache_pos)
                
                current_cache = decode_out.past_key_values
                if isinstance(current_cache, tuple): current_cache = DynamicCache.from_legacy_cache(current_cache)
                
                current_tokens = torch.argmax(decode_out.logits[:, -1, :], dim=-1, keepdim=True)
            
            # --- Stage 3: Score Calculation ---
            raw_qk_scores = self._compute_raw_qk_scores(prompt_only_cache_tuple)
            self._token_importance_from_attn_scores(raw_qk_scores)

        finally:
            self._unpatch_model()
            self.captured_qs.clear()
        
        return self.rankings



# --- Main Execution Logic ---

def main():
    parser = argparse.ArgumentParser(description="Generate FastKV, GemFilter, and Speculative Prefill token rankings.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--datasets", nargs='+', required=True)
    parser.add_argument("--num_samples", type=int, default=15)
    parser.add_argument("--max_len", type=int, default=128000)
    parser.add_argument("--output_dir", type=str, default="analysis_results/approx_rankings")
    parser.add_argument("--window_size", type=int, default=8)
    parser.add_argument("--lookahead_k_values", type=int, nargs='+', default=[1, 2, 4, 8, 16, 32, 64, 128, 256])
    parser.add_argument("--kernel_size", type=int, default=13)
    parser.add_argument("--pooling", type=str, default="avgpool", choices=["avgpool", "maxpool", "none"])
    args = parser.parse_args()

    set_seed(42)
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, '..'))
        config_path = os.path.join(project_root, 'eval/longbench/config/dataset2prompt.json')
        with open(config_path, "r") as f:
            dataset2prompt = json.load(f)
    except FileNotFoundError:
        print(f"Warning: Could not find dataset2prompt.json at {config_path}. This might fail for some datasets.")
        dataset2prompt = {}

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto")

    generators = {
        'fastkv': FastKVRankingGenerator(model, tokenizer, args),
        'gemfilter': GemFilterRankingGenerator(model, tokenizer, args),
        'claa': CLAARankingGenerator(model, tokenizer, args),
        'speculative': SpeculativeRankingGenerator(model, tokenizer, args)
    }
    print("Initialized all ranking generators.")

    model_name_sanitized = args.model.replace('/', '_')

    for dataset_name in args.datasets:
        print(f"\n--- Processing Dataset: {dataset_name.upper()} ---")
        if dataset_name not in dataset2prompt:
            print(f"Warning: No prompt format found for {dataset_name}. Skipping.")
            continue
            
        dataset_output_dir = os.path.join(args.output_dir, model_name_sanitized)
        os.makedirs(dataset_output_dir, exist_ok=True)
        output_path = os.path.join(dataset_output_dir, f"{dataset_name}.npz")

        try:
            # Added more specific exceptions for robustness
            if os.path.exists(output_path):
                existing_data = dict(np.load(output_path, allow_pickle=True))
            else:
                existing_data = {}
        except (FileNotFoundError, EOFError, zipfile.BadZipFile):
            print(f"Warning: Could not load existing data from {output_path}. Starting fresh for this dataset.")
            existing_data = {}

        data = load_dataset('THUDM/LongBench', dataset_name, split='test', trust_remote_code=True)
        processed_count = 0

        for i, sample in enumerate(data):
            if processed_count >= args.num_samples: break
            sample_key = f'sample_{i}'
            # Check if this specific sample and all its ranking types are already done.
            if sample_key in existing_data and all(f'{name}_rankings' in existing_data[sample_key].item() for name in generators):
                 print(f"Skipping already fully processed sample {i}.")
                 processed_count += 1
                 continue

            try:
                raw_prompt = dataset2prompt[dataset_name].format(**sample)
            except KeyError: continue

            datasets_without_chat_template = ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]
            if dataset_name not in datasets_without_chat_template:
                messages = [{"role": "user", "content": raw_prompt}]
                final_prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                final_prompt_text = raw_prompt

            inputs = tokenizer(final_prompt_text, return_tensors="pt").to(model.device)
            prompt_len = inputs.input_ids.shape[1]

            if prompt_len > args.max_len:
                continue
            if prompt_len <= args.window_size:
                print(f"Skipping sample {i}: too short ({prompt_len} <= {args.window_size} tokens)")
                continue

            
            all_rankings = {}
            for name, gen in generators.items():
                try:
                    all_rankings[name] = gen.generate_rankings(inputs)
                except Exception as e:
                    import traceback
                    print(f"  -> ERROR generating ranking for '{name}' on sample {i}: {e}")
                    print(f"  -> Full traceback:")
                    traceback.print_exc()
                    all_rankings[name] = {}

            if not any(all_rankings.values()):
                print(f"  -> Warning: Failed to generate any ranking types for sample index {i}. Skipping.")
                continue

            processed_count += 1
            to_numpy = lambda d: {k: v.cpu().to(torch.float16).numpy() for k, v in d.items()} if d else {}
            
            sample_data = {'input_ids': inputs.input_ids.squeeze(0).cpu().numpy()}
            for name, ranks in all_rankings.items():
                if ranks:
                    sample_data[f'{name}_rankings'] = pickle.dumps(to_numpy(ranks))

            existing_data[sample_key] = np.array(sample_data, dtype=object)
            
            try:
                np.savez_compressed(output_path, **existing_data)
            except Exception as e:
                print(f"  -> ERROR saving data to {output_path}: {e}")

            if torch.cuda.is_available(): torch.cuda.empty_cache()

        print(f"\nSaved/Updated ranking results for {dataset_name} to {output_path}")


if __name__ == "__main__":
    main()
