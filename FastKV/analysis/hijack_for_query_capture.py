# analysis/hijack_for_query_capture.py

import torch
import types
from typing import Optional, Tuple, List
from functools import partial
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, LlamaAttention

def query_capture_attention_forward(
    self: LlamaAttention,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    # --- Custom arguments for hijack ---
    captured_qs_storage: Optional[List[List[torch.Tensor]]] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
    """
    A hijacked forward pass that captures a query vector and correctly updates a modern Cache object.
    """
    if captured_qs_storage is None:
        raise ValueError("The 'captured_qs_storage' list must be provided.")

    bsz, q_len, _ = hidden_states.size()
    if q_len != 1:
        raise ValueError(f"Query capture hijack expects q_len=1, but got {q_len}")

    # Project, reshape, and apply RoPE
    query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    
    cos, sin = self.rotary_emb(value_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # Capture the query vector
    captured_qs_storage[self.layer_idx].append(query_states.detach().cpu())

    # Update the modern Cache object in-place
    if use_cache and past_key_value is not None:
        cache_kwargs = {"cache_position": cache_position} if cache_position is not None else {}
        past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
    
    # Return a dummy output to save compute, and the cache object which was updated in-place.
    dummy_attn_output = torch.zeros_like(hidden_states)
    return dummy_attn_output, None, past_key_value

def patch_model_for_query_capture(model, storage_list: List[List[torch.Tensor]]):
    """
    Patches the model's LlamaAttention layers to use the query capture forward pass.
    """
    original_forwards = {}
    
    for i, layer in enumerate(model.model.layers):
        attn_module = layer.self_attn
        original_forwards[i] = attn_module.forward
        
        new_forward_partial = partial(
            query_capture_attention_forward,
            captured_qs_storage=storage_list
        )
        attn_module.forward = types.MethodType(new_forward_partial, attn_module)

    def unpatch():
        for i, layer in enumerate(model.model.layers):
            if i in original_forwards:
                layer.self_attn.forward = original_forwards[i]

    return unpatch
