# Save as: analysis/hijack_for_ranking_analysis.py

import torch
from typing import Optional, Tuple
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
from transformers.modeling_flash_attention_utils import _flash_attention_forward

def analysis_attention_forward(
    self, # The LlamaFlashAttention2 instance
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    # --- FIX is here: Added position_embeddings ---
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    # ----------------------------------------------
    all_layer_rankings_storage: Optional[dict] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
    """
    A hijacked forward pass for LlamaFlashAttention2 that computes and stores
    token ranking scores for each layer instead of performing KV caching.
    """
    if all_layer_rankings_storage is None:
        raise ValueError("The 'all_layer_rankings_storage' dictionary must be provided for analysis.")

    bsz, q_len, _ = hidden_states.size()

    # Standard Attention Projections
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    # --- FIX is here: Use the passed-in embeddings ---
    if position_embeddings is None:
        # Fallback for older transformers versions or different call patterns
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        # This is the expected path for modern transformers
        cos, sin = position_embeddings
    # ---------------------------------------------------

    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # --- THE CRITICAL ANALYSIS STEP (remains the same) ---
    self.kv_cluster._compute_and_store_ranking(
        query_states=query_states.detach(),
        key_states=key_states.detach(),
        num_key_value_groups=self.num_key_value_groups,
        layer_idx=self.layer_idx,
        all_layer_rankings_storage=all_layer_rankings_storage
    )
    # ---------------------------------------------------

    # --- Perform a regular attention pass to allow the model to continue ---
    key_states_for_attn = repeat_kv(key_states, self.num_key_value_groups)
    value_states_for_attn = repeat_kv(value_states, self.num_key_value_groups)

    query_states = query_states.transpose(1, 2)
    key_states_for_attn = key_states_for_attn.transpose(1, 2)
    value_states_for_attn = value_states_for_attn.transpose(1, 2)

    attn_output = _flash_attention_forward(
        query_states,
        key_states_for_attn,
        value_states_for_attn,
        attention_mask,
        q_len,
        is_causal=self.is_causal,
    )

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
    attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value
