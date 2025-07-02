
import torch
import torch.nn as nn
import types
import math
from typing import Optional, Tuple, Union

from transformers.models.llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb, repeat_kv, LlamaMLP, LlamaRMSNorm, BaseModelOutputWithPast
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.cache_utils import Cache, DynamicCache

# This will store the speculator-generated sorted indices
_speculator_indices = None
_original_prompt_len = 0

def set_speculator_data(indices, prompt_len):
    """Sets the speculator data for the forward pass."""
    global _speculator_indices, _original_prompt_len
    _speculator_indices = indices
    _original_prompt_len = prompt_len

def _select_tokens_by_chunk(scores: torch.Tensor, num_to_keep: int, chunk_size: int) -> torch.Tensor:
    """Selects tokens based on chunking."""
    original_seq_len = scores.shape[0]
    if original_seq_len <= num_to_keep:
        return torch.arange(original_seq_len, device=scores.device)

    num_chunks = math.ceil(original_seq_len / chunk_size)
    
    padding_len = num_chunks * chunk_size - original_seq_len
    if padding_len > 0:
        padded_scores = F.pad(scores, (0, padding_len), value=float('-inf'))
    else:
        padded_scores = scores

    avg_chunk_scores = padded_scores.view(num_chunks, chunk_size).mean(dim=1)
    
    num_chunks_to_keep = min(math.ceil(num_chunks * (num_to_keep / original_seq_len)), num_chunks)
    _, top_chunk_indices = torch.topk(avg_chunk_scores, k=int(num_chunks_to_keep))
    
    selected_indices = torch.cat([
        torch.arange(idx * chunk_size, (idx + 1) * chunk_size, device=scores.device) 
        for idx in top_chunk_indices
    ])
    
    final_indices = selected_indices[selected_indices < original_seq_len]
    
    if len(final_indices) > num_to_keep:
        _, top_k_in_chunks_indices = torch.topk(scores[final_indices], k=num_to_keep)
        final_indices = final_indices[top_k_in_chunks_indices]
        
    return torch.sort(final_indices)[0]


def _draft_tsp_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    current_global_indices: Optional[torch.Tensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    self.next_global_indices = None
    self.local_indices_to_keep = None
    if q_len > 1 and self.layer_idx in self.tsp_schedule and _speculator_indices is not None:
        percentage_to_keep = self.tsp_schedule[self.layer_idx]
        num_tokens_to_keep = int(_original_prompt_len * percentage_to_keep)
        
        # Since _speculator_indices is now pre-sorted, we can just slice it.
        global_indices_to_keep = _speculator_indices[:num_tokens_to_keep]

        # Find which of our current tokens are in the list of tokens we want to keep.
        is_kept = torch.isin(current_global_indices, global_indices_to_keep)
        local_indices_to_keep, _ = torch.sort(torch.where(is_kept)[0])
        self.local_indices_to_keep = local_indices_to_keep
        
        # The next layer's global index list are the ones we just selected.
        self.next_global_indices = current_global_indices[local_indices_to_keep]

        key_states = key_states[:, :, local_indices_to_keep, :]
        value_states = value_states[:, :, local_indices_to_keep, :]
        
        if attention_mask is not None:
             attention_mask = attention_mask[:, :, :, local_indices_to_keep]

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    attn_output = _flash_attention_forward(
        query_states, key_states, value_states, attention_mask, q_len, is_causal=self.is_causal
    )
    
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
    attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value

def _draft_tsp_decoder_layer_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    current_global_indices: Optional[torch.Tensor] = None,
    **kwargs,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)

    hidden_states, self_attn_weights, present_key_value = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        position_embeddings=position_embeddings,
        current_global_indices=current_global_indices,
    )
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    local_indices_to_keep = getattr(self.self_attn, 'local_indices_to_keep', None)
    self.next_global_indices = getattr(self.self_attn, 'next_global_indices', None)

    if local_indices_to_keep is not None:
        self.new_position_ids = torch.gather(position_ids, dim=1, index=local_indices_to_keep.unsqueeze(0))
        hidden_states = torch.gather(hidden_states, dim=1, index=local_indices_to_keep.unsqueeze(0).unsqueeze(-1).expand(-1, -1, hidden_states.size(2)))
    else:
        self.new_position_ids = None
    
    outputs = (hidden_states,)
    if output_attentions:
        outputs += (self_attn_weights,)
    if use_cache:
        outputs += (present_key_value,)
    return outputs

def _draft_tsp_model_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    bsz, q_len, _ = inputs_embeds.shape

    past_key_values = DynamicCache.from_legacy_cache(past_key_values) if use_cache and not isinstance(past_key_values, Cache) else past_key_values
    
    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )
    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    causal_mask = self._update_causal_mask(
        attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
    )
    hidden_states = inputs_embeds
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    # The initial set of global indices is simply all of them.
    current_global_indices = torch.arange(q_len, device=hidden_states.device)

    # The initial ranked list is the one provided by the speculator.
    current_ranked_indices = _speculator_indices.to(hidden_states.device) if _speculator_indices is not None else None

    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    for decoder_layer in self.layers:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_value=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            current_global_indices=current_global_indices,
            current_ranked_indices=current_ranked_indices,
        )

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        hidden_states = layer_outputs[0]
        
        new_position_ids = getattr(decoder_layer, 'new_position_ids', None)
        if new_position_ids is not None:
            position_ids = new_position_ids
            position_embeddings = self.rotary_emb(hidden_states, position_ids)
            current_global_indices = getattr(decoder_layer, 'next_global_indices', None)
            current_ranked_indices = getattr(decoder_layer, 'next_ranked_indices', None)
            
            if causal_mask is not None:
                causal_mask = causal_mask[:, :, -hidden_states.shape[1]:, -hidden_states.shape[1]:]

    hidden_states = self.norm(hidden_states)
    if output_hidden_states:
        all_hidden_states += (hidden_states,)
    
    next_cache = next_decoder_cache if use_cache else None
    
    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )

def replace_llama_for_draft_tsp(model, args):
    """Replaces the LlamaAttention forward method with our patched version."""
    tsp_schedule_str = getattr(args, 'tsp_schedule', "")
    schedule = {}
    if tsp_schedule_str:
        try:
            items = sorted([item.split(':') for item in tsp_schedule_str.split(',')], key=lambda x: int(x[0]))
            schedule = {int(k): float(v) for k, v in items}
        except (ValueError, TypeError):
            raise ValueError("Invalid tsp_schedule format. Expected 'layer_idx1:percentage1,layer_idx2:percentage2,...'")

    use_chunk_selection = getattr(args, 'use_chunk_selection', False)
    chunk_size = getattr(args, 'chunk_size', 32)

    model.model.forward = types.MethodType(_draft_tsp_model_forward, model.model)
    for i, layer in enumerate(model.model.layers):
        attn_layer = layer.self_attn
        attn_layer.tsp_schedule = schedule
        attn_layer.layer_idx = i
        attn_layer.is_causal = True
        attn_layer.use_chunk_selection = use_chunk_selection
        attn_layer.chunk_size = chunk_size
        attn_layer.forward = types.MethodType(_draft_tsp_attention_forward, attn_layer)
        layer.forward = types.MethodType(_draft_tsp_decoder_layer_forward, layer)
