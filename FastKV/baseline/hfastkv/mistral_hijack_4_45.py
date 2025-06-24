# baseline/hfastkv/mistral_hijack_4_45.py

import inspect
import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, List
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.models.mistral.modeling_mistral import (
    apply_rotary_pos_emb,
    repeat_kv,
    BaseModelOutputWithPast,
    MistralFlashAttention2
)
from transformers.utils import logging
from baseline.hfastkv.hfastkv_utils import init_hfastkv

logger = logging.get_logger(__name__)

class MistralHFastKVAttention(MistralFlashAttention2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        init_hfastkv(self)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ):
        output_attentions = False
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}
            if q_len > 1:
                # Get compressed KV and the indices for the next layer's funneling
                key_states_compress, value_states_compress, self.propagated_indices = self.kv_cluster.update_kv(
                    key_states, query_states, value_states, attention_mask, self.num_key_value_groups, self.layer_idx)
                past_key_value.update(key_states_compress, value_states_compress, self.layer_idx, cache_kwargs)
            else: # Generation phase
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
                self.propagated_indices = None
        
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        attn_output = _flash_attention_forward(
            query_states, key_states, value_states, attention_mask, q_len,
            dropout=0.0, sliding_window=getattr(self.config, "sliding_window", None),
            is_causal=self.is_causal,
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value

def mistral_decoderlayer_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)

    hidden_states, self_attn_weights, present_key_value = self.self_attn(
        hidden_states=hidden_states, attention_mask=attention_mask, position_ids=position_ids,
        past_key_value=past_key_value, output_attentions=output_attentions, use_cache=use_cache,
        cache_position=cache_position, **kwargs,
    )
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    # HFastKV: Check for propagated indices from the attention layer
    propagated_indices = self.self_attn.propagated_indices
    if propagated_indices is not None:
        # Funnel position_ids for the next layer's RoPE
        self.new_position_ids = torch.gather(position_ids, dim=1, index=propagated_indices)
        # Funnel the hidden_states to the selected tokens
        propagated_indices = propagated_indices.unsqueeze(-1)
        hidden_states = torch.gather(hidden_states, dim=1, index=propagated_indices.expand(-1, -1, hidden_states.size(2)))
    else:
        self.new_position_ids = None

    outputs = (hidden_states,)
    if output_attentions:
        outputs += (self_attn_weights,)
    if use_cache:
        outputs += (present_key_value,)
    return outputs

def mistral_model_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one")

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)
    
    past_key_values = DynamicCache.from_legacy_cache(past_key_values) if use_cache and not isinstance(past_key_values, Cache) else past_key_values

    if cache_position is None:
        cache_position = torch.arange(past_key_values.get_seq_length() if past_key_values is not None else 0, inputs_embeds.shape[1] + (past_key_values.get_seq_length() if past_key_values is not None else 0), device=inputs_embeds.device)
    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position, past_key_values, use_cache, output_attentions)
    hidden_states = inputs_embeds

    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    for decoder_layer in self.layers:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        layer_outputs = decoder_layer(
            hidden_states, attention_mask=causal_mask, position_ids=position_ids,
            past_key_value=past_key_values, output_attentions=output_attentions,
            use_cache=use_cache, cache_position=cache_position,
        )

        # HFastKV: Check for new position_ids from the decoder layer and update for the next loop
        new_position_ids = getattr(decoder_layer, 'new_position_ids', None)
        if new_position_ids is not None:
            position_ids = new_position_ids
            # The causal mask will be correctly handled by Flash Attention's internal masking
            # for the now non-contiguous positions.
        
        hidden_states = layer_outputs[0]
        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]
        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)
    if output_hidden_states:
        all_hidden_states += (hidden_states,)
    
    next_cache = next_decoder_cache if use_cache else None
    # Final hidden state for generation is always the last token
    hidden_states = hidden_states[:, -1:, :]

    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_state,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )
