# baseline/hfastkv/llama_hijack_4_45.py (Corrected Version)

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, List
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    repeat_kv,
    BaseModelOutputWithPast,
    LlamaFlashAttention2
)
from transformers.utils import logging
from baseline.hfastkv.hfastkv_utils import init_hfastkv, HFastKVCluster

logger = logging.get_logger(__name__)

class LlamaHFastKVAttention(LlamaFlashAttention2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        init_hfastkv(self)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        output_attentions = False
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
            
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}
            if q_len > 1:
                key_states_compress, value_states_compress, self.propagated_indices = self.kv_cluster.update_kv(
                    key_states, query_states, value_states, attention_mask, self.num_key_value_groups, self.layer_idx)
                past_key_value.update(key_states_compress, value_states_compress, self.layer_idx, cache_kwargs)
            else:
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
                self.propagated_indices = None
        
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        attn_output = _flash_attention_forward(
            query_states, key_states, value_states, attention_mask, q_len,
            is_causal=self.is_causal,
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value

def llama_decoderlayer_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)

    hidden_states, self_attn_weights, present_key_value = self.self_attn(
        hidden_states=hidden_states, attention_mask=attention_mask, position_ids=position_ids,
        past_key_value=past_key_value, output_attentions=output_attentions, use_cache=use_cache,
        cache_position=cache_position, position_embeddings=position_embeddings, **kwargs,
    )
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    propagated_indices = getattr(self.self_attn, 'propagated_indices', None)
    if propagated_indices is not None:
        self.new_position_ids = torch.gather(position_ids, dim=1, index=propagated_indices)
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

def llama_model_forward(
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

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    past_key_values = DynamicCache.from_legacy_cache(past_key_values) if use_cache and not isinstance(past_key_values, Cache) else past_key_values
    
    if cache_position is None:
        cache_position = torch.arange(past_key_values.get_seq_length() if past_key_values is not None else 0, inputs_embeds.shape[1] + (past_key_values.get_seq_length() if past_key_values is not None else 0), device=inputs_embeds.device)
    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions)
    hidden_states = inputs_embeds
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    for decoder_layer in self.layers:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        layer_outputs = decoder_layer(
            hidden_states, attention_mask=causal_mask, position_ids=position_ids,
            past_key_value=past_key_values, output_attentions=output_attentions,
            use_cache=use_cache, cache_position=cache_position, position_embeddings=position_embeddings,
        )

        new_position_ids = getattr(decoder_layer, 'new_position_ids', None)
        if new_position_ids is not None:
            position_ids = new_position_ids
            # *** THE FIX: Recompute rotary embeddings for the new, smaller set of tokens and positions ***
            position_embeddings = self.rotary_emb(layer_outputs[0], position_ids)
            
        hidden_states = layer_outputs[0]
        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]
        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)
    if output_hidden_states:
        all_hidden_states += (hidden_states,)
    
    next_cache = next_decoder_cache if use_cache else None
    hidden_states = hidden_states[:, -1:, :]

    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states, past_key_values=next_cache,
        hidden_states=all_hidden_states, attentions=all_self_attns,
    )
