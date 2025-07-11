# baseline/draft_tsp/monkeypatch.py

import torch
import torch.nn as nn
import types
import math
from typing import Optional, Tuple, Union

from transformers.models.llama.modeling_llama import (
    LlamaModel, 
    LlamaDecoderLayer, 
    BaseModelOutputWithPast
)
from transformers.cache_utils import Cache, DynamicCache

# --- Globals to hold speculator data ---
_speculator_indices: Optional[torch.Tensor] = None
_original_prompt_len: int = 0

def set_speculator_data(indices: torch.Tensor, prompt_len: int):
    """
    Sets the speculator-generated master plan for pruning decisions.
    Called once per run from the pipeline.
    """
    global _speculator_indices, _original_prompt_len
    _speculator_indices = indices
    _original_prompt_len = prompt_len


def _draft_tsp_decoder_layer_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    cache_position: Optional[torch.LongTensor] = None,
    # This argument will be passed explicitly by our patched model forward
    current_global_indices: Optional[torch.Tensor] = None,
    **kwargs,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    """
    A decoder layer forward pass that prunes its own output `hidden_states`
    based on an intersection of its current tokens and the master plan.
    """
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)

    # Self Attention
    attn_outputs = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        cache_position=cache_position,
    )
    hidden_states = residual + attn_outputs[0]

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    self.new_position_ids = None 
    self.new_global_indices = None
    
    is_prefill = hidden_states.shape[1] > 1

    if is_prefill and self.layer_idx in self.tsp_schedule and _speculator_indices is not None:
        percentage_to_keep = self.tsp_schedule[self.layer_idx]
        num_tokens_to_keep = int(_original_prompt_len * percentage_to_keep)
        
        # Get the master list of GLOBAL indices that should be kept for this stage
        master_indices_for_this_stage = _speculator_indices[:num_tokens_to_keep]
        
        # Find which of the tokens we *currently have* are also in the master plan for this stage.
        is_kept_mask = torch.isin(current_global_indices, master_indices_for_this_stage)
        
        # Convert this boolean mask to the LOCAL indices we need to gather from the current hidden_states.
        local_indices_to_keep = torch.where(is_kept_mask)[0]

        # Gather the new, shorter position_ids using the correct LOCAL indices
        self.new_position_ids = torch.gather(position_ids, dim=1, index=local_indices_to_keep.unsqueeze(0))

        # Gather the new set of global indices to pass to the next stage
        self.new_global_indices = torch.gather(current_global_indices, dim=0, index=local_indices_to_keep)
        
        # Slice the hidden_states that will be returned, using the correct LOCAL indices
        indices_to_keep_expanded = local_indices_to_keep.unsqueeze(0).unsqueeze(-1).expand(-1, -1, hidden_states.size(2))
        hidden_states = torch.gather(hidden_states, dim=1, index=indices_to_keep_expanded)

    outputs = (hidden_states,)
    if output_attentions:
        outputs += (attn_outputs[1],)
    
    if use_cache:
        outputs += (attn_outputs[-1],)
        
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
) -> Union[Tuple, BaseModelOutputWithPast]:
    """
    The main model forward pass, which now explicitly manages all state variables
    in its loop, ensuring correct propagation.
    """
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if use_cache and past_key_values is None:
        past_key_values = DynamicCache()

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    # --- Initialize all state variables for the loop ---
    hidden_states = inputs_embeds
    bsz, q_len, _ = hidden_states.shape

    if cache_position is None:
        cache_position = torch.arange(q_len, device=inputs_embeds.device)
    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    # This is now a loop-managed variable
    current_global_indices = torch.arange(q_len, device=inputs_embeds.device)
    
    causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions)

    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    
    for decoder_layer in self.layers:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # Pass the current, correct state to the layer
        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_value=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            current_global_indices=current_global_indices, # Explicitly pass global indices
        )
        
        hidden_states = layer_outputs[0]
        
        # --- Update all state variables based on the layer's output ---
        new_position_ids = getattr(decoder_layer, 'new_position_ids', None)
        if new_position_ids is not None:
            position_ids = new_position_ids
            # Crucially, update the global indices tracker for the next iteration
            current_global_indices = getattr(decoder_layer, 'new_global_indices')
            
            new_seq_len = position_ids.shape[-1]
            
            causal_mask = self._update_causal_mask(None, hidden_states, cache_position[:new_seq_len], past_key_values, output_attentions)
            cache_position = torch.arange(new_seq_len, device=hidden_states.device)

        if use_cache:
            if output_attentions:
                next_decoder_cache = layer_outputs[2]
            else:
                next_decoder_cache = layer_outputs[1]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = past_key_values if use_cache else None

    if not return_dict:
        output_tuple = (hidden_states,)
        if use_cache:
            output_tuple += (next_cache,)
        if output_hidden_states:
            output_tuple += (all_hidden_states,)
        if output_attentions:
            output_tuple += (all_self_attns,)
        return output_tuple
    
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


def replace_llama_for_draft_tsp(model: LlamaModel, args):
    """
    Replaces the Llama forward methods with our new, corrected, and
    generalizable in-flight funneling architecture.
    """
    tsp_schedule_str = getattr(args, 'tsp_schedule', "")
    schedule = {}
    if tsp_schedule_str:
        try:
            items = sorted([item.split(':') for item in tsp_schedule_str.split(',')], key=lambda x: int(x[0]))
            schedule = {int(k): float(v) for k, v in items}
        except (ValueError, TypeError):
            raise ValueError("Invalid tsp_schedule format. Expected 'layer_idx1:percentage1,layer_idx2:percentage2,...'")

    model.model.forward = types.MethodType(_draft_tsp_model_forward, model.model)

    for i, layer in enumerate(model.model.layers):
        layer.layer_idx = i
        layer.tsp_schedule = schedule
        layer.forward = types.MethodType(_draft_tsp_decoder_layer_forward, layer)
