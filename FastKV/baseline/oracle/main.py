# baseline/oracle/main.py
import torch
import torch.nn as nn
import transformers
import logging
from typing import Optional, Tuple, Union, List
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb, BaseModelOutputWithPast, LlamaFlashAttention2, LLAMA_ATTENTION_CLASSES
)
from transformers.models.mistral.modeling_mistral import (
    repeat_kv as mistral_repeat_kv, MistralFlashAttention2, MISTRAL_ATTENTION_CLASSES
)

logger = logging.getLogger(__name__)

# =====================================================================================
# 1. ORACLE KV CACHE LOGIC
# =====================================================================================

class OracleKVCluster:
    """
    A KV Cluster that prunes the cache based on pre-computed oracle rankings.
    It does not calculate attention scores for pruning.
    """
    def __init__(self, tsp_layer=False, max_capacity_prompt_percentage=None):
        self.tsp_layer = tsp_layer
        self.max_capacity_prompt_percentage = max_capacity_prompt_percentage
        self.original_prompt_len = 0

    def update_kv(self, key_states, value_states, layer_idx, **kwargs):
        """
        Prunes the key and value states using oracle rankings passed via kwargs.
        """
        bsz, num_kv_heads, q_len, head_dim = key_states.shape
        
        # On the first layer of a new pass, record the original prompt length
        if layer_idx == 0:
            self.original_prompt_len = q_len
            logging.info(f"OracleKV: Set prompt length = {self.original_prompt_len}")

        # Retrieve oracle rankings from the model object (passed via kwargs)
        oracle_rankings = kwargs.get("oracle_rankings", None)

        # Determine the number of tokens to keep
        if self.max_capacity_prompt_percentage is not None:
            num_to_keep = int(self.original_prompt_len * self.max_capacity_prompt_percentage)
        else:
            # Fallback to a default if not set, though it should be.
            num_to_keep = int(self.original_prompt_len * 0.1) 

        # If no oracle is provided or we are keeping everything, do nothing.
        if oracle_rankings is None or num_to_keep >= q_len:
            return key_states, value_states, None

        # --- ORACLE LOGIC ---
        # Instead of calculating attention, we directly use the provided oracle rankings.
        _, top_k_indices = torch.topk(oracle_rankings, k=num_to_keep, dim=-1)
        
        # Sort indices to maintain positional order for RoPE
        indices_to_keep = torch.sort(top_k_indices).values
        
        # Expand indices for gathering
        indices_for_gather = indices_to_keep.unsqueeze(0).unsqueeze(-1).expand(bsz, num_kv_heads, -1, head_dim)
        
        # Gather the important K/V pairs
        k_past_compress = key_states.gather(dim=2, index=indices_for_gather)
        v_past_compress = value_states.gather(dim=2, index=indices_for_gather)

        tsp_indices = None
        if self.tsp_layer:
            # The indices for pruning hidden states are the ones we just selected
            tsp_indices = indices_to_keep.unsqueeze(0) # Add batch dim
            logging.info(f"OracleKV Layer {layer_idx}: TSP applied - selected {tsp_indices.shape[-1]} tokens using oracle.")

        return k_past_compress, v_past_compress, tsp_indices

def init_oracle_kv(self):
    """Initializes the cluster on an attention module."""
    self.kv_cluster = OracleKVCluster()

# =====================================================================================
# 2. LLAMA MODEL HIJACKING
# =====================================================================================

class LlamaOracleKVAttention(LlamaFlashAttention2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        init_oracle_kv(self)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
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
            if q_len > 1: # Prefill phase
                key_states_compress, value_states_compress, self.tsp_idx = self.kv_cluster.update_kv(
                    key_states, value_states, self.layer_idx, **kwargs)
                past_key_value.update(key_states_compress, value_states_compress, self.layer_idx, cache_kwargs)
                key_states, value_states = key_states_compress, value_states_compress
            else: # Decoding phase
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
                self.tsp_idx = None
        
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        attn_output = _flash_attention_forward(
            query_states, key_states, value_states, attention_mask, q_len, is_causal=self.is_causal,
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
        cache_position=cache_position,
        **kwargs,
    )
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states
    
    tsp_idx = getattr(self.self_attn, 'tsp_idx', None)
    if self.self_attn.kv_cluster.tsp_layer and tsp_idx is not None:
        self.new_position_ids = torch.gather(position_ids, dim=1, index=tsp_idx)
        tsp_idx_expanded = tsp_idx.unsqueeze(-1).expand(-1, -1, hidden_states.size(2))
        hidden_states = torch.gather(hidden_states, dim=1, index=tsp_idx_expanded)
    else:
        self.new_position_ids = None
    
    outputs = (hidden_states,)
    if use_cache: outputs += (present_key_value,)
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
    # Retrieve oracle_rankings from the model object itself, where the runner script will place it.
    layer_kwargs = {"oracle_rankings": getattr(self, "oracle_rankings", None)}

    # Standard LlamaModel forward logic...
    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)
    if use_cache and not isinstance(past_key_values, Cache):
        past_key_values = DynamicCache.from_legacy_cache(past_key_values)
    if cache_position is None:
        cache_position = torch.arange(
            past_key_values.get_seq_length() if past_key_values is not None else 0,
            inputs_embeds.shape[1], device=inputs_embeds.device
        )
    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions)
    hidden_states = inputs_embeds
    
    next_decoder_cache = None
    for decoder_layer in self.layers:
        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_value=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **layer_kwargs,
        )

        new_position_ids = getattr(decoder_layer, 'new_position_ids', None)
        if new_position_ids is not None:
            position_ids = new_position_ids
            
        hidden_states = layer_outputs[0]
        if use_cache:
            next_decoder_cache = layer_outputs[1]

    hidden_states = self.norm(hidden_states)
    
    # Cut-off hidden states for generation
    if hidden_states.shape[1] > 1:
        hidden_states = hidden_states[:, -1:, :]

    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_decoder_cache,
        hidden_states=None,
        attentions=None,
    )

# =====================================================================================
# 3. PUBLIC API FOR EVALUATION SCRIPTS
# =====================================================================================

def compress(model, args): 
    """Configures the OracleKVCluster on each layer of the model."""
    if not hasattr(model, 'model') or not hasattr(model.model, 'layers'):
        raise TypeError("Model does not have the expected structure with 'model.layers'.")
        
    layers = model.model.layers
    # Use keep_percentage for oracle mode, which is aliased from max_capacity_prompt_percentage
    keep_pct = args.keep_percentage if hasattr(args, 'keep_percentage') else args.max_capacity_prompt_percentage
    
    logging.info(f"OracleKV Configure: layers={len(layers)}, tsp_idx={args.tsp_idx}, keep_percentage={keep_pct}")

    for i, layer in enumerate(layers):
        if not hasattr(layer, 'self_attn') or not hasattr(layer.self_attn, 'kv_cluster'):
             raise TypeError(f"Layer {i} was not patched correctly. It's missing 'self_attn.kv_cluster'.")
             
        cluster = layer.self_attn.kv_cluster
        cluster.max_capacity_prompt_percentage = keep_pct
        cluster.tsp_layer = (i == args.tsp_idx)
        if cluster.tsp_layer:
            logging.info(f"OracleKV: Set layer {i} as TSP layer")

def replace_llama():
    """Applies the OracleKV monkey patches to the Llama model classes."""
    logging.info("Applying OracleKV patch to Llama model classes.")
    LLAMA_ATTENTION_CLASSES['flash_attention_2'] = LlamaOracleKVAttention
    transformers.models.llama.modeling_llama.LlamaDecoderLayer.forward = llama_decoderlayer_forward
    transformers.models.llama.modeling_llama.LlamaModel.forward = llama_model_forward

def replace_mistral():
    """Placeholder for Mistral support. Not yet implemented."""
    logging.warning("OracleKV for Mistral is not implemented. Skipping patch.")
    pass
