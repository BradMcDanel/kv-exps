import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.models.llama.modeling_llama import (
    LlamaAttention as HfLlamaAttention,
    apply_rotary_pos_emb as hf_apply_rotary_pos_emb,
    repeat_kv as hf_repeat_kv
)
from transformers.cache_utils import Cache, DynamicCache
import types
from typing import List, Dict, Tuple, Optional, Any, Union
import math
import argparse
from datasets import load_dataset
import time
from functools import partial

from qtip.model.llama import LlamaAttention as QTipLlamaAttentionEager
from qtip.model.llama import LlamaSdpaAttention as QTipLlamaSdpaAttention
from qtip.model.llama import LlamaFlashAttention2 as QTipLlamaFlashAttention2

# Define all attention types that might be patched
ALL_PATCHABLE_ATTENTION_TYPES = (
    HfLlamaAttention,
    QTipLlamaAttentionEager,
    QTipLlamaSdpaAttention,
    QTipLlamaFlashAttention2
)
# Define SDPA-like attention types specifically
SDPA_LIKE_ATTENTION_TYPES = (
    HfLlamaAttention,
    QTipLlamaSdpaAttention,
    QTipLlamaFlashAttention2
)


def _common_qkv_rope_capture_cache(
    self_attn: Union[HfLlamaAttention, QTipLlamaAttentionEager, QTipLlamaSdpaAttention, QTipLlamaFlashAttention2],
    pipeline_instance: 'SpeculativePrefillPipeline',
    hidden_states: torch.Tensor,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]], # Should always be provided
    past_key_value: Optional[Cache],
    use_cache: bool,
    cache_position: Optional[torch.LongTensor]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:

    batch_size, query_length, _ = hidden_states.size()

    query_projection = self_attn.q_proj(hidden_states)
    key_projection = self_attn.k_proj(hidden_states)
    value_projection = self_attn.v_proj(hidden_states)

    query_states = query_projection.view(batch_size, query_length, self_attn.num_heads, self_attn.head_dim).transpose(1, 2)
    key_states_before_rope = key_projection.view(batch_size, query_length, self_attn.num_key_value_heads, self_attn.head_dim).transpose(1, 2)
    value_states_for_cache = value_projection.view(batch_size, query_length, self_attn.num_key_value_heads, self_attn.head_dim).transpose(1, 2)
    
    if position_embeddings is None:
        raise ValueError("_common_qkv_rope_capture_cache expects position_embeddings to be provided.")
    cos, sin = position_embeddings 

    query_states_rotated, key_states_rotated = hf_apply_rotary_pos_emb(query_states, key_states_before_rope, cos, sin)

    if pipeline_instance.is_generating_lookaheads:
        pipeline_instance.captured_qs[self_attn.layer_idx].append(query_states_rotated.detach().clone())

    key_states_to_use: torch.Tensor
    value_states_to_use: torch.Tensor
    if use_cache:
        if past_key_value is None:
             raise ValueError("past_key_value cannot be None when use_cache is True")
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states_to_use, value_states_to_use = past_key_value.update(key_states_rotated, value_states_for_cache, self_attn.layer_idx, cache_kwargs) # type: ignore
    else:
        key_states_to_use, value_states_to_use = key_states_rotated, value_states_for_cache
    
    return query_states_rotated, key_states_to_use, value_states_to_use, batch_size, query_length


def _patched_sdpa_compatible_attention_forward(
    self_attn: Union[HfLlamaAttention, QTipLlamaSdpaAttention, QTipLlamaFlashAttention2], 
    pipeline_instance: 'SpeculativePrefillPipeline', 
    hidden_states: torch.Tensor, 
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None, 
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False, 
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, 
    **kwargs 
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

    query_states_rotated, key_states_for_sdpa, value_states_for_sdpa, batch_size, query_length = \
        _common_qkv_rope_capture_cache(self_attn, pipeline_instance, hidden_states,
                                       position_embeddings, past_key_value, use_cache, cache_position)

    key_states_for_sdpa = hf_repeat_kv(key_states_for_sdpa, self_attn.num_key_value_groups)
    value_states_for_sdpa = hf_repeat_kv(value_states_for_sdpa, self_attn.num_key_value_groups)

    attn_mask_input = attention_mask
    is_sdpa_causal = (query_length > 1) and (attn_mask_input is None)
    
    if attn_mask_input is not None:
        is_sdpa_causal = False 
        actual_kv_sequence_length = key_states_for_sdpa.shape[-2]
        if attn_mask_input.shape[-1] > actual_kv_sequence_length: 
            attn_mask_input = attn_mask_input[:, :, :, :actual_kv_sequence_length]

    # Contiguous checks might be needed for some torch versions with SDPA
    query_states_sdpa_final = query_states_rotated
    key_states_sdpa_final = key_states_for_sdpa
    value_states_sdpa_final = value_states_for_sdpa
    if query_states_rotated.device.type == "cuda" and attn_mask_input is not None: # As per HF LlamaSdpaAttention
        query_states_sdpa_final = query_states_rotated.contiguous()
        key_states_sdpa_final = key_states_for_sdpa.contiguous()
        value_states_sdpa_final = value_states_for_sdpa.contiguous()

    attention_output = F.scaled_dot_product_attention(
        query_states_sdpa_final, key_states_sdpa_final, value_states_sdpa_final, 
        attn_mask=attn_mask_input, 
        dropout_p=self_attn.attention_dropout if self_attn.training else 0.0, # Use original dropout
        is_causal=is_sdpa_causal, 
        **kwargs # Pass along other sdpa kwargs if any are in the original signature
    )
    
    attention_output = attention_output.transpose(1, 2).contiguous().reshape(batch_size, query_length, self_attn.hidden_size)
    attention_output = self_attn.o_proj(attention_output)
    
    return attention_output, None 


def _patched_qtip_eager_attention_forward(
    self_attn: QTipLlamaAttentionEager, 
    pipeline_instance: 'SpeculativePrefillPipeline', 
    hidden_states: torch.Tensor, 
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None, 
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False, 
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, 
    **kwargs 
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    
    query_states_rotated, key_states_eager, value_states_eager, batch_size, query_length = \
        _common_qkv_rope_capture_cache(self_attn, pipeline_instance, hidden_states,
                                       position_embeddings, past_key_value, use_cache, cache_position)

    key_states_eager = hf_repeat_kv(key_states_eager, self_attn.num_key_value_groups)
    value_states_eager = hf_repeat_kv(value_states_eager, self_attn.num_key_value_groups)
    
    attn_weights = torch.matmul(query_states_rotated, key_states_eager.transpose(2, 3)) / math.sqrt(self_attn.head_dim)

    if attention_mask is not None:
        # qtip LlamaAttention eager: causal_mask = attention_mask[:, :, :, :key_states.shape[-2]]
        # attn_weights = attn_weights + causal_mask
        expected_kv_len = key_states_eager.shape[-2]
        attention_mask_for_eager: Optional[torch.Tensor] = attention_mask

        if attention_mask.shape[-1] > expected_kv_len:
            attention_mask_for_eager = attention_mask[:, :, :, :expected_kv_len]
        elif attention_mask.shape[-1] < expected_kv_len: # Should not happen if mask is prepared correctly
             # Pad with a large negative number if mask is too short (so softmax makes it zero)
             padding_size = expected_kv_len - attention_mask.shape[-1]
             attention_mask_for_eager = F.pad(attention_mask, (0, padding_size), value=torch.finfo(attn_weights.dtype).min)


        if attention_mask_for_eager is not None: # Add mask if it's valid
            attn_weights = attn_weights + attention_mask_for_eager

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states_rotated.dtype)
    attn_weights = F.dropout(attn_weights, p=self_attn.attention_dropout, training=self_attn.training) # Use original dropout
    attention_output = torch.matmul(attn_weights, value_states_eager)
    
    attention_output = attention_output.transpose(1, 2).contiguous().reshape(batch_size, query_length, self_attn.hidden_size)
    attention_output = self_attn.o_proj(attention_output)
    
    return attention_output, None


class SpeculativePrefillPipeline:
    def __init__(self, base_model_name: str, speculator_model_name: Optional[str],
                 share_kv_cache: bool = False):
        self.base_model_name = base_model_name
        self.speculator_model_name = speculator_model_name
        self.share_kv_cache = share_kv_cache
        
        self.base_config = AutoConfig.from_pretrained(self.base_model_name, trust_remote_code=True)
        self.tokenizer = self._load_tokenizer()
        
        self.speculator_model: Optional[AutoModelForCausalLM] = None
        self.device: Optional[torch.device] = None # type: ignore
        self.dtype: Optional[torch.dtype] = None # type: ignore

        if self.speculator_model_name is not None:
            spec_config_name = self.speculator_model_name
            spec_config = AutoConfig.from_pretrained(spec_config_name, trust_remote_code=True)
            self.speculator_model = self._load_model_with_config(spec_config_name, None, spec_config)
            self.device = self.speculator_model.device # type: ignore
            self.dtype = self.speculator_model.dtype # type: ignore
        
        self.base_model = self._load_model_with_config(self.base_model_name, None, self.base_config)

        if self.device is None: 
            self.device = self.base_model.device # type: ignore
            self.dtype = self.base_model.dtype # type: ignore
            
        if self.share_kv_cache and self.speculator_model is not None: 
            self._check_model_compatibility()
        
        self.captured_qs: List[List[torch.Tensor]] = []
        self.orig_spec_fwds: Dict[int, Any] = {}
        self.is_generating_lookaheads = False

    def _check_model_compatibility(self):
        if self.speculator_model is None: return 

        scfg = self.speculator_model.config; bcfg = self.base_model.config
        compatible = (scfg.num_hidden_layers == bcfg.num_hidden_layers and
                      scfg.hidden_size == bcfg.hidden_size and
                      scfg.num_attention_heads == bcfg.num_attention_heads and
                      getattr(scfg, 'num_key_value_heads', scfg.num_attention_heads) == \
                      getattr(bcfg, 'num_key_value_heads', bcfg.num_attention_heads))
        if not compatible: raise ValueError("Models not compatible for KV cache sharing.")

    def _load_tokenizer(self): 
        tok = AutoTokenizer.from_pretrained(self.base_model_name, trust_remote_code=True)
        if tok.pad_token is None and tok.eos_token is not None: 
            tok.pad_token = tok.eos_token
        
        eos_id_val = self.base_config.eos_token_id
        if isinstance(eos_id_val, int):
            self.eos_token_ids = [eos_id_val]
        elif isinstance(eos_id_val, list):
            self.eos_token_ids = list(eos_id_val) 
        elif eos_id_val is None: 
            self.eos_token_ids = [] 
            if self.tokenizer.eos_token_id is not None:
                self.eos_token_ids = [self.tokenizer.eos_token_id]
        else: 
            self.eos_token_ids = [] 
            
        return tok

    def _load_model_with_config(self, model_name: str, attn_impl: Optional[str], config_obj: AutoConfig) -> AutoModelForCausalLM:
        load_kwargs: Dict[str, Any] = {
            "torch_dtype": "float16", # Can be "auto"
            "config": config_obj,
            "trust_remote_code": True,
            "device_map": "auto"
        }
        if attn_impl is not None: # For HF models, this can force "eager", "sdpa", etc.
            load_kwargs["attn_implementation"] = attn_impl
        
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs) # type: ignore
        return model.eval() # type: ignore

    def _patch_speculator(self):
        if self.speculator_model is None: return 0
        if not hasattr(self.speculator_model, 'model') or not hasattr(self.speculator_model.model, 'layers'): # type: ignore
            return 0

        num_layers = self.speculator_model.config.num_hidden_layers
        self.captured_qs = [[] for _ in range(num_layers)]
        num_patched_layers = 0
        for i, layer in enumerate(self.speculator_model.model.layers): # type: ignore
            if not hasattr(layer, 'self_attn'):
                continue
            
            attention_module = layer.self_attn
            chosen_patch_method = None

            if isinstance(attention_module, QTipLlamaAttentionEager):
                chosen_patch_method = _patched_qtip_eager_attention_forward
            elif isinstance(attention_module, SDPA_LIKE_ATTENTION_TYPES):
                chosen_patch_method = _patched_sdpa_compatible_attention_forward
            else:
                continue # Not a known type to patch

            if i not in self.orig_spec_fwds: self.orig_spec_fwds[i] = attention_module.forward
            
            partially_applied_func = partial(chosen_patch_method, pipeline_instance=self)
            attention_module.forward = types.MethodType(partially_applied_func, attention_module)
            num_patched_layers +=1
        return num_patched_layers
    
    def _unpatch_speculator(self):
        if self.speculator_model is None or not self.orig_spec_fwds: return
        if not hasattr(self.speculator_model, 'model') or not hasattr(self.speculator_model.model, 'layers'): # type: ignore
            return

        for i, layer in enumerate(self.speculator_model.model.layers): # type: ignore
            if i in self.orig_spec_fwds and hasattr(layer, 'self_attn') and \
               isinstance(layer.self_attn, ALL_PATCHABLE_ATTENTION_TYPES):
                layer.self_attn.forward = self.orig_spec_fwds[i]
        self.orig_spec_fwds.clear()


    def run(self, prompt_text: str, look_ahead_k: int,
            prompt_keep_percentage: float, max_generation_length: int) -> Tuple[str, Dict[str, float]]:
        
        timing_info: Dict[str, float] = {}
        overall_start_time = time.perf_counter()

        num_patched_layers = 0
        if self.speculator_model is not None:
            num_patched_layers = self._patch_speculator()

        limit_due_to_base = self.base_model.config.max_position_embeddings - max_generation_length
        max_prompt_len_calculated: float
        if self.speculator_model is not None:
            limit_due_to_speculator = self.speculator_model.config.max_position_embeddings - look_ahead_k
            max_prompt_len_calculated = float(min(limit_due_to_base, limit_due_to_speculator) - 20)
        else:
            max_prompt_len_calculated = float(limit_due_to_base - 20)
        max_prompt_length = max(1, int(max_prompt_len_calculated))
        
        inputs = self.tokenizer(prompt_text, return_tensors="pt", padding=False, truncation=True, max_length=max_prompt_length).to(self.device)
        prompt_input_ids, prompt_length, batch_size = inputs.input_ids, inputs.input_ids.shape[1], inputs.input_ids.shape[0]

        if prompt_length == 0: 
            timing_info["total_time"] = time.perf_counter() - overall_start_time
            if num_patched_layers > 0: self._unpatch_speculator()
            return "", timing_info

        speculator_prefill_cache: Optional[Cache] = None
        speculator_prefill_cache_as_tuple: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None
        speculator_next_token_ids: Optional[torch.Tensor] = None
        
        speculation_prefill_time = 0.0
        if self.speculator_model is not None:
            stage_start_time = time.perf_counter()
            self.is_generating_lookaheads = False 
            speculator_prompt_cache_position = torch.arange(prompt_length, device=self.device)
            # Position IDs for prefill must be passed for LlamaModel to compute RoPE correctly
            speculator_prompt_position_ids = torch.arange(0, prompt_length, device=self.device).unsqueeze(0)

            with torch.no_grad():
                speculator_prefill_output = self.speculator_model(
                    input_ids=prompt_input_ids, 
                    position_ids=speculator_prompt_position_ids, # Crucial for RoPE in LlamaModel
                    use_cache=True, 
                    cache_position=speculator_prompt_cache_position
                )
            
            speculator_prefill_cache = speculator_prefill_output.past_key_values # type: ignore 
            if speculator_prefill_cache is not None and hasattr(speculator_prefill_cache, 'to_legacy_cache'): 
                 speculator_prefill_cache_as_tuple = speculator_prefill_cache.to_legacy_cache()

            speculator_next_token_ids = torch.argmax(speculator_prefill_output.logits[:, -1, :], dim=-1, keepdim=True)
            if num_patched_layers > 0: 
                for q_list in self.captured_qs: q_list.clear()
            speculation_prefill_time = time.perf_counter() - stage_start_time
        timing_info["speculation_prefill"] = speculation_prefill_time

        generated_speculator_ids = []
        current_speculator_cache: Optional[Cache] = speculator_prefill_cache
        speculation_decode_time = 0.0

        if self.speculator_model is not None and num_patched_layers > 0 and look_ahead_k > 0 and \
           speculator_next_token_ids is not None and current_speculator_cache is not None:
            stage_start_time = time.perf_counter()
            self.is_generating_lookaheads = True
            current_speculator_token_ids = speculator_next_token_ids
            current_speculator_position = prompt_length # Actual position for RoPE

            for _ in range(look_ahead_k):
                current_cache_len = current_speculator_cache.get_seq_length(0) # type: ignore
                lookahead_cache_position = torch.tensor([current_cache_len], device=self.device, dtype=torch.long)
                lookahead_position_ids = torch.tensor([[current_speculator_position]], device=self.device, dtype=torch.long)

                with torch.no_grad():
                    lookahead_output = self.speculator_model(
                        input_ids=current_speculator_token_ids, 
                        position_ids=lookahead_position_ids, # Crucial for RoPE
                        past_key_values=current_speculator_cache, 
                        use_cache=True, 
                        cache_position=lookahead_cache_position
                    )
                current_speculator_cache = lookahead_output.past_key_values # type: ignore 
                current_speculator_token_ids = torch.argmax(lookahead_output.logits[:, -1, :], dim=-1, keepdim=True)
                token_id = current_speculator_token_ids.item()
                generated_speculator_ids.append(token_id)
                current_speculator_position += 1
                if token_id in self.eos_token_ids: break
            self.is_generating_lookaheads = False
            speculation_decode_time = time.perf_counter() - stage_start_time
        timing_info["speculation_decode"] = speculation_decode_time
        
        importance_scores = torch.zeros(batch_size, prompt_length, device=self.device, dtype=self.dtype)
        num_lookahead_steps = len(generated_speculator_ids)
        
        if self.speculator_model is not None and num_lookahead_steps > 0 and num_patched_layers > 0 and \
           speculator_prefill_cache_as_tuple is not None and hasattr(self.speculator_model.model, 'layers'): # type: ignore
            
            example_attn_layer_obj = self.speculator_model.model.layers[0].self_attn # type: ignore
            if isinstance(example_attn_layer_obj, ALL_PATCHABLE_ATTENTION_TYPES): 
                head_dim = example_attn_layer_obj.head_dim
                num_kv_groups = example_attn_layer_obj.num_key_value_groups 
                for layer_idx in range(self.speculator_model.config.num_hidden_layers):
                    if layer_idx >= len(speculator_prefill_cache_as_tuple) or \
                       layer_idx >= len(self.captured_qs): # Safety check
                        continue
                    key_layer_prompt = speculator_prefill_cache_as_tuple[layer_idx][0].detach()
                    key_layer_prompt_repeated = hf_repeat_kv(key_layer_prompt, num_kv_groups)
                    for spec_idx in range(num_lookahead_steps):
                        if spec_idx < len(self.captured_qs[layer_idx]): 
                            query_speculator_lookahead = self.captured_qs[layer_idx][spec_idx]
                            # Q shape: (bs, n_heads, 1, head_dim), K_T shape: (bs, n_heads, head_dim, prompt_len)
                            logits = torch.matmul(query_speculator_lookahead, key_layer_prompt_repeated.transpose(-1, -2)) / math.sqrt(head_dim)
                            # logits shape: (bs, n_heads, 1, prompt_len) -> sum over heads -> (bs, 1, prompt_len)
                            importance_scores += logits.sum(dim=1).squeeze(dim=1) 

        sorted_top_k_indices: torch.Tensor
        if self.speculator_model is not None and num_lookahead_steps > 0 and num_patched_layers > 0:
            num_tokens_to_keep_from_prompt = max(1, math.ceil(prompt_length * prompt_keep_percentage))
            num_top_k_to_select = min(int(num_tokens_to_keep_from_prompt), prompt_length)
            
            if num_top_k_to_select > 0 and importance_scores.sum().item() != 0: 
                 _, top_k_indices = torch.topk(importance_scores[0], k=num_top_k_to_select)
                 sorted_top_k_indices = torch.sort(top_k_indices)[0]
            else: 
                sorted_top_k_indices = torch.empty(0, dtype=torch.long, device=self.device)
        else:
            sorted_top_k_indices = torch.arange(prompt_length, device=self.device) if prompt_length > 0 else torch.empty(0, dtype=torch.long, device=self.device)


        base_model_first_token_gen_start_time = time.perf_counter()
        base_model_cache_after_prefill: Optional[Cache] = None
        base_model_next_token_ids: Optional[torch.Tensor] = None

        base_prefill_position_ids = torch.arange(0, prompt_input_ids.shape[1], device=self.device).unsqueeze(0)

        if self.speculator_model is None: 
            base_prefill_cache_position = torch.arange(prompt_length, device=self.device)
            with torch.no_grad():
                base_prefill_output = self.base_model(
                    input_ids=prompt_input_ids, 
                    position_ids=base_prefill_position_ids,
                    use_cache=True, 
                    cache_position=base_prefill_cache_position
                )
            base_model_next_token_ids = torch.argmax(base_prefill_output.logits[:, -1, :], dim=-1, keepdim=True)
            base_model_cache_after_prefill = base_prefill_output.past_key_values # type: ignore
        
        elif self.share_kv_cache: 
            pruned_kv_cache = DynamicCache()
            n_pruned_tokens_for_cache = 0
            if len(sorted_top_k_indices) > 0 and speculator_prefill_cache is not None and \
               hasattr(speculator_prefill_cache, 'key_cache') and hasattr(speculator_prefill_cache, 'value_cache'): # Check if DynamicCache
                for layer_idx in range(self.base_model.config.num_hidden_layers): 
                    pruned_key = speculator_prefill_cache.key_cache[layer_idx][:, :, sorted_top_k_indices, :] # type: ignore
                    pruned_value = speculator_prefill_cache.value_cache[layer_idx][:, :, sorted_top_k_indices, :] # type: ignore
                    pruned_kv_cache.update(pruned_key, pruned_value, layer_idx)
                n_pruned_tokens_for_cache = len(sorted_top_k_indices)
            
            knockout_token_ids = prompt_input_ids[:, -1:]
            knockout_position_ids = torch.tensor([[prompt_length - 1]], device=self.device, dtype=torch.long)
            knockout_cache_position = torch.tensor([n_pruned_tokens_for_cache], device=self.device, dtype=torch.long)
            with torch.no_grad():
                knockout_output = self.base_model(
                    knockout_token_ids, 
                    position_ids=knockout_position_ids, 
                    past_key_values=pruned_kv_cache, 
                    use_cache=True, 
                    cache_position=knockout_cache_position
                )
            base_model_next_token_ids = torch.argmax(knockout_output.logits[:, -1, :], dim=-1, keepdim=True)
            base_model_cache_after_prefill = knockout_output.past_key_values # type: ignore
        
        else: # No KV sharing, base model does its own selective prefill
            selective_prefill_cache = DynamicCache()
            # If sorted_top_k_indices is empty (e.g. no useful speculation), prefill with full prompt
            # or a subset like last N tokens + first token, or just last token (knockout).
            # For simplicity here, if it's empty from speculation, we'll prefill with full prompt.
            # Or, more aligned with the paper's spirit, prefill with the *selected* tokens.
            # If sorted_top_k_indices is empty due to no speculation, then this becomes full prompt.
            # The original code had: if len(sorted_top_k_indices) > 0
            # Let's ensure sorted_top_k_indices is never empty if prompt_length > 0 for this branch.
            # Default to all prompt tokens if no better selection.
            # The line `sorted_top_k_indices = torch.arange(prompt_length, device=self.device) if prompt_length > 0 else torch.empty(0, dtype=torch.long, device=self.device)`
            # above should handle this for the `else` of speculator existing.

            if len(sorted_top_k_indices) > 0 :
                selected_prompt_ids = prompt_input_ids[:, sorted_top_k_indices]
                selected_position_ids = sorted_top_k_indices.unsqueeze(0) # Original positions
                # Cache positions are sequential for the items being added to *this* cache
                selective_prefill_cache_position = torch.arange(selected_prompt_ids.shape[1], device=self.device)
                with torch.no_grad():
                    selective_prefill_output = self.base_model(
                        selected_prompt_ids, 
                        position_ids=selected_position_ids, 
                        past_key_values=selective_prefill_cache, # starts empty
                        use_cache=True, 
                        cache_position=selective_prefill_cache_position
                    )
                # The next token is predicted based on the *last token processed* in this selective prefill
                base_model_next_token_ids = torch.argmax(selective_prefill_output.logits[:, -1, :], dim=-1, keepdim=True)
                base_model_cache_after_prefill = selective_prefill_output.past_key_values # type: ignore
            else: # Should not happen if prompt_length > 0 due to default above. This is a fallback.
                 base_prefill_cache_position = torch.arange(prompt_length, device=self.device)
                 with torch.no_grad():
                     base_prefill_output = self.base_model(
                         input_ids=prompt_input_ids, 
                         position_ids=base_prefill_position_ids, # Full prompt positions
                         use_cache=True, 
                         cache_position=base_prefill_cache_position, 
                         past_key_values=selective_prefill_cache # starts empty
                    ) 
                 base_model_next_token_ids = torch.argmax(base_prefill_output.logits[:, -1, :], dim=-1, keepdim=True)
                 base_model_cache_after_prefill = base_prefill_output.past_key_values # type: ignore


        base_model_first_token_gen_time = time.perf_counter() - base_model_first_token_gen_start_time
        timing_info["base_ttft"] = speculation_prefill_time + speculation_decode_time + base_model_first_token_gen_time
        
        generated_token_ids: List[int] = []
        final_generated_text = ""
        
        if base_model_next_token_ids is not None and base_model_cache_after_prefill is not None:
            first_generated_token_id = base_model_next_token_ids.item(); generated_token_ids.append(first_generated_token_id)
            
            current_decode_token_ids = base_model_next_token_ids
            current_decode_cache: Cache = base_model_cache_after_prefill
            
            # Determine actual position for RoPE in decode loop.
            # If selective prefill was used, the last *actual* position processed by base model determines the start.
            # If share_kv_cache, last actual pos was prompt_length -1.
            # If no speculator, last actual pos was prompt_length -1.
            # For simplicity, we assume the first generated token is at prompt_length
            current_real_position = prompt_length 
            
            current_cache_write_position = current_decode_cache.get_seq_length(0) # Position in the *current_decode_cache*

            if first_generated_token_id not in self.eos_token_ids:
                for _ in range(max_generation_length - 1):
                    decode_position_ids = torch.tensor([[current_real_position]], device=self.device, dtype=torch.long)
                    decode_cache_position = torch.tensor([current_cache_write_position], device=self.device, dtype=torch.long)
                    with torch.no_grad():
                        decode_output = self.base_model(
                            current_decode_token_ids, 
                            position_ids=decode_position_ids, 
                            past_key_values=current_decode_cache, 
                            use_cache=True, 
                            cache_position=decode_cache_position
                        )
                    next_base_token_ids = torch.argmax(decode_output.logits[:, -1, :], dim=-1, keepdim=True)
                    next_base_token_id = next_base_token_ids.item()
                    generated_token_ids.append(next_base_token_id)
                    
                    current_decode_token_ids = next_base_token_ids
                    current_decode_cache = decode_output.past_key_values # type: ignore
                    
                    current_real_position += 1
                    current_cache_write_position +=1 
                    
                    if next_base_token_id in self.eos_token_ids: break
            
            final_generated_text = self.tokenizer.decode(generated_token_ids, skip_special_tokens=True)
        
        if num_patched_layers > 0:
            self._unpatch_speculator()
            
        timing_info["total_time"] = time.perf_counter() - overall_start_time
        
        reported_timing_info = {
            "speculation_prefill": timing_info.get("speculation_prefill", 0.0),
            "speculation_decode": timing_info.get("speculation_decode", 0.0),
            "base_ttft": timing_info.get("base_ttft", 0.0),
            "total_time": timing_info.get("total_time", 0.0)
        }
        return final_generated_text, reported_timing_info

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Speculative Prefill Pipeline")
    parser.add_argument("--base_model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct") # Changed default for example
    parser.add_argument("--speculator_model_name", type=str, default=None, help="Path to speculator model. If None, speculative part is skipped.")
    parser.add_argument("--dataset_name", type=str, default="hotpotqa") 
    parser.add_argument("--look_ahead_k", type=int, default=1)
    parser.add_argument("--prompt_keep_percentage", type=float, default=0.2)
    parser.add_argument("--max_generation_length", type=int, default=32)
    parser.add_argument("--share_kv_cache", action="store_true", default=False)
    args = parser.parse_args()

    pipeline = SpeculativePrefillPipeline(
        base_model_name=args.base_model_name, 
        speculator_model_name=args.speculator_model_name,
        share_kv_cache=args.share_kv_cache)

    prompt_to_run_str: str
    if args.dataset_name and args.dataset_name.lower() != 'none':
        # Using a different dataset split that's commonly available for hotpotqa
        try:
            dataset = load_dataset(args.dataset_name, 'fullwiki', split='validation') # type: ignore
            sample = dataset[0] # type: ignore
            # HotpotQA structure: question, context (often a list of paragraphs)
            # Let's simplify for the prompt template
            context_str = ""
            if 'context' in sample and sample['context'] and 'sentences' in sample['context']: # type: ignore
                # Join paragraphs if context is structured
                context_str = " ".join([" ".join(para_sentences) for para_sentences in sample['context']['sentences']]) # type: ignore
            elif isinstance(sample.get('context'), str): # type: ignore
                context_str = sample.get('context', '') # type: ignore

            messages = [
                {"role": "user", "content": f"Context: {context_str}\nQuestion: {sample.get('question', '')}\nAnswer:"} # type: ignore
            ]
            prompt_to_run_str = pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) # type: ignore
        except Exception as e:
            print(f"Failed to load dataset {args.dataset_name}: {e}. Using default prompt.")
            messages = [{"role": "user", "content": "Explain the theory of relativity in simple terms."}]
            prompt_to_run_str = pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) # type: ignore
    else:
        messages = [
            {"role": "user", "content": "Explain the theory of relativity in simple terms."}
        ]
        prompt_to_run_str = pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) # type: ignore
    
    print(f"\n--- Running with Prompt (first 100 chars) ---")
    print(f"{prompt_to_run_str[:100]}...")

    generated_text, run_timing_info = pipeline.run(
        prompt_text=prompt_to_run_str, 
        look_ahead_k=args.look_ahead_k,
        prompt_keep_percentage=args.prompt_keep_percentage, 
        max_generation_length=args.max_generation_length
    )
    
    print(f"\n--- Generated Output ({len(pipeline.tokenizer.encode(generated_text))} tokens) ---")
    print(f"{generated_text}")

    print(f"\n--- Run Timing Information (seconds) ---")
    for stage, duration in run_timing_info.items():
        print(f"  {stage}: {duration:.4f}")
