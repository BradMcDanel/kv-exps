import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
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
import sys
import os

# Add QTIP to path if needed
sys.path.append(os.path.join(os.path.dirname(__file__), 'qtip')) # Assuming QTIP is in qtip subdir

try:
    from lib.linear import QuantizedLinear # type: ignore
    from lib.utils.unsafe_import import model_from_hf_path as qtip_model_from_hf_path # type: ignore
    from model.llama import LlamaAttention as QTIP_LlamaAttention # type: ignore
    QTIP_AVAILABLE = True
except ImportError:
    # print("Warning: QTIP modules not found. QTIP model support disabled.")
    QTIP_AVAILABLE = False
    QuantizedLinear = None # Placeholder
    qtip_model_from_hf_path = None # Placeholder
    QTIP_LlamaAttention = None # Placeholder


# Patched attention forward methods (unchanged from original)
def _hf_patched_attention_forward_method(
    self_attn: LlamaAttention,
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

    batch_size, query_length, _ = hidden_states.size()
    query_projection = self_attn.q_proj(hidden_states)
    key_projection = self_attn.k_proj(hidden_states)
    value_projection = self_attn.v_proj(hidden_states)
    query_states = query_projection.view(batch_size, query_length, self_attn.config.num_attention_heads, self_attn.head_dim).transpose(1, 2)
    key_states_before_rope = key_projection.view(batch_size, query_length, self_attn.config.num_key_value_heads, self_attn.head_dim).transpose(1, 2)
    value_states_for_cache = value_projection.view(batch_size, query_length, self_attn.config.num_key_value_heads, self_attn.head_dim).transpose(1, 2)
    cos, sin = position_embeddings # type: ignore
    query_states_rotated, key_states_rotated = hf_apply_rotary_pos_emb(query_states, key_states_before_rope, cos, sin)
    if pipeline_instance.is_generating_lookaheads:
        pipeline_instance.captured_qs[self_attn.layer_idx].append(query_states_rotated.detach().clone())
    if use_cache:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states_for_sdpa, value_states_for_sdpa = past_key_value.update(key_states_rotated, value_states_for_cache, self_attn.layer_idx, cache_kwargs) # type: ignore
    else:
        key_states_for_sdpa, value_states_for_sdpa = key_states_rotated, value_states_for_cache
    key_states_for_sdpa = hf_repeat_kv(key_states_for_sdpa, self_attn.num_key_value_groups)
    value_states_for_sdpa = hf_repeat_kv(value_states_for_sdpa, self_attn.num_key_value_groups)
    attn_mask_input = attention_mask
    is_sdpa_causal = (query_length > 1) and (attn_mask_input is None)
    if attn_mask_input is not None:
        is_sdpa_causal = False
        actual_kv_sequence_length = key_states_for_sdpa.shape[-2]
        if attn_mask_input.shape[-1] > actual_kv_sequence_length:
            attn_mask_input = attn_mask_input[:, :, :, :actual_kv_sequence_length]
    attention_output = F.scaled_dot_product_attention(
        query_states_rotated, key_states_for_sdpa, value_states_for_sdpa, attn_mask=attn_mask_input, dropout_p=0.0, is_causal=is_sdpa_causal, **kwargs
    )
    attention_output = attention_output.transpose(1, 2).contiguous().reshape(batch_size, query_length, self_attn.o_proj.in_features)
    attention_output = self_attn.o_proj(attention_output)
    return attention_output, None

def _qtip_patched_attention_forward_method(
    self_attn: Any, pipeline_instance: 'SpeculativePrefillPipeline', hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None, position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None, output_attentions: bool = False, use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None, position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
    batch_size, query_length, _ = hidden_states.size()
    query_projection = self_attn.q_proj(hidden_states)
    key_projection = self_attn.k_proj(hidden_states)
    value_projection = self_attn.v_proj(hidden_states)
    query_states = query_projection.view(batch_size, query_length, self_attn.num_heads, self_attn.head_dim).transpose(1, 2)
    key_states_before_rope = key_projection.view(batch_size, query_length, self_attn.num_key_value_heads, self_attn.head_dim).transpose(1, 2)
    value_states_for_cache = value_projection.view(batch_size, query_length, self_attn.num_key_value_heads, self_attn.head_dim).transpose(1, 2)
    cos, sin = position_embeddings # type: ignore
    query_states_rotated, key_states_rotated = hf_apply_rotary_pos_emb(query_states, key_states_before_rope, cos, sin)
    if pipeline_instance.is_generating_lookaheads:
        pipeline_instance.captured_qs[self_attn.layer_idx].append(query_states_rotated.detach().clone())
    if use_cache:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states_for_sdpa, value_states_for_sdpa = past_key_value.update(key_states_rotated, value_states_for_cache, self_attn.layer_idx, cache_kwargs) # type: ignore
    else:
        key_states_for_sdpa, value_states_for_sdpa = key_states_rotated, value_states_for_cache
    key_states_for_sdpa = hf_repeat_kv(key_states_for_sdpa, self_attn.num_key_value_groups)
    value_states_for_sdpa = hf_repeat_kv(value_states_for_sdpa, self_attn.num_key_value_groups)
    attn_mask_input = attention_mask
    is_sdpa_causal = (query_length > 1) and (attn_mask_input is None)
    if attn_mask_input is not None:
        is_sdpa_causal = False
        actual_kv_sequence_length = key_states_for_sdpa.shape[-2]
        if attn_mask_input.shape[-1] > actual_kv_sequence_length:
            attn_mask_input = attn_mask_input[:, :, :, :actual_kv_sequence_length]
    attention_output = F.scaled_dot_product_attention(
        query_states_rotated, key_states_for_sdpa, value_states_for_sdpa, attn_mask=attn_mask_input, dropout_p=0.0, is_causal=is_sdpa_causal, **kwargs
    )
    attention_output = attention_output.transpose(1, 2).contiguous().reshape(batch_size, query_length, self_attn.hidden_size)
    attention_output = self_attn.o_proj(attention_output)
    return attention_output, None, past_key_value


class SpeculativePrefillPipeline:
    def __init__(self, base_model_name: str,
                 speculator_model_name: Optional[str],
                 share_kv_cache: bool = False,
                 kv_sharing_granularity: str = "global"):
        self.base_model_name = base_model_name
        self.speculator_model_name = speculator_model_name
        self.share_kv_cache = share_kv_cache
        self.kv_sharing_granularity = kv_sharing_granularity

        if self.kv_sharing_granularity not in ["global", "layer", "head"]:
            raise ValueError(f"Unsupported kv_sharing_granularity: {self.kv_sharing_granularity}. Must be 'global', 'layer', or 'head'.")
        if not self.share_kv_cache and self.kv_sharing_granularity in ["layer", "head"]:
            raise ValueError(f"kv_sharing_granularity='{self.kv_sharing_granularity}' is specified, but share_kv_cache=False. ")

        self.tokenizer = self._load_tokenizer()
        self.speculator_model: Optional[AutoModelForCausalLM] = None
        self.device: Optional[torch.device] = None
        self.dtype: Optional[torch.dtype] = None

        if self.speculator_model_name is not None and self.speculator_model_name.lower() != "none":
            self.speculator_model = self._load_model(self.speculator_model_name)
            self.device = self.speculator_model.device
            self.dtype = self.speculator_model.dtype

        self.base_model = self._load_model(self.base_model_name)
        self.base_config: AutoConfig = self.base_model.config # type: ignore

        eos_id_val = self.base_config.eos_token_id
        if isinstance(eos_id_val, int): self.eos_token_ids = [eos_id_val]
        elif isinstance(eos_id_val, list): self.eos_token_ids = list(eos_id_val) # type: ignore
        elif eos_id_val is None: self.eos_token_ids = []
        else: self.eos_token_ids = []

        if self.device is None: self.device = self.base_model.device
        if self.dtype is None: self.dtype = self.base_model.dtype

        if self.share_kv_cache and self.speculator_model is not None:
            self._check_model_compatibility()

        self.captured_qs: List[List[torch.Tensor]] = []
        self.orig_spec_fwds: Dict[int, Any] = {}
        self.is_generating_lookaheads = False

        self.global_qk_scores: Optional[torch.Tensor] = None
        self.layer_qk_scores: List[torch.Tensor] = []
        self.head_qk_scores: List[torch.Tensor] = []


    def _load_model(self, model_name: str) -> AutoModelForCausalLM:
        is_qtip_model = False
        if QTIP_AVAILABLE and qtip_model_from_hf_path is not None and ("relaxml" in model_name.lower() or "qtip" in model_name.lower()):
            is_qtip_model = True
        
        if is_qtip_model:
            model, _ = qtip_model_from_hf_path(model_name, max_mem_ratio=0.7, attn_implementation="sdpa") # type: ignore
            return model.eval() # type: ignore
        else:
            load_kwargs: Dict[str, Any] = {"torch_dtype": torch.float16, "trust_remote_code": True, "device_map": "auto"}
            model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
            return model.eval()

    def _check_model_compatibility(self):
        if self.speculator_model is None: return
        scfg = self.speculator_model.config
        bcfg = self.base_model.config
        spec_num_heads = getattr(scfg, 'num_attention_heads', getattr(scfg, 'num_heads', None))
        base_num_heads = getattr(bcfg, 'num_attention_heads', getattr(bcfg, 'num_heads', None))
        spec_num_kv_heads = getattr(scfg, 'num_key_value_heads', spec_num_heads)
        base_num_kv_heads = getattr(bcfg, 'num_key_value_heads', base_num_heads)
        compatible = (
            getattr(scfg, 'num_hidden_layers', None) == getattr(bcfg, 'num_hidden_layers', None) and
            getattr(scfg, 'hidden_size', None) == getattr(bcfg, 'hidden_size', None) and
            spec_num_heads == base_num_heads and spec_num_kv_heads == base_num_kv_heads )
        if not compatible: raise ValueError("Models not compatible for KV cache sharing.")

    def _load_tokenizer(self): # Unchanged from original logic
        try: tok = AutoTokenizer.from_pretrained(self.base_model_name, trust_remote_code=True)
        except OSError:
            if "qtip" in self.base_model_name.lower():
                if "llama-3.1" in self.base_model_name.lower(): fallback_name = "meta-llama/Llama-3.1-8B-Instruct"
                elif "llama-3" in self.base_model_name.lower(): fallback_name = "meta-llama/Meta-Llama-3-8B-Instruct"
                elif "llama-2" in self.base_model_name.lower(): fallback_name = "meta-llama/Llama-2-7b-hf"
                else: fallback_name = "meta-llama/Llama-2-7b-hf" # Default fallback for qtip if specific llama not matched
                tok = AutoTokenizer.from_pretrained(fallback_name, trust_remote_code=True)
            else: raise
        if tok.pad_token is None and tok.eos_token is not None: tok.pad_token = tok.eos_token
        if tok.chat_template is None: # Set default chat template if not present
            if "llama-3" in self.base_model_name.lower() or ("qtip" in self.base_model_name.lower() and ("llama-3" in self.base_model_name.lower() or "Llama-3" in self.base_model_name)):
                 tok.chat_template = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
            elif "llama-2" in self.base_model_name.lower() or ("qtip" in self.base_model_name.lower() and ("llama-2" in self.base_model_name.lower() or "Llama-2" in self.base_model_name)):
                 tok.chat_template = "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 and system_message != false %}{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}"
            else: # A more generic fallback
                 tok.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        return tok

    def _patch_speculator(self):
        if self.speculator_model is None: return 0
        if not hasattr(self.speculator_model, 'model') or not hasattr(self.speculator_model.model, 'layers'): return 0
        num_layers = self.speculator_model.config.num_hidden_layers
        self.captured_qs = [[] for _ in range(num_layers)]
        self.global_qk_scores = None
        self.layer_qk_scores = []
        self.head_qk_scores = []

        num_patched_layers = 0
        for i, layer in enumerate(self.speculator_model.model.layers): # type: ignore
            if hasattr(layer, 'self_attn') and isinstance(layer.self_attn, LlamaAttention):
                attn = layer.self_attn
                if i not in self.orig_spec_fwds: self.orig_spec_fwds[i] = attn.forward
                attn.forward = types.MethodType(partial(_hf_patched_attention_forward_method, pipeline_instance=self), attn)
                num_patched_layers += 1
            elif QTIP_AVAILABLE and QTIP_LlamaAttention is not None and hasattr(layer, 'self_attn') and isinstance(layer.self_attn, QTIP_LlamaAttention):
                attn = layer.self_attn
                if i not in self.orig_spec_fwds: self.orig_spec_fwds[i] = attn.forward
                attn.forward = types.MethodType(partial(_qtip_patched_attention_forward_method, pipeline_instance=self), attn) # type: ignore
                num_patched_layers += 1
        return num_patched_layers

    def _compute_granular_qk_scores(self,
                                   speculator_prefill_cache_as_tuple: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
                                   prompt_length: int,
                                   batch_size: int):
        if self.speculator_model is None or not self.captured_qs or not any(self.captured_qs):
            return

        spec_config = self.speculator_model.config
        num_spec_layers = spec_config.num_hidden_layers
        
        spec_num_q_heads = getattr(spec_config, 'num_attention_heads', getattr(spec_config, 'num_heads', None))
        spec_num_kv_heads = getattr(spec_config, 'num_key_value_heads', spec_num_q_heads)
        if spec_num_q_heads is None or spec_num_kv_heads is None:
            print("Warning: Could not determine head counts for speculator. Skipping QK score calculation.")
            return
        spec_num_kv_groups = spec_num_q_heads // spec_num_kv_heads

        self.layer_qk_scores = [
            torch.zeros(batch_size, prompt_length, device=self.device, dtype=self.dtype)
            for _ in range(num_spec_layers)
        ]
        self.head_qk_scores = [
            torch.zeros(batch_size, spec_num_kv_heads, prompt_length, device=self.device, dtype=self.dtype)
            for _ in range(num_spec_layers)
        ]

        for layer_idx in range(num_spec_layers):
            if not self.captured_qs[layer_idx] or speculator_prefill_cache_as_tuple[layer_idx][0].numel() == 0:
                continue

            key_prompt_layer_spec = speculator_prefill_cache_as_tuple[layer_idx][0].detach()
            key_prompt_layer_rep_spec = hf_repeat_kv(key_prompt_layer_spec, spec_num_kv_groups)
            
            if not self.captured_qs[layer_idx]: continue
            head_dim_from_q = self.captured_qs[layer_idx][0].shape[-1]

            for spec_idx in range(len(self.captured_qs[layer_idx])):
                query_lookahead_layer = self.captured_qs[layer_idx][spec_idx]
                attn_logits_layer_step = torch.matmul(query_lookahead_layer, key_prompt_layer_rep_spec.transpose(-1, -2)) / math.sqrt(head_dim_from_q)
                attn_scores_layer_step = attn_logits_layer_step.squeeze(2) 

                layer_total_scores_step = attn_scores_layer_step.sum(dim=1) 
                self.layer_qk_scores[layer_idx] += layer_total_scores_step

                for kv_head_idx in range(spec_num_kv_heads):
                    start_q_head_idx = kv_head_idx * spec_num_kv_groups
                    end_q_head_idx = (kv_head_idx + 1) * spec_num_kv_groups
                    kv_head_specific_scores_step = attn_scores_layer_step[:, start_q_head_idx:end_q_head_idx, :].sum(dim=1)
                    self.head_qk_scores[layer_idx][:, kv_head_idx, :] += kv_head_specific_scores_step
        
        if self.layer_qk_scores:
            valid_layer_scores = [ls for ls in self.layer_qk_scores if ls.numel() > 0 and ls.shape[0] == batch_size and ls.shape[-1] == prompt_length]
            if valid_layer_scores:
                self.global_qk_scores = torch.sum(torch.stack(valid_layer_scores), dim=0)
            else:
                self.global_qk_scores = torch.zeros(batch_size, prompt_length, device=self.device, dtype=self.dtype)
        else:
            self.global_qk_scores = torch.zeros(batch_size, prompt_length, device=self.device, dtype=self.dtype)

    def _get_sorted_top_k_indices(self, scores: torch.Tensor, num_to_keep: int, current_seq_len: int, device: torch.device) -> torch.Tensor:
        if current_seq_len == 0:
            return torch.empty(0, dtype=torch.long, device=device)

        # Ensure num_to_keep is valid for the current sequence length
        if num_to_keep == 0 and current_seq_len > 0: # Avoid pruning to zero if possible
            num_to_keep = 1 
        num_to_keep = min(num_to_keep, current_seq_len)
        
        if num_to_keep == 0 : # Only if current_seq_len was also 0 initially or num_to_keep target was 0
             return torch.empty(0, dtype=torch.long, device=device)


        if scores.ndim > 1: scores = scores.squeeze() # Assumes it becomes 1D after squeeze for selection
        if scores.shape[0] != current_seq_len:
            raise ValueError(f"Score length {scores.shape[0]} does not match current_seq_len {current_seq_len}")

        if num_to_keep >= current_seq_len:
            return torch.arange(current_seq_len, device=device, dtype=torch.long)
        
        _, top_k_indices = torch.topk(scores, k=num_to_keep)
        return torch.sort(top_k_indices)[0]

    def _prune_kv_cache(self,
                           spec_cache_tuple: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
                           granularity: str,
                           prompt_keep_percentage: float,
                           target_dtype: torch.dtype,
                           device: torch.device
                          ) -> Tuple[Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]], int]:
        if not spec_cache_tuple or spec_cache_tuple[0][0].numel() == 0:
            return None, 0
        
        batch_size = spec_cache_tuple[0][0].shape[0] 
        original_seq_len = spec_cache_tuple[0][0].shape[2]
        if original_seq_len == 0: return spec_cache_tuple, 0

        num_tokens_to_keep_target = math.ceil(original_seq_len * prompt_keep_percentage)
        if original_seq_len > 0 and num_tokens_to_keep_target == 0: # Ensure at least 1 token is kept if possible
            num_tokens_to_keep_target = 1
        num_tokens_to_keep_target = min(num_tokens_to_keep_target, original_seq_len) # Cap at original length

        pruned_kv_list: List[Tuple[torch.Tensor, torch.Tensor]] = []
        
        if granularity == "global":
            if not (self.global_qk_scores is not None and 
                    self.global_qk_scores.shape[0] == batch_size and 
                    self.global_qk_scores.shape[-1] == original_seq_len):
                raise ValueError(f"Global QK scores shape ({self.global_qk_scores.shape if self.global_qk_scores is not None else 'None'}) is unsuitable for 'global' pruning. Expected bsz={batch_size}, seq_len={original_seq_len}.")
            if self.global_qk_scores.sum().item() == 0 and num_tokens_to_keep_target < original_seq_len:
                raise ValueError("Global QK scores are all zero, cannot perform meaningful 'global' pruning when prompt_keep_percentage < 1.0.")
            
            global_scores_for_selection = self.global_qk_scores[0] # Assuming batch_size 1 for selection logic
            sorted_indices = self._get_sorted_top_k_indices(global_scores_for_selection, num_tokens_to_keep_target, original_seq_len, device)
            
            for k_l_spec, v_l_spec in spec_cache_tuple:
                if k_l_spec.shape[2] == 0: 
                    pruned_kv_list.append((k_l_spec.to(dtype=target_dtype), v_l_spec.to(dtype=target_dtype)))
                    continue
                pruned_k = torch.index_select(k_l_spec, 2, sorted_indices).to(dtype=target_dtype)
                pruned_v = torch.index_select(v_l_spec, 2, sorted_indices).to(dtype=target_dtype)
                pruned_kv_list.append((pruned_k, pruned_v))

        elif granularity == "layer":
            if not (self.layer_qk_scores and len(self.layer_qk_scores) == len(spec_cache_tuple)):
                raise ValueError(f"Layer QK scores list length ({len(self.layer_qk_scores) if self.layer_qk_scores else 'None'}) mismatch for 'layer' pruning. Expected {len(spec_cache_tuple)} layers.")

            for layer_idx, (k_l_spec, v_l_spec) in enumerate(spec_cache_tuple):
                current_layer_seq_len = k_l_spec.shape[2]
                if current_layer_seq_len == 0:
                    pruned_kv_list.append((k_l_spec.to(dtype=target_dtype), v_l_spec.to(dtype=target_dtype)))
                    continue

                if not (self.layer_qk_scores[layer_idx].shape[0] == batch_size and 
                        self.layer_qk_scores[layer_idx].shape[-1] == current_layer_seq_len):
                    raise ValueError(f"Layer {layer_idx} QK scores shape ({self.layer_qk_scores[layer_idx].shape}) is unsuitable. Expected bsz={batch_size}, seq_len={current_layer_seq_len}.")
                if self.layer_qk_scores[layer_idx].sum().item() == 0 and num_tokens_to_keep_target < current_layer_seq_len:
                     raise ValueError(f"Layer {layer_idx} QK scores are all zero, cannot perform meaningful pruning when prompt_keep_percentage < 1.0.")
                
                layer_scores = self.layer_qk_scores[layer_idx][0] # Assuming batch_size 1
                sorted_indices = self._get_sorted_top_k_indices(layer_scores, num_tokens_to_keep_target, current_layer_seq_len, device)

                pruned_k = torch.index_select(k_l_spec, 2, sorted_indices).to(dtype=target_dtype)
                pruned_v = torch.index_select(v_l_spec, 2, sorted_indices).to(dtype=target_dtype)
                pruned_kv_list.append((pruned_k, pruned_v))
            
        elif granularity == "head":
            if not (self.head_qk_scores and len(self.head_qk_scores) == len(spec_cache_tuple)):
                raise ValueError(f"Head QK scores list length ({len(self.head_qk_scores) if self.head_qk_scores else 'None'}) mismatch for 'head' pruning. Expected {len(spec_cache_tuple)} layers.")

            for layer_idx, (k_l_spec, v_l_spec) in enumerate(spec_cache_tuple):
                current_layer_seq_len = k_l_spec.shape[2] 
                num_kv_heads_cache = k_l_spec.shape[1]

                if current_layer_seq_len == 0:
                    pruned_kv_list.append((k_l_spec.to(dtype=target_dtype), v_l_spec.to(dtype=target_dtype)))
                    continue
                
                if not (self.head_qk_scores[layer_idx].shape[0] == batch_size and 
                        self.head_qk_scores[layer_idx].shape[1] == num_kv_heads_cache and 
                        self.head_qk_scores[layer_idx].shape[2] == current_layer_seq_len):
                     raise ValueError(f"Layer {layer_idx} Head QK scores shape ({self.head_qk_scores[layer_idx].shape}) is unsuitable. Expected bsz={batch_size}, num_kv_heads={num_kv_heads_cache}, seq_len={current_layer_seq_len}.")
                if self.head_qk_scores[layer_idx].sum().item() == 0 and num_tokens_to_keep_target < current_layer_seq_len :
                     raise ValueError(f"Layer {layer_idx} Head QK scores are all zero, cannot perform meaningful pruning when prompt_keep_percentage < 1.0.")

                layer_pruned_k_heads: List[torch.Tensor] = []
                layer_pruned_v_heads: List[torch.Tensor] = []
                
                for head_idx in range(num_kv_heads_cache):
                    k_lh_spec = k_l_spec[:, head_idx, :, :] 
                    v_lh_spec = v_l_spec[:, head_idx, :, :]
                    head_seq_len = k_lh_spec.shape[1] # This is the sequence length for this specific head's slice

                    if head_seq_len == 0:
                        layer_pruned_k_heads.append(k_lh_spec.unsqueeze(1).to(dtype=target_dtype)) 
                        layer_pruned_v_heads.append(v_lh_spec.unsqueeze(1).to(dtype=target_dtype))
                        continue

                    head_scores = self.head_qk_scores[layer_idx][0, head_idx, :] # Assuming bsz 1
                    sorted_indices_head = self._get_sorted_top_k_indices(head_scores, num_tokens_to_keep_target, head_seq_len, device)
                    
                    pruned_k_head = torch.index_select(k_lh_spec, 1, sorted_indices_head).unsqueeze(1).to(dtype=target_dtype)
                    pruned_v_head = torch.index_select(v_lh_spec, 1, sorted_indices_head).unsqueeze(1).to(dtype=target_dtype)
                    layer_pruned_k_heads.append(pruned_k_head)
                    layer_pruned_v_heads.append(pruned_v_head)
                
                pruned_kv_list.append((torch.cat(layer_pruned_k_heads, dim=1), torch.cat(layer_pruned_v_heads, dim=1)))
        else:
            raise ValueError(f"Unsupported granularity for pruning: {granularity}")

        final_kept_tokens = 0
        if pruned_kv_list and pruned_kv_list[0][0].numel() > 0: # Check if the first layer's cache is not empty
            final_kept_tokens = pruned_kv_list[0][0].shape[2]
        
        return tuple(pruned_kv_list), final_kept_tokens

    def run(self, prompt_text: str, look_ahead_k: int,
            prompt_keep_percentage: float,
            max_generation_length: int) -> Tuple[str, Dict[str, float]]:

        timing_info: Dict[str, float] = {}
        overall_start_time = time.perf_counter()
        num_patched_layers = 0
        if self.speculator_model is not None: num_patched_layers = self._patch_speculator()

        limit_due_to_base = self.base_config.max_position_embeddings - max_generation_length
        if self.speculator_model is not None:
            limit_due_to_speculator = self.speculator_model.config.max_position_embeddings - look_ahead_k
            max_prompt_len_calc = float(min(limit_due_to_base, limit_due_to_speculator) - 20) # Buffer
        else: max_prompt_len_calc = float(limit_due_to_base - 20) # Buffer
        max_prompt_length = max(1, int(max_prompt_len_calc))

        inputs = self.tokenizer(prompt_text, return_tensors="pt", padding=False, truncation=True, max_length=max_prompt_length).to(self.device) # type: ignore
        prompt_input_ids, prompt_length, batch_size = inputs.input_ids, inputs.input_ids.shape[1], inputs.input_ids.shape[0]
        
        if prompt_length == 0:
            timing_info["total_time"] = time.perf_counter() - overall_start_time
            return "", timing_info

        speculator_prefill_cache: Optional[Cache] = None
        speculator_prefill_cache_as_tuple: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None
        speculator_next_token_ids: Optional[torch.Tensor] = None
        speculation_prefill_time = 0.0

        if self.speculator_model is not None:
            stage_start_time = time.perf_counter()
            self.is_generating_lookaheads = False 
            spec_prompt_cache_pos = torch.arange(prompt_length, device=self.device)
            with torch.no_grad():
                spec_out = self.speculator_model(input_ids=prompt_input_ids, use_cache=True, cache_position=spec_prompt_cache_pos)
            speculator_prefill_cache = spec_out.past_key_values
            if speculator_prefill_cache is not None:
                if not isinstance(speculator_prefill_cache, tuple): 
                    speculator_prefill_cache_as_tuple = speculator_prefill_cache.to_legacy_cache()
                else: 
                    speculator_prefill_cache_as_tuple = speculator_prefill_cache
                    speculator_prefill_cache = DynamicCache.from_legacy_cache(speculator_prefill_cache) 
            speculator_next_token_ids = torch.argmax(spec_out.logits[:, -1, :], dim=-1, keepdim=True)
            if num_patched_layers > 0:
                for q_list in self.captured_qs: q_list.clear() 
            speculation_prefill_time = time.perf_counter() - stage_start_time
        timing_info["speculation_prefill"] = speculation_prefill_time

        generated_speculator_ids: List[int] = []
        current_spec_cache_lookahead: Optional[Cache] = speculator_prefill_cache 
        speculation_decode_time = 0.0
        if (self.speculator_model is not None and num_patched_layers > 0 and look_ahead_k > 0 and
            speculator_next_token_ids is not None and current_spec_cache_lookahead is not None):
            stage_start_time = time.perf_counter()
            self.is_generating_lookaheads = True 
            current_spec_tokens = speculator_next_token_ids
            current_spec_pos = prompt_length
            if isinstance(current_spec_cache_lookahead, tuple): # Ensure DynamicCache for lookahead
                current_spec_cache_lookahead = DynamicCache.from_legacy_cache(current_spec_cache_lookahead) 

            for _ in range(look_ahead_k):
                cache_len = current_spec_cache_lookahead.get_seq_length(0) # type: ignore
                lookahead_cache_pos = torch.tensor([cache_len], device=self.device, dtype=torch.long)
                lookahead_pos_ids = torch.tensor([[current_spec_pos]], device=self.device, dtype=torch.long)
                with torch.no_grad():
                    lookahead_out = self.speculator_model(input_ids=current_spec_tokens, position_ids=lookahead_pos_ids,
                                                          past_key_values=current_spec_cache_lookahead, use_cache=True, cache_position=lookahead_cache_pos)
                current_spec_cache_lookahead = lookahead_out.past_key_values 
                current_spec_tokens = torch.argmax(lookahead_out.logits[:, -1, :], dim=-1, keepdim=True)
                token_id = current_spec_tokens.item(); generated_speculator_ids.append(token_id); current_spec_pos += 1
                if token_id in self.eos_token_ids: break
            self.is_generating_lookaheads = False 
            speculation_decode_time = time.perf_counter() - stage_start_time
        timing_info["speculation_decode"] = speculation_decode_time

        num_lookahead_steps = len(generated_speculator_ids)
        if (self.speculator_model is not None and num_lookahead_steps > 0 and num_patched_layers > 0 and
            speculator_prefill_cache_as_tuple is not None):
            self._compute_granular_qk_scores(speculator_prefill_cache_as_tuple, prompt_length, batch_size)

        base_model_first_token_gen_start_time = time.perf_counter()
        base_model_cache_after_prefill: Optional[Cache] = None
        base_model_next_token_ids: Optional[torch.Tensor] = None

        if self.speculator_model is None: # No speculator, standard base model prefill
            base_prefill_cache_pos = torch.arange(prompt_length, device=self.device)
            with torch.no_grad():
                base_out = self.base_model(input_ids=prompt_input_ids, use_cache=True, cache_position=base_prefill_cache_pos)
            base_model_next_token_ids = torch.argmax(base_out.logits[:, -1, :], dim=-1, keepdim=True)
            base_model_cache_after_prefill = base_out.past_key_values

        elif self.share_kv_cache: # Speculator used, sharing KV cache
            knockout_past_kv_dynamic: DynamicCache = DynamicCache()
            n_tokens_in_knockout_cache = 0
            if speculator_prefill_cache_as_tuple is not None and num_lookahead_steps > 0 : # Only prune if lookahead happened
                pruned_kv_tuple, n_tokens_in_knockout_cache = self._prune_kv_cache(
                    speculator_prefill_cache_as_tuple,
                    self.kv_sharing_granularity,
                    prompt_keep_percentage,
                    self.base_model.dtype, # type: ignore
                    self.device # type: ignore
                )
                if pruned_kv_tuple is not None:
                    for layer_idx, (k, v) in enumerate(pruned_kv_tuple):
                        knockout_past_kv_dynamic.update(k, v, layer_idx)
            else: # No prefill cache from speculator or no lookahead, so no cache to prune/share
                 n_tokens_in_knockout_cache = 0
            
            knockout_tokens = prompt_input_ids[:, -1:] 
            knockout_pos_ids = torch.tensor([[prompt_length - 1]], device=self.device, dtype=torch.long)
            knockout_cache_pos = torch.tensor([n_tokens_in_knockout_cache], device=self.device, dtype=torch.long)
            with torch.no_grad():
                knockout_out = self.base_model(knockout_tokens, position_ids=knockout_pos_ids,
                                               past_key_values=knockout_past_kv_dynamic, use_cache=True,
                                               cache_position=knockout_cache_pos)
            base_model_next_token_ids = torch.argmax(knockout_out.logits[:, -1, :], dim=-1, keepdim=True)
            base_model_cache_after_prefill = knockout_out.past_key_values

        else: # Speculator used, but not sharing KV cache (selective prefill for base model)
            indices_for_selective_prefill: torch.Tensor
            can_do_selective_prefill = False
            if (self.speculator_model is not None and num_lookahead_steps > 0 and num_patched_layers > 0 and
                self.global_qk_scores is not None and 
                self.global_qk_scores.shape[0] == batch_size and 
                self.global_qk_scores.shape[-1] == prompt_length):
                
                if self.global_qk_scores.sum().item() != 0 or prompt_keep_percentage == 1.0:
                    num_tokens_to_keep_sel = math.ceil(prompt_length * prompt_keep_percentage)
                    if prompt_length > 0 and num_tokens_to_keep_sel == 0: num_tokens_to_keep_sel = 1
                    num_tokens_to_keep_sel = min(num_tokens_to_keep_sel, prompt_length)

                    global_scores = self.global_qk_scores[0] # Assuming bsz 1
                    indices_for_selective_prefill = self._get_sorted_top_k_indices(
                        global_scores, num_tokens_to_keep_sel, prompt_length, self.device # type: ignore
                    )
                    if len(indices_for_selective_prefill) < prompt_length and len(indices_for_selective_prefill) > 0 :
                        can_do_selective_prefill = True
                else: # Scores are all zero and we intend to prune
                    print("Warning: Global QK scores are all zero. Cannot perform selective prefill based on QK. Falling back to full prefill.")
            
            if can_do_selective_prefill and prompt_keep_percentage < 1.0: # Ensure pruning was intended
                selected_ids = prompt_input_ids[:, indices_for_selective_prefill]
                selected_pos_ids = indices_for_selective_prefill.unsqueeze(0).to(torch.long)
                selective_cache_pos = torch.arange(selected_ids.shape[1], device=self.device)
                
                selective_prefill_base_cache = DynamicCache()
                with torch.no_grad():
                    selective_out = self.base_model(selected_ids, position_ids=selected_pos_ids,
                                                    past_key_values=selective_prefill_base_cache, use_cache=True,
                                                    cache_position=selective_cache_pos)
                base_model_cache_after_sel_prefill = selective_out.past_key_values

                knockout_tokens = prompt_input_ids[:, -1:]
                knockout_pos_ids = torch.tensor([[prompt_length - 1]], device=self.device, dtype=torch.long)
                knockout_cache_pos_base = torch.tensor([base_model_cache_after_sel_prefill.get_seq_length(0) if base_model_cache_after_sel_prefill.key_cache else 0], device=self.device, dtype=torch.long) # type: ignore
                with torch.no_grad():
                    first_token_out = self.base_model(knockout_tokens, position_ids=knockout_pos_ids,
                                                      past_key_values=base_model_cache_after_sel_prefill, use_cache=True,
                                                      cache_position=knockout_cache_pos_base)
                base_model_next_token_ids = torch.argmax(first_token_out.logits[:, -1, :], dim=-1, keepdim=True)
                base_model_cache_after_prefill = first_token_out.past_key_values
            else: # Full prefill for base model
                base_prefill_cache_pos = torch.arange(prompt_length, device=self.device)
                with torch.no_grad():
                    base_out = self.base_model(input_ids=prompt_input_ids, use_cache=True,
                                               cache_position=base_prefill_cache_pos,
                                               past_key_values=DynamicCache()) # Fresh cache
                base_model_next_token_ids = torch.argmax(base_out.logits[:, -1, :], dim=-1, keepdim=True)
                base_model_cache_after_prefill = base_out.past_key_values

        base_model_first_token_time = time.perf_counter() - base_model_first_token_gen_start_time
        timing_info["base_prefill"] = base_model_first_token_time
        if self.speculator_model is not None:
            timing_info["base_ttft"] = speculation_prefill_time + speculation_decode_time + base_model_first_token_time
        else: timing_info["base_ttft"] = base_model_first_token_time

        gen_token_ids_list: List[int] = []
        final_gen_text = ""
        if base_model_next_token_ids is not None and base_model_cache_after_prefill is not None:
            first_gen_token_id = base_model_next_token_ids.item()
            gen_token_ids_list.append(first_gen_token_id)
            current_decode_tokens = base_model_next_token_ids
            current_decode_kv_cache: Cache = base_model_cache_after_prefill
            current_real_pos = prompt_length
            if not isinstance(current_decode_kv_cache, Cache): 
                current_decode_kv_cache = DynamicCache.from_legacy_cache(current_decode_kv_cache) # type: ignore

            current_cache_write_pos = current_decode_kv_cache.get_seq_length(0) if current_decode_kv_cache.key_cache else 0 # type: ignore

            if first_gen_token_id not in self.eos_token_ids:
                for _ in range(max_generation_length - 1):
                    decode_pos_ids = torch.tensor([[current_real_pos]], device=self.device, dtype=torch.long)
                    decode_cache_pos = torch.tensor([current_cache_write_pos], device=self.device, dtype=torch.long)
                    with torch.no_grad():
                        decode_out = self.base_model(current_decode_tokens, position_ids=decode_pos_ids,
                                                     past_key_values=current_decode_kv_cache, use_cache=True,
                                                     cache_position=decode_cache_pos)
                    next_tokens = torch.argmax(decode_out.logits[:, -1, :], dim=-1, keepdim=True)
                    next_token_id = next_tokens.item(); gen_token_ids_list.append(next_token_id)
                    current_decode_tokens, current_decode_kv_cache = next_tokens, decode_out.past_key_values # type: ignore
                    current_real_pos += 1; current_cache_write_pos += 1
                    if next_token_id in self.eos_token_ids: break
            final_gen_text = self.tokenizer.decode(gen_token_ids_list, skip_special_tokens=True)

        timing_info["total_time"] = time.perf_counter() - overall_start_time
        if final_gen_text.startswith("assistant\n\n"): final_gen_text = final_gen_text[len("assistant\n\n"):] # Llama3 specific
        elif final_gen_text.startswith(" assistant\n"): final_gen_text = final_gen_text[len(" assistant\n"):]


        return final_gen_text, timing_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Speculative Prefill Pipeline with QTIP Support")
    parser.add_argument("--base_model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--speculator_model_name", type=str, default=None, help="Set to 'None' or omit for no speculator.")
    parser.add_argument("--dataset_name", type=str, default="hotpotqa")
    parser.add_argument("--look_ahead_k", type=int, default=1)
    parser.add_argument("--prompt_keep_percentage", type=float, default=0.2,
                        help="Percentage of tokens to keep from prompt based on importance scores. Default: 0.2 (20%)")
    parser.add_argument("--max_generation_length", type=int, default=32)
    parser.add_argument("--share_kv_cache", action="store_true", default=False)
    parser.add_argument("--kv_sharing_granularity", type=str, default="global",
                        choices=["global", "layer", "head"],
                        help="Granularity for KV cache pruning when sharing. 'global', 'layer', 'head' use QK scores. Errors if scores unsuitable.")
    args = parser.parse_args()

    if args.speculator_model_name and args.speculator_model_name.lower() == "none":
        args.speculator_model_name = None

    if QTIP_AVAILABLE: print("QTIP modules found and enabled IF qtip models are used.")
    else: print("Warning: QTIP modules not found. QTIP model support disabled.")

    pipeline = SpeculativePrefillPipeline(
        base_model_name=args.base_model_name,
        speculator_model_name=args.speculator_model_name,
        share_kv_cache=args.share_kv_cache,
        kv_sharing_granularity=args.kv_sharing_granularity
    )

    prompt_str: str
    if args.dataset_name == "hotpotqa":
        dataset = load_dataset('THUDM/LongBench', args.dataset_name, split='test', trust_remote_code=True) # type: ignore
        sample = dataset[0]; context = sample.get('context', '') if isinstance(sample, dict) else '' # type: ignore
        input_text = sample.get('input', '') if isinstance(sample, dict) else '' # type: ignore
        messages = [{"role": "user", "content": f"Context: {context}\nQuestion: {input_text}\nAnswer:"}]
        prompt_str = pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) # type: ignore
    else:
        messages = [{"role": "user", "content": "Explain the theory of relativity in simple terms."}]
        prompt_str = pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) # type: ignore
    
    generated_text, timing_info_run = pipeline.run(
        prompt_text=prompt_str,
        look_ahead_k=args.look_ahead_k,
        prompt_keep_percentage=args.prompt_keep_percentage,
        max_generation_length=args.max_generation_length
    )

    print(f"\n--- Generated Output ({len(pipeline.tokenizer.encode(generated_text))} tokens) ---")
    print(f"{generated_text}")
    print(f"\n--- Run Timing Information (seconds) ---")
    for stage, duration in timing_info_run.items(): print(f"  {stage}: {duration:.4f}")

    print(f"\n--- Model Information ---")
    print(f"Base model: {args.base_model_name}")
    spec_name = args.speculator_model_name
    if spec_name: print(f"Speculator model: {spec_name}")
    else: print("No speculator model used")
    print(f"KV Cache Sharing: {args.share_kv_cache}")
    if args.share_kv_cache and spec_name: print(f"Sharing Granularity: {args.kv_sharing_granularity}")
    print(f"Prompt Keep Percentage: {args.prompt_keep_percentage}")
    print(f"Lookahead K: {args.look_ahead_k}")
