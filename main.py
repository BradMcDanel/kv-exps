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

sys.path.append(os.path.join(os.path.dirname(__file__), 'qtip'))

try:
    from lib.linear import QuantizedLinear
    from lib.utils.unsafe_import import model_from_hf_path as qtip_model_from_hf_path
    from model.llama import LlamaAttention as QTIP_LlamaAttention
    QTIP_AVAILABLE = True
except ImportError:
    QTIP_AVAILABLE = False
    QuantizedLinear = None
    qtip_model_from_hf_path = None
    QTIP_LlamaAttention = None


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
    
    query_states = query_projection.view(
        batch_size, query_length, self_attn.config.num_attention_heads, self_attn.head_dim
    ).transpose(1, 2)
    key_states_before_rope = key_projection.view(
        batch_size, query_length, self_attn.config.num_key_value_heads, self_attn.head_dim
    ).transpose(1, 2)
    value_states_for_cache = value_projection.view(
        batch_size, query_length, self_attn.config.num_key_value_heads, self_attn.head_dim
    ).transpose(1, 2)
    
    cos, sin = position_embeddings
    query_states_rotated, key_states_rotated = hf_apply_rotary_pos_emb(
        query_states, key_states_before_rope, cos, sin
    )
    
    if pipeline_instance.is_generating_lookaheads:
        pipeline_instance.captured_qs[self_attn.layer_idx].append(query_states_rotated.detach().clone())
    
    if use_cache:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states_for_sdpa, value_states_for_sdpa = past_key_value.update(
            key_states_rotated, value_states_for_cache, self_attn.layer_idx, cache_kwargs
        )
    else:
        key_states_for_sdpa = key_states_rotated
        value_states_for_sdpa = value_states_for_cache
    
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
        query_states_rotated, 
        key_states_for_sdpa, 
        value_states_for_sdpa, 
        attn_mask=attn_mask_input, 
        dropout_p=0.0, 
        is_causal=is_sdpa_causal, 
        **kwargs
    )
    
    attention_output = attention_output.transpose(1, 2).contiguous().reshape(
        batch_size, query_length, self_attn.o_proj.in_features
    )
    attention_output = self_attn.o_proj(attention_output)
    
    return attention_output, None, past_key_value


def _qtip_patched_attention_forward_method(
    self_attn: Any,
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
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
    batch_size, query_length, _ = hidden_states.size()
    
    query_projection = self_attn.q_proj(hidden_states)
    key_projection = self_attn.k_proj(hidden_states)
    value_projection = self_attn.v_proj(hidden_states)
    
    query_states = query_projection.view(
        batch_size, query_length, self_attn.num_heads, self_attn.head_dim
    ).transpose(1, 2)
    key_states_before_rope = key_projection.view(
        batch_size, query_length, self_attn.num_key_value_heads, self_attn.head_dim
    ).transpose(1, 2)
    value_states_for_cache = value_projection.view(
        batch_size, query_length, self_attn.num_key_value_heads, self_attn.head_dim
    ).transpose(1, 2)
    
    cos, sin = position_embeddings
    query_states_rotated, key_states_rotated = hf_apply_rotary_pos_emb(
        query_states, key_states_before_rope, cos, sin
    )
    
    if pipeline_instance.is_generating_lookaheads:
        pipeline_instance.captured_qs[self_attn.layer_idx].append(query_states_rotated.detach().clone())
    
    if use_cache:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states_for_sdpa, value_states_for_sdpa = past_key_value.update(
            key_states_rotated, value_states_for_cache, self_attn.layer_idx, cache_kwargs
        )
    else:
        key_states_for_sdpa = key_states_rotated
        value_states_for_sdpa = value_states_for_cache
    
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
        query_states_rotated, 
        key_states_for_sdpa, 
        value_states_for_sdpa, 
        attn_mask=attn_mask_input, 
        dropout_p=0.0, 
        is_causal=is_sdpa_causal, 
        **kwargs
    )
    
    attention_output = attention_output.transpose(1, 2).contiguous().reshape(
        batch_size, query_length, self_attn.hidden_size
    )
    attention_output = self_attn.o_proj(attention_output)
    
    return attention_output, None, past_key_value


class SpeculativePrefillPipeline:
    POSITION_BUFFER = 20
    ADAPTIVE_TEMPERATURE = 1.0 
    
    def __init__(
        self,
        base_model_name: str,
        speculator_model_name: Optional[str],
        share_kv_cache: bool = False,
        pruning_strategy: str = "fixed", 
        pruning_value: float = 0.2       
    ):
        self.base_model_name = base_model_name
        self.speculator_model_name = speculator_model_name
        self.share_kv_cache = share_kv_cache
        self.pruning_strategy = pruning_strategy  
        self.pruning_value = pruning_value        
        
        self.tokenizer = self._load_tokenizer()
        self.speculator_model: Optional[AutoModelForCausalLM] = None
        self.device: Optional[torch.device] = None
        self.dtype: Optional[torch.dtype] = None
        
        if self.speculator_model_name is not None and self.speculator_model_name.lower() != "none":
            self.speculator_model = self._load_model(self.speculator_model_name)
            self.device = self.speculator_model.device
            self.dtype = self.speculator_model.dtype
        
        self.base_model = self._load_model(self.base_model_name)
        self.base_config: AutoConfig = self.base_model.config
        
        self.eos_token_ids = self._extract_eos_token_ids(self.base_config.eos_token_id)
        
        if self.device is None:
            self.device = self.base_model.device
        if self.dtype is None:
            self.dtype = self.base_model.dtype
        
        if self.share_kv_cache and self.speculator_model is not None:
            self._check_model_compatibility()
        
        self.captured_qs: List[List[torch.Tensor]] = []
        self.orig_spec_fwds: Dict[int, Any] = {}
        self.is_generating_lookaheads = False
        
        self.global_qk_scores: Optional[torch.Tensor] = None
        self.first_step_prompt_qk_scores: Optional[torch.Tensor] = None 
        self.layer_qk_scores: List[torch.Tensor] = []
    
    def _extract_eos_token_ids(self, eos_token_id: Union[int, List[int], None]) -> List[int]:
        if isinstance(eos_token_id, int):
            return [eos_token_id]
        elif isinstance(eos_token_id, list):
            return list(eos_token_id)
        else:
            return []
    
    def _is_qtip_model(self, model_name: str) -> bool:
        return QTIP_AVAILABLE and ("relaxml" in model_name.lower() or "qtip" in model_name.lower())
    
    def _load_model(self, model_name: str) -> AutoModelForCausalLM:
        if self._is_qtip_model(model_name):
            if not QTIP_AVAILABLE:
                raise ImportError(f"QTIP model requested ({model_name}) but QTIP modules not available")
            model, _ = qtip_model_from_hf_path(model_name, max_mem_ratio=0.7, attn_implementation="sdpa")
            return model.eval()
        else:
            load_kwargs = {
                "torch_dtype": torch.float16,
                "trust_remote_code": True,
                "device_map": "auto"
            }
            model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
            return model.eval()
    
    def _check_model_compatibility(self):
        if self.speculator_model is None:
            return
        
        spec_config = self.speculator_model.config
        base_config = self.base_model.config
        
        spec_num_heads = getattr(spec_config, 'num_attention_heads', 
                                getattr(spec_config, 'num_heads', None))
        base_num_heads = getattr(base_config, 'num_attention_heads', 
                                getattr(base_config, 'num_heads', None))
        
        spec_num_kv_heads = getattr(spec_config, 'num_key_value_heads', spec_num_heads)
        base_num_kv_heads = getattr(base_config, 'num_key_value_heads', base_num_heads)
        
        compatible = (
            getattr(spec_config, 'num_hidden_layers', None) == getattr(base_config, 'num_hidden_layers', None) and
            getattr(spec_config, 'hidden_size', None) == getattr(base_config, 'hidden_size', None) and
            spec_num_heads == base_num_heads and
            spec_num_kv_heads == base_num_kv_heads
        )
        
        if not compatible:
            raise ValueError("Speculator and base models are not compatible for KV cache sharing.")
    
    def _get_tokenizer_for_model(self, model_name: str) -> str:
        if "qtip" in model_name.lower():
            if "llama-3.1" in model_name.lower():
                return "meta-llama/Llama-3.1-8B-Instruct"
            elif "llama-3" in model_name.lower():
                return "meta-llama/Meta-Llama-3-8B-Instruct"
            elif "llama-2" in model_name.lower():
                return "meta-llama/Llama-2-7b-hf"
            else:
                return "meta-llama/Llama-2-7b-hf"
        return model_name
    
    def _get_chat_template(self, model_name: str) -> str:
        if "llama-3" in model_name.lower() or ("qtip" in model_name.lower() and "llama-3" in model_name.lower()):
            return "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
        elif "llama-2" in model_name.lower() or ("qtip" in model_name.lower() and "llama-2" in model_name.lower()):
            return "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 and system_message != false %}{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}"
        else:
            return "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    
    def _load_tokenizer(self):
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.base_model_name, trust_remote_code=True)
        except OSError:
            fallback_name = self._get_tokenizer_for_model(self.base_model_name)
            tokenizer = AutoTokenizer.from_pretrained(fallback_name, trust_remote_code=True)
        
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        
        if tokenizer.chat_template is None:
            tokenizer.chat_template = self._get_chat_template(self.base_model_name)
        
        return tokenizer
    
    def _patch_speculator(self):
        if self.speculator_model is None:
            return 0
        
        if not hasattr(self.speculator_model, 'model') or not hasattr(self.speculator_model.model, 'layers'):
            return 0
        
        num_layers = self.speculator_model.config.num_hidden_layers
        self.captured_qs = [[] for _ in range(num_layers)]
        self.global_qk_scores = None
        self.first_step_prompt_qk_scores = None 
        self.layer_qk_scores = []
        
        num_patched_layers = 0
        
        for i, layer in enumerate(self.speculator_model.model.layers):
            if hasattr(layer, 'self_attn'):
                attn = layer.self_attn
                
                if isinstance(attn, LlamaAttention):
                    if i not in self.orig_spec_fwds:
                        self.orig_spec_fwds[i] = attn.forward
                    attn.forward = types.MethodType(partial(_hf_patched_attention_forward_method, pipeline_instance=self), attn)
                    num_patched_layers += 1
                
                elif QTIP_AVAILABLE and QTIP_LlamaAttention is not None and isinstance(attn, QTIP_LlamaAttention):
                    if i not in self.orig_spec_fwds:
                        self.orig_spec_fwds[i] = attn.forward
                    attn.forward = types.MethodType(partial(_qtip_patched_attention_forward_method, pipeline_instance=self), attn)
                    num_patched_layers += 1
        
        return num_patched_layers
    
    def _compute_qk_scores(
        self,
        speculator_prefill_cache_as_tuple: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
        prompt_length: int,
        batch_size: int
    ):
        if self.speculator_model is None or not self.captured_qs or not any(self.captured_qs):
            return
        
        spec_config = self.speculator_model.config
        num_spec_layers = spec_config.num_hidden_layers
        
        spec_num_q_heads = getattr(spec_config, 'num_attention_heads', 
                                  getattr(spec_config, 'num_heads', None))
        spec_num_kv_heads = getattr(spec_config, 'num_key_value_heads', spec_num_q_heads)
        
        if spec_num_q_heads is None or spec_num_kv_heads is None:
            print("Warning: Could not determine head counts for speculator. Skipping QK score calculation.")
            return
        
        spec_num_kv_groups = spec_num_q_heads // spec_num_kv_heads
        
        self.layer_qk_scores = [
            torch.zeros(batch_size, prompt_length, device=self.device, dtype=self.dtype)
            for _ in range(num_spec_layers)
        ]
        first_step_layer_scores_list = [
            torch.zeros(batch_size, prompt_length, device=self.device, dtype=self.dtype)
            for _ in range(num_spec_layers)
        ]
        
        for layer_idx in range(num_spec_layers):
            if not self.captured_qs[layer_idx] or speculator_prefill_cache_as_tuple[layer_idx][0].numel() == 0:
                continue
            
            key_prompt_layer_spec = speculator_prefill_cache_as_tuple[layer_idx][0].detach()
            key_prompt_layer_rep_spec = hf_repeat_kv(key_prompt_layer_spec, spec_num_kv_groups)
            
            if not self.captured_qs[layer_idx]:
                continue
            
            head_dim_from_q = self.captured_qs[layer_idx][0].shape[-1]
            
            if len(self.captured_qs[layer_idx]) > 0:
                query_first_lookahead_layer = self.captured_qs[layer_idx][0] 
                attn_logits_first_step = torch.matmul(
                    query_first_lookahead_layer, 
                    key_prompt_layer_rep_spec.transpose(-1, -2)
                ) / math.sqrt(head_dim_from_q)
                attn_scores_first_step = attn_logits_first_step.squeeze(2) 
                first_step_layer_scores_list[layer_idx] = attn_scores_first_step.sum(dim=1) 

            for spec_idx in range(len(self.captured_qs[layer_idx])):
                query_lookahead_layer_step = self.captured_qs[layer_idx][spec_idx]
                attn_logits_layer_step = torch.matmul(
                    query_lookahead_layer_step, 
                    key_prompt_layer_rep_spec.transpose(-1, -2)
                ) / math.sqrt(head_dim_from_q)
                attn_scores_layer_step = attn_logits_layer_step.squeeze(2)
                layer_total_scores_step = attn_scores_layer_step.sum(dim=1)
                self.layer_qk_scores[layer_idx] += layer_total_scores_step
        
        if self.layer_qk_scores:
            valid_layer_scores = [
                ls for ls in self.layer_qk_scores 
                if ls.numel() > 0 and ls.shape[0] == batch_size and ls.shape[-1] == prompt_length
            ]
            if valid_layer_scores:
                self.global_qk_scores = torch.sum(torch.stack(valid_layer_scores), dim=0)
            else:
                self.global_qk_scores = torch.zeros(batch_size, prompt_length, device=self.device, dtype=self.dtype)
        else:
            self.global_qk_scores = torch.zeros(batch_size, prompt_length, device=self.device, dtype=self.dtype)

        valid_first_step_layer_scores = [
            fsls for fsls in first_step_layer_scores_list
            if fsls.numel() > 0 and fsls.shape[0] == batch_size and fsls.shape[-1] == prompt_length
        ]
        if valid_first_step_layer_scores:
            self.first_step_prompt_qk_scores = torch.sum(torch.stack(valid_first_step_layer_scores), dim=0)
        else:
            self.first_step_prompt_qk_scores = torch.zeros(batch_size, prompt_length, device=self.device, dtype=self.dtype)

    def _get_sorted_top_k_indices(
        self, 
        scores: torch.Tensor, 
        num_to_keep: int, 
        current_seq_len: int, 
        device: torch.device
    ) -> torch.Tensor:
        if current_seq_len == 0:
            return torch.empty(0, dtype=torch.long, device=device)
        
        if num_to_keep == 0 and current_seq_len > 0 and self.pruning_value > 0 :
            if self.pruning_strategy == "adaptive" or (self.pruning_strategy == "fixed" and self.pruning_value > 0): # Ensure we keep 1 if any budget
                 num_to_keep = 1 
        
        num_to_keep = min(num_to_keep, current_seq_len)
        
        if num_to_keep == 0:
            return torch.empty(0, dtype=torch.long, device=device)
        
        if scores.ndim > 1:
            scores = scores.squeeze() 
        
        if scores.shape[0] != current_seq_len:
             # This can happen if scores is an empty tensor due to no tokens in a category (e.g. no non-positive scores)
            if current_seq_len == 0 and scores.numel() == 0: # If both are empty, it's fine
                return torch.empty(0, dtype=torch.long, device=device)
            raise ValueError(f"Score length {scores.shape[0]} does not match current_seq_len {current_seq_len}")
        
        if num_to_keep >= current_seq_len:
            return torch.arange(current_seq_len, device=device, dtype=torch.long)
        
        _, top_k_indices = torch.topk(scores, k=num_to_keep)
        return torch.sort(top_k_indices)[0]


    def _calculate_num_tokens_to_keep( 
        self,
        original_seq_len: int,
        batch_size: int, 
        device: torch.device
    ) -> Tuple[str, Any, Any]: 
        
        if self.pruning_strategy == "adaptive":
            mode = "adaptive"
            if not (self.first_step_prompt_qk_scores is not None and
                    self.first_step_prompt_qk_scores.shape[0] == batch_size and
                    self.first_step_prompt_qk_scores.shape[-1] == original_seq_len):
                raise ValueError("First-step QK scores shape is unsuitable for 'adaptive' pruning.")

            raw_scores_all = self.first_step_prompt_qk_scores[0].clone()

            if original_seq_len == 0:
                return mode, torch.empty(0, dtype=torch.long, device=device), 0

            all_indices = torch.arange(original_seq_len, device=device)
            positive_scores_mask = raw_scores_all > 0
            
            mandatory_positive_indices = all_indices[positive_scores_mask]
            
            non_positive_scores_mask = ~positive_scores_mask
            scores_non_positive_remainder = raw_scores_all[non_positive_scores_mask]
            num_non_positive_remainder_tokens = scores_non_positive_remainder.numel()

            num_additional_from_non_positive = 0

            if num_non_positive_remainder_tokens > 0 and self.pruning_value > 0:
                if num_non_positive_remainder_tokens == 1:
                    num_additional_from_non_positive = 1 
                else:
                    mean_remainder = scores_non_positive_remainder.mean()
                    std_remainder = scores_non_positive_remainder.std()

                    standardized_remainder_scores = None
                    if std_remainder < 1e-6:
                        standardized_remainder_scores = torch.zeros_like(scores_non_positive_remainder)
                    else:
                        standardized_remainder_scores = (scores_non_positive_remainder - mean_remainder) / std_remainder
                    
                    probabilities_remainder = torch.softmax(
                        standardized_remainder_scores / self.ADAPTIVE_TEMPERATURE, dim=-1
                    )
                    
                    # --- Vectorized budget calculation ---
                    sorted_probabilities_remainder, _ = torch.sort(probabilities_remainder, descending=True)
                    
                    cumulative_probabilities = torch.cumsum(sorted_probabilities_remainder, dim=0)
                    
                    target_cumulative_probability_remainder = torch.tensor([self.pruning_value], device=device, dtype=probabilities_remainder.dtype)

                    num_additional_from_non_positive = torch.searchsorted(
                        cumulative_probabilities, 
                        target_cumulative_probability_remainder, 
                        right=False 
                    ).item() + 1

                    if num_additional_from_non_positive > num_non_positive_remainder_tokens and cumulative_probabilities[-1] < target_cumulative_probability_remainder[0] :
                         num_additional_from_non_positive = num_non_positive_remainder_tokens
                    
                    if num_non_positive_remainder_tokens > 0 and \
                       num_additional_from_non_positive == 0 and \
                       self.pruning_value > 0:
                        num_additional_from_non_positive = 1
            
            num_additional_from_non_positive = min(
                num_additional_from_non_positive, num_non_positive_remainder_tokens
            )
            
            return mode, mandatory_positive_indices, num_additional_from_non_positive
        
        elif self.pruning_strategy == "fixed":
            mode = "fixed"
            fixed_total_count = math.ceil(original_seq_len * self.pruning_value)
            if self.pruning_value == 0:
                fixed_total_count = 0
            fixed_total_count = min(fixed_total_count, original_seq_len)

            scores_for_fixed_selection = None
            if self.first_step_prompt_qk_scores is not None and \
               self.first_step_prompt_qk_scores.shape[0] == batch_size and \
               self.first_step_prompt_qk_scores.shape[-1] == original_seq_len:
                scores_for_fixed_selection = self.first_step_prompt_qk_scores[0].clone()
            elif self.global_qk_scores is not None and \
                 self.global_qk_scores.shape[0] == batch_size and \
                 self.global_qk_scores.shape[-1] == original_seq_len:
                scores_for_fixed_selection = self.global_qk_scores[0].clone()
            
            if fixed_total_count > 0 and fixed_total_count < original_seq_len and scores_for_fixed_selection is None:
                raise ValueError("Fixed pruning needs scores for selection, but none are suitable.")
            
            if scores_for_fixed_selection is None and (fixed_total_count >= original_seq_len or fixed_total_count == 0):
                 scores_for_fixed_selection = torch.arange(original_seq_len, device=device, dtype=torch.float) 
                 if fixed_total_count == 0:
                     scores_for_fixed_selection = torch.empty(0, device=device, dtype=torch.float)

            return mode, fixed_total_count, scores_for_fixed_selection
        else:
            raise ValueError(f"Unknown pruning_strategy: {self.pruning_strategy}")

    def _prune_kv_cache(
        self,
        spec_cache_tuple: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
        target_dtype: torch.dtype,
        device: torch.device
    ) -> Tuple[Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]], int]:
        if not spec_cache_tuple or spec_cache_tuple[0][0].numel() == 0:
            return None, 0
        
        batch_size = spec_cache_tuple[0][0].shape[0]
        original_seq_len = spec_cache_tuple[0][0].shape[2]
        
        if original_seq_len == 0:
            return spec_cache_tuple, 0

        mode, data1, data2 = self._calculate_num_tokens_to_keep(original_seq_len, batch_size, device)
        
        final_indices_to_keep: Optional[torch.Tensor] = None
        num_tokens_kept_for_metadata = 0 # For metadata

        if mode == "adaptive":
            mandatory_positive_indices: torch.Tensor = data1
            num_additional_needed: int = data2
            
            additional_absolute_indices = torch.empty(0, dtype=torch.long, device=device)

            if num_additional_needed > 0:
                all_raw_scores = self.first_step_prompt_qk_scores[0] # Should be [original_seq_len]
                non_positive_mask = ~(all_raw_scores > 0)
                all_original_indices = torch.arange(original_seq_len, device=device)
                
                non_positive_original_indices = all_original_indices[non_positive_mask]
                non_positive_scores_for_selection = all_raw_scores[non_positive_mask]

                if non_positive_scores_for_selection.numel() > 0:
                    relative_indices_from_non_positive = self._get_sorted_top_k_indices(
                        non_positive_scores_for_selection,
                        num_additional_needed,
                        non_positive_scores_for_selection.numel(),
                        device
                    )
                    additional_absolute_indices = non_positive_original_indices[relative_indices_from_non_positive]
            
            if mandatory_positive_indices.numel() > 0 or additional_absolute_indices.numel() > 0:
                final_indices_to_keep = torch.cat((mandatory_positive_indices, additional_absolute_indices))
                final_indices_to_keep = torch.unique(final_indices_to_keep)
                final_indices_to_keep = torch.sort(final_indices_to_keep)[0]
            else: # No tokens to keep
                final_indices_to_keep = torch.empty(0, dtype=torch.long, device=device)
            
            num_tokens_kept_for_metadata = final_indices_to_keep.numel()

        elif mode == "fixed":
            num_tokens_to_keep_target: int = data1
            scores_for_selection: Optional[torch.Tensor] = data2
            
            if scores_for_selection is None: # Should only happen if keeping all or none and scores weren't found
                if num_tokens_to_keep_target == original_seq_len:
                    final_indices_to_keep = torch.arange(original_seq_len, device=device, dtype=torch.long)
                elif num_tokens_to_keep_target == 0:
                    final_indices_to_keep = torch.empty(0, dtype=torch.long, device=device)
                else: # This case should have raised error in _calculate_num_tokens_to_keep
                     raise ValueError("Scores for fixed selection are missing when needed for partial keep.")
            else:
                final_indices_to_keep = self._get_sorted_top_k_indices(
                    scores_for_selection, num_tokens_to_keep_target, original_seq_len, device
                )
            num_tokens_kept_for_metadata = final_indices_to_keep.numel()
        
        else: # Should not happen
            raise ValueError(f"Invalid mode {mode} from _calculate_num_tokens_to_keep")

        pruned_kv_list: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for k_l_spec, v_l_spec in spec_cache_tuple:
            if k_l_spec.shape[2] == 0: 
                pruned_kv_list.append((k_l_spec.to(dtype=target_dtype), v_l_spec.to(dtype=target_dtype)))
                continue
            
            pruned_k = torch.index_select(k_l_spec, 2, final_indices_to_keep).to(dtype=target_dtype)
            pruned_v = torch.index_select(v_l_spec, 2, final_indices_to_keep).to(dtype=target_dtype)
            pruned_kv_list.append((pruned_k, pruned_v))
        
        final_kept_tokens_count = 0
        if pruned_kv_list and pruned_kv_list[0][0].numel() > 0:
            final_kept_tokens_count = pruned_kv_list[0][0].shape[2]
        
        # Ensure consistency, this value should be same as num_tokens_kept_for_metadata
        assert final_kept_tokens_count == num_tokens_kept_for_metadata, \
               f"Mismatch in kept token count: {final_kept_tokens_count} vs {num_tokens_kept_for_metadata}"

        return tuple(pruned_kv_list), final_kept_tokens_count
    
    def run(
        self,
        prompt_text: str,
        look_ahead_k: int,
        max_generation_length: int
    ) -> Tuple[str, Dict[str, Any]]: 
        run_metadata: Dict[str, Any] = {} 
        overall_start_time = time.perf_counter()

        run_metadata["pruning_strategy_used"] = self.pruning_strategy
        run_metadata["pruning_threshold_value"] = self.pruning_value
        if self.pruning_strategy == "adaptive":
            run_metadata["adaptive_temperature"] = self.ADAPTIVE_TEMPERATURE
        
        num_patched_layers = 0
        if self.speculator_model is not None:
            num_patched_layers = self._patch_speculator() 
        
        limit_due_to_base = self.base_config.max_position_embeddings - max_generation_length
        if self.speculator_model is not None:
            limit_due_to_speculator = self.speculator_model.config.max_position_embeddings - look_ahead_k
            max_prompt_len_calc = float(min(limit_due_to_base, limit_due_to_speculator) - self.POSITION_BUFFER)
        else:
            max_prompt_len_calc = float(limit_due_to_base - self.POSITION_BUFFER)
        
        max_prompt_length = max(1, int(max_prompt_len_calc))
        
        inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=max_prompt_length
        ).to(self.device)
        
        prompt_input_ids = inputs.input_ids
        prompt_length = inputs.input_ids.shape[1]
        batch_size = inputs.input_ids.shape[0]
        num_kept_tokens_for_base_prefill = prompt_length # Default to full prefill

        run_metadata["prompt_input_length"] = prompt_length 
        
        if prompt_length == 0:
            run_metadata["total_time"] = time.perf_counter() - overall_start_time
            run_metadata["token_keep_rate"] = 100.0
            return "", run_metadata
        
        speculator_prefill_cache: Optional[Cache] = None
        speculator_prefill_cache_as_tuple: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None
        speculator_next_token_ids: Optional[torch.Tensor] = None
        speculation_prefill_time = 0.0
        
        if self.speculator_model is not None:
            stage_start_time = time.perf_counter()
            self.is_generating_lookaheads = False
            
            spec_prompt_cache_pos = torch.arange(prompt_length, device=self.device)
            with torch.no_grad():
                spec_out = self.speculator_model(
                    input_ids=prompt_input_ids,
                    use_cache=True,
                    cache_position=spec_prompt_cache_pos
                )
            
            speculator_prefill_cache = spec_out.past_key_values
            if speculator_prefill_cache is not None:
                if not isinstance(speculator_prefill_cache, tuple): 
                    speculator_prefill_cache_as_tuple = speculator_prefill_cache.to_legacy_cache()
                else: 
                    speculator_prefill_cache_as_tuple = speculator_prefill_cache
                    speculator_prefill_cache = DynamicCache.from_legacy_cache(speculator_prefill_cache) 
            
            speculator_next_token_ids = torch.argmax(spec_out.logits[:, -1, :], dim=-1, keepdim=True)
            
            if num_patched_layers > 0: 
                for q_list in self.captured_qs:
                    q_list.clear()
            
            speculation_prefill_time = time.perf_counter() - stage_start_time
        
        run_metadata["speculation_prefill"] = speculation_prefill_time
        
        generated_speculator_ids: List[int] = []
        current_spec_cache_lookahead: Optional[Cache] = speculator_prefill_cache 
        speculation_decode_time = 0.0
        
        if (self.speculator_model is not None and 
            num_patched_layers > 0 and 
            look_ahead_k > 0 and
            speculator_next_token_ids is not None and 
            current_spec_cache_lookahead is not None): 
            
            stage_start_time = time.perf_counter()
            self.is_generating_lookaheads = True 
            
            current_spec_tokens = speculator_next_token_ids
            current_spec_pos = prompt_length
            
            for _ in range(look_ahead_k):
                cache_len = current_spec_cache_lookahead.get_seq_length(0) 
                lookahead_cache_pos = torch.tensor([cache_len], device=self.device, dtype=torch.long)
                lookahead_pos_ids = torch.tensor([[current_spec_pos]], device=self.device, dtype=torch.long)
                
                with torch.no_grad():
                    lookahead_out = self.speculator_model(
                        input_ids=current_spec_tokens,
                        position_ids=lookahead_pos_ids,
                        past_key_values=current_spec_cache_lookahead,
                        use_cache=True,
                        cache_position=lookahead_cache_pos
                    )
                
                current_spec_cache_lookahead = lookahead_out.past_key_values
                current_spec_tokens = torch.argmax(lookahead_out.logits[:, -1, :], dim=-1, keepdim=True)
                
                token_id = current_spec_tokens.item()
                generated_speculator_ids.append(token_id)
                current_spec_pos += 1
                
                if token_id in self.eos_token_ids:
                    break
            
            self.is_generating_lookaheads = False
            speculation_decode_time = time.perf_counter() - stage_start_time
        
        run_metadata["speculation_decode"] = speculation_decode_time
        
        num_lookahead_steps = len(generated_speculator_ids)
        if (self.speculator_model is not None and 
            num_lookahead_steps > 0 and 
            num_patched_layers > 0 and
            speculator_prefill_cache_as_tuple is not None):
            self._compute_qk_scores(speculator_prefill_cache_as_tuple, prompt_length, batch_size)
        
        base_model_first_token_gen_start_time = time.perf_counter()
        base_model_cache_after_prefill: Optional[Cache] = None
        base_model_next_token_ids: Optional[torch.Tensor] = None
        
        current_pruning_applicable = (self.speculator_model is not None and 
                                     num_lookahead_steps > 0 and 
                                     num_patched_layers > 0 and
                                     self.first_step_prompt_qk_scores is not None) # Scores must be computed
        
        if self.speculator_model is None: 
            base_prefill_cache_pos = torch.arange(prompt_length, device=self.device)
            with torch.no_grad():
                base_out = self.base_model(
                    input_ids=prompt_input_ids,
                    use_cache=True,
                    cache_position=base_prefill_cache_pos
                )
            base_model_next_token_ids = torch.argmax(base_out.logits[:, -1, :], dim=-1, keepdim=True)
            base_model_cache_after_prefill = base_out.past_key_values
        
        elif self.share_kv_cache:
            knockout_past_kv_dynamic: DynamicCache = DynamicCache()
            n_tokens_in_knockout_cache = 0 
            
            if speculator_prefill_cache_as_tuple is not None:
                spec_cache_original_len = speculator_prefill_cache_as_tuple[0][0].shape[2]
                run_metadata["spec_cache_len_before_prune"] = spec_cache_original_len

                pruned_kv_tuple_result: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None
                if current_pruning_applicable: 
                    pruned_kv_tuple_result, n_tokens_in_knockout_cache = self._prune_kv_cache(
                        speculator_prefill_cache_as_tuple,
                        self.base_model.dtype,
                        self.device
                    )
                else: 
                    pruned_kv_tuple_result = speculator_prefill_cache_as_tuple
                    n_tokens_in_knockout_cache = spec_cache_original_len
                
                run_metadata["spec_cache_len_after_prune"] = n_tokens_in_knockout_cache 
                num_kept_tokens_for_base_prefill = n_tokens_in_knockout_cache
                
                if pruned_kv_tuple_result is not None:
                    for layer_idx, (k, v) in enumerate(pruned_kv_tuple_result):
                        knockout_past_kv_dynamic.update(k.to(self.base_model.dtype), v.to(self.base_model.dtype), layer_idx)
            else: 
                run_metadata["spec_cache_len_before_prune"] = 0 
                run_metadata["spec_cache_len_after_prune"] = 0
            
            knockout_tokens = prompt_input_ids[:, -1:]
            knockout_pos_ids = torch.tensor([[prompt_length - 1]], device=self.device, dtype=torch.long)
            # Use actual length of cache after pruning for knockout_cache_pos
            actual_knockout_cache_len = knockout_past_kv_dynamic.get_seq_length(0) if knockout_past_kv_dynamic.key_cache else 0
            knockout_cache_pos = torch.tensor([actual_knockout_cache_len], device=self.device, dtype=torch.long)
            
            with torch.no_grad():
                knockout_out = self.base_model(
                    knockout_tokens,
                    position_ids=knockout_pos_ids,
                    past_key_values=knockout_past_kv_dynamic,
                    use_cache=True,
                    cache_position=knockout_cache_pos
                )
            
            base_model_next_token_ids = torch.argmax(knockout_out.logits[:, -1, :], dim=-1, keepdim=True)
            base_model_cache_after_prefill = knockout_out.past_key_values
        
        else: # Selective prefill
            final_indices_for_selective_prefill: Optional[torch.Tensor] = None
            
            if current_pruning_applicable:
                mode, data1, data2 = self._calculate_num_tokens_to_keep(prompt_length, batch_size, self.device)
                if mode == "adaptive":
                    mandatory_positive_indices: torch.Tensor = data1
                    num_additional_needed: int = data2
                    additional_absolute_indices = torch.empty(0, dtype=torch.long, device=self.device)

                    if num_additional_needed > 0:
                        all_raw_scores = self.first_step_prompt_qk_scores[0]
                        non_positive_mask = ~(all_raw_scores > 0)
                        all_original_indices = torch.arange(prompt_length, device=self.device)
                        non_positive_original_indices = all_original_indices[non_positive_mask]
                        non_positive_scores_for_selection = all_raw_scores[non_positive_mask]

                        if non_positive_scores_for_selection.numel() > 0:
                            relative_indices = self._get_sorted_top_k_indices(
                                non_positive_scores_for_selection, num_additional_needed,
                                non_positive_scores_for_selection.numel(), self.device
                            )
                            additional_absolute_indices = non_positive_original_indices[relative_indices]
                    
                    if mandatory_positive_indices.numel() > 0 or additional_absolute_indices.numel() > 0:
                        final_indices_for_selective_prefill = torch.cat((mandatory_positive_indices, additional_absolute_indices))
                        final_indices_for_selective_prefill = torch.unique(final_indices_for_selective_prefill)
                        final_indices_for_selective_prefill = torch.sort(final_indices_for_selective_prefill)[0]
                    else:
                         final_indices_for_selective_prefill = torch.empty(0, dtype=torch.long, device=self.device)

                elif mode == "fixed":
                    num_to_keep: int = data1
                    scores_for_sel: Optional[torch.Tensor] = data2
                    if scores_for_sel is None:
                        if num_to_keep == prompt_length: final_indices_for_selective_prefill = torch.arange(prompt_length, device=self.device)
                        elif num_to_keep == 0: final_indices_for_selective_prefill = torch.empty(0, dtype=torch.long, device=self.device)
                        else: raise ValueError("Scores missing for fixed selective prefill.")
                    else:
                        final_indices_for_selective_prefill = self._get_sorted_top_k_indices(
                            scores_for_sel, num_to_keep, prompt_length, self.device
                        )
            else: # Not applicable for pruning, so full prefill
                final_indices_for_selective_prefill = torch.arange(prompt_length, device=self.device)

            num_kept_tokens_for_base_prefill = final_indices_for_selective_prefill.numel()
            run_metadata["selective_prefill_original_len"] = prompt_length
            run_metadata["selective_prefill_kept_token_count"] = final_indices_for_selective_prefill.numel()

            if final_indices_for_selective_prefill.numel() < prompt_length and final_indices_for_selective_prefill.numel() > 0 :
                selected_ids = prompt_input_ids[:, final_indices_for_selective_prefill]
                selected_pos_ids = final_indices_for_selective_prefill.unsqueeze(0).to(torch.long)
                selective_cache_pos = torch.arange(selected_ids.shape[1], device=self.device)
                
                selective_prefill_base_cache = DynamicCache()
                with torch.no_grad():
                    selective_out = self.base_model(
                        selected_ids, position_ids=selected_pos_ids,
                        past_key_values=selective_prefill_base_cache, use_cache=True,
                        cache_position=selective_cache_pos
                    )
                base_model_cache_after_sel_prefill = selective_out.past_key_values
                
                knockout_tokens = prompt_input_ids[:, -1:]
                knockout_pos_ids = torch.tensor([[prompt_length - 1]], device=self.device, dtype=torch.long)
                cache_len_after_sel = base_model_cache_after_sel_prefill.get_seq_length(0) if base_model_cache_after_sel_prefill.key_cache else 0
                knockout_cache_pos_base = torch.tensor([cache_len_after_sel], device=self.device, dtype=torch.long)
                
                with torch.no_grad():
                    first_token_out = self.base_model(
                        knockout_tokens, position_ids=knockout_pos_ids,
                        past_key_values=base_model_cache_after_sel_prefill, use_cache=True,
                        cache_position=knockout_cache_pos_base
                    )
                base_model_next_token_ids = torch.argmax(first_token_out.logits[:, -1, :], dim=-1, keepdim=True)
                base_model_cache_after_prefill = first_token_out.past_key_values
            else: # Full prefill (either all tokens selected, or no tokens for selective prefill, or pruning not applicable)
                base_prefill_cache_pos = torch.arange(prompt_length, device=self.device)
                with torch.no_grad():
                    base_out = self.base_model(
                        input_ids=prompt_input_ids, use_cache=True,
                        cache_position=base_prefill_cache_pos, past_key_values=DynamicCache()
                    )
                base_model_next_token_ids = torch.argmax(base_out.logits[:, -1, :], dim=-1, keepdim=True)
                base_model_cache_after_prefill = base_out.past_key_values
        
        if prompt_length > 0:
            run_metadata["token_keep_rate"] = (num_kept_tokens_for_base_prefill / prompt_length) * 100.0
        else:
            run_metadata["token_keep_rate"] = 100.0
        
        base_model_first_token_time = time.perf_counter() - base_model_first_token_gen_start_time
        run_metadata["base_prefill"] = base_model_first_token_time
        
        if self.speculator_model is not None:
            run_metadata["base_ttft"] = speculation_prefill_time + speculation_decode_time + base_model_first_token_time
        else:
            run_metadata["base_ttft"] = base_model_first_token_time
        
        gen_token_ids_list: List[int] = []
        final_gen_text = ""
        
        if base_model_next_token_ids is not None and base_model_cache_after_prefill is not None:
            first_gen_token_id = base_model_next_token_ids.item()
            gen_token_ids_list.append(first_gen_token_id)
            
            current_decode_tokens = base_model_next_token_ids
            current_decode_kv_cache: Cache = base_model_cache_after_prefill
            current_real_pos = prompt_length
            
            if not isinstance(current_decode_kv_cache, DynamicCache): 
                 if isinstance(current_decode_kv_cache, tuple): # Legacy
                    current_decode_kv_cache = DynamicCache.from_legacy_cache(current_decode_kv_cache)
                 elif not isinstance(current_decode_kv_cache, Cache): # Should not happen with HF models
                    raise TypeError(f"Unexpected cache type: {type(current_decode_kv_cache)}")


            current_cache_write_pos = current_decode_kv_cache.get_seq_length(0) if current_decode_kv_cache.key_cache else 0
            
            if first_gen_token_id not in self.eos_token_ids:
                for _ in range(max_generation_length - 1):
                    decode_pos_ids = torch.tensor([[current_real_pos]], device=self.device, dtype=torch.long)
                    decode_cache_pos = torch.tensor([current_cache_write_pos], device=self.device, dtype=torch.long)
                    
                    with torch.no_grad():
                        decode_out = self.base_model(
                            current_decode_tokens,
                            position_ids=decode_pos_ids,
                            past_key_values=current_decode_kv_cache,
                            use_cache=True,
                            cache_position=decode_cache_pos
                        )
                    
                    next_tokens = torch.argmax(decode_out.logits[:, -1, :], dim=-1, keepdim=True)
                    next_token_id = next_tokens.item()
                    gen_token_ids_list.append(next_token_id)
                    
                    current_decode_tokens = next_tokens
                    current_decode_kv_cache = decode_out.past_key_values
                    current_real_pos += 1
                    current_cache_write_pos += 1
                    
                    if next_token_id in self.eos_token_ids:
                        break
            
            final_gen_text = self.tokenizer.decode(gen_token_ids_list, skip_special_tokens=True)
        
        run_metadata["total_time"] = time.perf_counter() - overall_start_time
        
        if final_gen_text.startswith("assistant\n\n"):
            final_gen_text = final_gen_text[len("assistant\n\n"):]
        elif final_gen_text.startswith(" assistant\n"):
            final_gen_text = final_gen_text[len(" assistant\n"):]
        
        return final_gen_text, run_metadata


def main():
    parser = argparse.ArgumentParser(description="Speculative Prefill Pipeline with QTIP Support")
    parser.add_argument("--base_model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--speculator_model_name", type=str, default=None, 
                        help="Set to 'None' or omit for no speculator.")
    parser.add_argument("--dataset_name", type=str, default="hotpotqa")
    parser.add_argument("--look_ahead_k", type=int, default=1)
    parser.add_argument("--pruning_value", type=float, default=0.2,
                        help="Value for pruning. If strategy is 'fixed', this is fraction of tokens (0.0-1.0). "
                             "If strategy is 'adaptive', this is fraction of QK score energy (0.0-1.0) from non-positive scores. Default: 0.2")
    parser.add_argument("--pruning_strategy", type=str, default="fixed", choices=["fixed", "adaptive"],
                        help="Pruning strategy. 'fixed': use pruning_value as token fraction. "
                             "'adaptive': keep all positive-scoring tokens, then use pruning_value on Z-score+softmax of non-positive scores. Default: fixed")
    parser.add_argument("--max_generation_length", type=int, default=32)
    parser.add_argument("--share_kv_cache", action="store_true", default=False)
    args = parser.parse_args()
    
    if args.speculator_model_name and args.speculator_model_name.lower() == "none":
        args.speculator_model_name = None
    
    if QTIP_AVAILABLE:
        print("QTIP modules found and enabled for QTIP models.")
    else:
        print("Warning: QTIP modules not found. QTIP model support disabled.")
    
    pipeline = SpeculativePrefillPipeline(
        base_model_name=args.base_model_name,
        speculator_model_name=args.speculator_model_name,
        share_kv_cache=args.share_kv_cache,
        pruning_strategy=args.pruning_strategy, 
        pruning_value=args.pruning_value        
    )
    
    prompt_str: str
    if args.dataset_name == "hotpotqa":
        dataset = load_dataset('THUDM/LongBench', args.dataset_name, split='test', trust_remote_code=True)
        sample = dataset[0]
        context = sample.get('context', '') if isinstance(sample, dict) else ''
        input_text = sample.get('input', '') if isinstance(sample, dict) else ''
        messages = [{"role": "user", "content": f"Context: {context}\nQuestion: {input_text}\nAnswer:"}]
        prompt_str = pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        messages = [{"role": "user", "content": "Explain the theory of relativity in simple terms."}]
        prompt_str = pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    generated_text, run_metadata = pipeline.run(
        prompt_text=prompt_str,
        look_ahead_k=args.look_ahead_k,
        max_generation_length=args.max_generation_length
    )
    
    print(f"\n--- Generated Output ({len(pipeline.tokenizer.encode(generated_text))} tokens) ---")
    print(f"{generated_text}")
    
    print(f"\n--- Run Metadata & Timing Information ---") 
    for key, value in run_metadata.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}") 
    
    print(f"\n--- Model Information ---")
    print(f"Base model: {args.base_model_name}")
    
    if args.speculator_model_name:
        print(f"Speculator model: {args.speculator_model_name}")
    else:
        print("No speculator model used")
    
    print(f"KV Cache Sharing: {args.share_kv_cache}")
    
    print(f"Pruning Strategy: {args.pruning_strategy}")
    print(f"Pruning Value: {args.pruning_value}")
    if args.pruning_strategy == "adaptive":
        print(f"Adaptive Temperature: {SpeculativePrefillPipeline.ADAPTIVE_TEMPERATURE}")
    print(f"Lookahead K: {args.look_ahead_k}")


if __name__ == "__main__":
    main()
