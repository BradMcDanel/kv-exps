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
    
    def __init__(
        self,
        base_model_name: str,
        speculator_model_name: Optional[str],
        max_capacity_prompt: int = 512
    ):
        self.base_model_name = base_model_name
        self.speculator_model_name = speculator_model_name
        self.max_capacity_prompt = max_capacity_prompt
        
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
    

    def _denoise_and_select_chunks(
        self,
        scores: torch.Tensor,
        num_to_keep: int,
        original_seq_len: int,
        chunk_size: int = 64, # A reasonable default, can be tuned
        pool_kernel_size: int = 7 # As mentioned in FastKV/Speculative Prefill
    ) -> torch.Tensor:
        """
        Applies 1D average pooling to denoise scores, then selects the top-K chunks.
        This function implements the logic from Section 3.2.3 of the paper.
        """
        device = scores.device
        if original_seq_len <= num_to_keep:
            return torch.arange(original_seq_len, device=device, dtype=torch.long)

        # 1. Denoise with 1D Average Pooling
        # Reshape for avg_pool1d: (N, C_in, L_in) -> (batch_size=1, channels=1, length)
        scores_for_pooling = scores.view(1, 1, -1)
        
        # The paper mentions pooling before chunking.
        # We use padding to keep the tensor size the same.
        padding = (pool_kernel_size - 1) // 2
        smoothed_scores = F.avg_pool1d(
            scores_for_pooling,
            kernel_size=pool_kernel_size,
            stride=1,
            padding=padding
        ).squeeze()

        # 2. Chunk Selection
        # If the sequence is too short for chunking, fall back to simple top-k on smoothed scores
        if original_seq_len < chunk_size:
            return self._get_sorted_top_k_indices(smoothed_scores, num_to_keep, original_seq_len, device)

        # Pad scores to be divisible by chunk_size
        num_chunks = math.ceil(original_seq_len / chunk_size)
        padding_len = num_chunks * chunk_size - original_seq_len
        if padding_len > 0:
            # Pad with a very low value so these don't get selected
            padded_scores = F.pad(smoothed_scores, (0, padding_len), value=float('-inf'))
        else:
            padded_scores = smoothed_scores

        # Reshape into chunks and calculate the average score per chunk
        chunked_scores = padded_scores.view(num_chunks, chunk_size)
        avg_chunk_scores = chunked_scores.mean(dim=1)

        # Determine how many chunks to keep
        # We want to keep at least `num_to_keep` tokens
        num_chunks_to_keep = math.ceil(num_to_keep / chunk_size)
        num_chunks_to_keep = min(num_chunks_to_keep, num_chunks) # Cannot keep more chunks than exist

        # Select the top-K chunks
        _, top_chunk_indices = torch.topk(avg_chunk_scores, k=num_chunks_to_keep)

        # 3. Restore Final Token Indices
        # Create a tensor of all indices from the selected chunks
        selected_indices = []
        for chunk_idx in top_chunk_indices:
            start = chunk_idx * chunk_size
            end = start + chunk_size
            selected_indices.append(torch.arange(start, end, device=device))

        final_indices = torch.cat(selected_indices)

        # The selected indices might be longer than original_seq_len due to padding,
        # and might be more than num_to_keep due to chunk granularity.
        # We filter out padded indices and then, if needed, take top-k from the selected tokens.
        final_indices = final_indices[final_indices < original_seq_len]
        
        # The paper isn't explicit on what to do if chunking gives you > num_to_keep tokens.
        # A robust strategy is to re-score only the selected tokens and take the best.
        if len(final_indices) > num_to_keep:
            final_scores = smoothed_scores[final_indices]
            _, top_k_in_chunks_indices = torch.topk(final_scores, k=num_to_keep)
            final_indices = final_indices[top_k_in_chunks_indices]

        return torch.sort(final_indices)[0]

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
        
        # This will now store a list of score tensors, one for each look-ahead step.
        lookahead_step_scores = []
        
        # Store scores for the very first step separately for potential pruning
        first_step_layer_scores_list = [
            torch.zeros(batch_size, prompt_length, device=self.device, dtype=self.dtype)
            for _ in range(num_spec_layers)
        ]

        for spec_idx in range(len(self.captured_qs[0])): # Iterate over look-ahead steps
            
            # This will store the scores for each layer for the *current* look-ahead step
            current_step_layer_scores = []
            
            for layer_idx in range(num_spec_layers):
                if not self.captured_qs[layer_idx] or len(self.captured_qs[layer_idx]) <= spec_idx:
                    continue
                if speculator_prefill_cache_as_tuple[layer_idx][0].numel() == 0:
                    continue

                key_prompt_layer_rep_spec = hf_repeat_kv(
                    speculator_prefill_cache_as_tuple[layer_idx][0].detach(), 
                    spec_num_kv_groups
                )
                
                head_dim_from_q = self.captured_qs[layer_idx][spec_idx].shape[-1]
                
                query_lookahead_layer_step = self.captured_qs[layer_idx][spec_idx]
                
                attn_logits_layer_step = torch.matmul(
                    query_lookahead_layer_step, 
                    key_prompt_layer_rep_spec.transpose(-1, -2)
                ) / math.sqrt(head_dim_from_q)
                
                # According to the paper, take max over heads (dim=1)
                # The result is (batch_size, seq_len)
                layer_max_head_scores = attn_logits_layer_step.max(dim=1).values.squeeze(1)
                
                current_step_layer_scores.append(layer_max_head_scores)

                # Keep track of the first step scores separately
                if spec_idx == 0:
                    first_step_layer_scores_list[layer_idx] = layer_max_head_scores

            if current_step_layer_scores:
                # According to the paper, take max over layers (dim=0)
                # Stack along a new dimension and take the max
                step_final_scores = torch.stack(current_step_layer_scores).max(dim=0).values
                lookahead_step_scores.append(step_final_scores)

        if lookahead_step_scores:
            # According to the paper, average over look-ahead steps (dim=0)
            self.global_qk_scores = torch.stack(lookahead_step_scores).mean(dim=0)
        else:
            self.global_qk_scores = torch.zeros(batch_size, prompt_length, device=self.device, dtype=self.dtype)

        # For the first-step scores, we still use max over layers
        valid_first_step_layer_scores = [
            fsls for fsls in first_step_layer_scores_list
            if fsls.numel() > 0 and fsls.shape[0] == batch_size and fsls.shape[-1] == prompt_length
        ]
        if valid_first_step_layer_scores:
            self.first_step_prompt_qk_scores = torch.stack(valid_first_step_layer_scores).max(dim=0).values
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
        
        num_to_keep = min(num_to_keep, current_seq_len)
        
        if num_to_keep == 0:
            return torch.empty(0, dtype=torch.long, device=device)
        
        if scores.ndim > 1:
            scores = scores.squeeze() 
        
        if scores.shape[0] != current_seq_len:
            if current_seq_len == 0 and scores.numel() == 0:
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
    ) -> torch.Tensor:
        num_tokens_to_keep = min(self.max_capacity_prompt, original_seq_len)
        
        if num_tokens_to_keep <= 0:
            return torch.empty(0, dtype=torch.long, device=device)
        
        if num_tokens_to_keep >= original_seq_len:
            return torch.arange(original_seq_len, device=device, dtype=torch.long)
        
        # Use global_qk_scores as it incorporates look-ahead information.
        # The paper is a bit ambiguous, but this is a more robust choice.
        scores_for_selection = None
        if (self.global_qk_scores is not None and 
            self.global_qk_scores.shape[0] == batch_size and 
            self.global_qk_scores.shape[-1] == original_seq_len):
            scores_for_selection = self.global_qk_scores[0].clone()
        
        if scores_for_selection is None:
            raise ValueError("scores_for_selection is None")
        
        return self._denoise_and_select_chunks(
            scores=scores_for_selection,
            num_to_keep=num_tokens_to_keep,
            original_seq_len=original_seq_len
        )    

    def run(
        self,
        prompt_text: str,
        look_ahead_k: int,
        max_generation_length: int
    ) -> Tuple[str, Dict[str, Any]]: 
        run_metadata: Dict[str, Any] = {} 
        overall_start_time = time.perf_counter()

        run_metadata["max_capacity_prompt"] = self.max_capacity_prompt
        
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
        
        else: # Selective prefill
            final_indices_for_selective_prefill: Optional[torch.Tensor] = None
            
            if current_pruning_applicable:
                final_indices_for_selective_prefill = self._calculate_num_tokens_to_keep(
                    prompt_length, batch_size, self.device
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
    parser.add_argument("--max_capacity_prompt", type=int, default=512,
                        help="Maximum number of tokens to keep for base model prefill. Default: 512")
    parser.add_argument("--max_generation_length", type=int, default=32)
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
        max_capacity_prompt=args.max_capacity_prompt
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
    
    print(f"Max Capacity Prompt: {args.max_capacity_prompt}")
    print(f"Lookahead K: {args.look_ahead_k}")


if __name__ == "__main__":
    main()
