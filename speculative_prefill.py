import math
import sys
import os
import time
import argparse
import types
from functools import partial
from typing import List, Dict, Tuple, Optional, Any, Union

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    apply_rotary_pos_emb as hf_apply_rotary_pos_emb,
    repeat_kv as hf_repeat_kv
)
from transformers.cache_utils import Cache, DynamicCache
from datasets import load_dataset

# --- QTIP Imports (preserved for compatibility) ---
sys.path.append(os.path.join(os.path.dirname(__file__), 'qtip'))
try:
    from lib.utils.unsafe_import import model_from_hf_path as qtip_model_from_hf_path
    from model.llama import LlamaAttention as QTIP_LlamaAttention
    QTIP_AVAILABLE = True
except ImportError:
    QTIP_AVAILABLE = False
    qtip_model_from_hf_path = None
    QTIP_LlamaAttention = None

# --- Attention Forward Methods ---
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
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
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
    
    # --- Start of Major Fix 1 ---
    # Capture Q-vector from the last token of the prefill pass, or the single token of a decode pass.
    if pipeline_instance.is_prefilling and query_length > 1:
        last_q_vector = query_states_rotated[:, :, -1:, :].detach().clone()
        pipeline_instance.captured_qs[self_attn.layer_idx].append(last_q_vector)
    elif not pipeline_instance.is_prefilling and query_length == 1:
        pipeline_instance.captured_qs[self_attn.layer_idx].append(query_states_rotated.detach().clone())
    # --- End of Major Fix 1 ---
    
    cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
    key_states_for_sdpa, value_states_for_sdpa = past_key_value.update(
        key_states_rotated, value_states_for_cache, self_attn.layer_idx, cache_kwargs
    )
    
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
        is_causal=is_sdpa_causal
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
    
    # --- Start of Major Fix 1 (QTIP version) ---
    if pipeline_instance.is_prefilling and query_length > 1:
        last_q_vector = query_states_rotated[:, :, -1:, :].detach().clone()
        pipeline_instance.captured_qs[self_attn.layer_idx].append(last_q_vector)
    elif not pipeline_instance.is_prefilling and query_length == 1:
        pipeline_instance.captured_qs[self_attn.layer_idx].append(query_states_rotated.detach().clone())
    # --- End of Major Fix 1 (QTIP version) ---

    cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
    key_states_for_sdpa, value_states_for_sdpa = past_key_value.update(
        key_states_rotated, value_states_for_cache, self_attn.layer_idx, cache_kwargs
    )
    
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
        is_causal=is_sdpa_causal
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
        speculator_model_name: str,
        max_capacity_prompt: int = 512,
        pool_kernel_size: Optional[int] = 7,
        pool_type: str = 'avgpool',
        use_chunk_selection: bool = True,  # Major Fix 2: Added parameter
        chunk_size: int = 64,
    ):
        self.base_model_name = base_model_name
        self.speculator_model_name = speculator_model_name
        self.max_capacity_prompt = max_capacity_prompt
        
        self.pool_kernel_size = pool_kernel_size
        self.pool_type = pool_type.lower()
        self.use_chunk_selection = use_chunk_selection # Major Fix 2: Store parameter
        self.chunk_size = chunk_size
        self._validate_config()
        
        self.tokenizer = self._load_tokenizer()
        self.speculator_model = self._load_model(self.speculator_model_name)
        self.base_model = self._load_model(self.base_model_name)
        self.base_config: AutoConfig = self.base_model.config
        
        self.device = self.speculator_model.device
        self.dtype = self.speculator_model.dtype
        self.eos_token_ids = self._extract_eos_token_ids(self.base_config.eos_token_id)
        
        self.captured_qs: List[List[torch.Tensor]] = []
        self.orig_spec_fwds: Dict[int, Any] = {}
        self.is_prefilling = False # Major Fix 1: Added flag
        self.token_importance_scores: Optional[torch.Tensor] = None

    def _validate_config(self):
        if self.pool_type not in ['avgpool', 'maxpool', 'none']:
            raise ValueError(f"pool_type must be 'avgpool', 'maxpool', or 'none', but got {self.pool_type}")
        if self.pool_kernel_size is not None:
            if self.pool_kernel_size <= 1:
                self.pool_kernel_size = None
            elif self.pool_kernel_size % 2 == 0:
                raise ValueError("pool_kernel_size must be an odd number for symmetric padding.")
            if self.pool_type == 'none':
                 raise ValueError("pool_kernel_size is specified, but pool_type is 'none'. Set pool_kernel_size to None.")
        if self.pool_type != 'none' and self.pool_kernel_size is None:
            raise ValueError(f"pool_type is '{self.pool_type}', but pool_kernel_size is not specified.")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer.")
    
    def _extract_eos_token_ids(self, eos_token_id: Union[int, List[int], None]) -> List[int]:
        if isinstance(eos_token_id, int): return [eos_token_id]
        if isinstance(eos_token_id, list): return list(eos_token_id)
        if self.tokenizer.eos_token_id: return [self.tokenizer.eos_token_id]
        raise ValueError("eos_token_id is not defined in the model config or tokenizer.")
    
    def _is_qtip_model(self, model_name: str) -> bool:
        return QTIP_AVAILABLE and ("relaxml" in model_name.lower() or "qtip" in model_name.lower())
    
    def _load_model(self, model_name: str) -> AutoModelForCausalLM:
        if self._is_qtip_model(model_name):
            if not QTIP_AVAILABLE:
                raise ImportError(f"QTIP model requested ({model_name}) but QTIP modules are not available.")
            model, _ = qtip_model_from_hf_path(model_name, max_mem_ratio=0.7, attn_implementation="sdpa")
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                device_map="auto",
                attn_implementation="sdpa"
            )
        return model.eval()
    
    def _get_tokenizer_for_model(self, model_name: str) -> str:
        if "qtip" in model_name.lower():
            if "llama-3.1" in model_name.lower(): return "meta-llama/Llama-3.1-8B-Instruct"
            if "llama-3" in model_name.lower(): return "meta-llama/Meta-Llama-3-8B-Instruct"
            return "meta-llama/Llama-2-7b-hf"
        return model_name
    
    def _load_tokenizer(self):
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.base_model_name, trust_remote_code=True)
        except OSError:
            fallback_name = self._get_tokenizer_for_model(self.base_model_name)
            tokenizer = AutoTokenizer.from_pretrained(fallback_name, trust_remote_code=True)
        
        if tokenizer.pad_token is None:
            if tokenizer.eos_token:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                raise ValueError("Tokenizer has no EOS token to use as a PAD token.")
        return tokenizer
    
    def _patch_speculator(self) -> int:
        if not hasattr(self.speculator_model, 'model') or not hasattr(self.speculator_model.model, 'layers'):
            return 0
            
        num_layers = self.speculator_model.config.num_hidden_layers
        self.captured_qs = [[] for _ in range(num_layers)]
        self.token_importance_scores = None
        num_patched_layers = 0

        for i, layer in enumerate(self.speculator_model.model.layers):
            if hasattr(layer, 'self_attn'):
                attn = layer.self_attn
                patch_method = None
                if isinstance(attn, LlamaAttention):
                    patch_method = _hf_patched_attention_forward_method
                elif QTIP_AVAILABLE and QTIP_LlamaAttention and isinstance(attn, QTIP_LlamaAttention):
                    patch_method = _qtip_patched_attention_forward_method
                
                if patch_method:
                    if i not in self.orig_spec_fwds: self.orig_spec_fwds[i] = attn.forward
                    attn.forward = types.MethodType(partial(patch_method, pipeline_instance=self), attn)
                    num_patched_layers += 1
        return num_patched_layers
    
    def _token_importance_from_attn_scores(self, attention_scores: torch.Tensor):
        # Input shape: [batch_size, num_layers, num_heads, k+1, prompt_len]
        if attention_scores.numel() == 0:
            raise RuntimeError("Cannot calculate importance from empty attention scores.")
            
        bs, num_layers, num_heads, num_steps, key_len = attention_scores.shape
        if bs != 1:
            raise NotImplementedError("Batch size > 1 is not supported for importance calculation yet.")

        # Squeeze batch dim for simplicity
        all_layers_tensor = attention_scores.squeeze(0) # -> [num_layers, num_heads, k+1, key_len]

        original_dtype = all_layers_tensor.dtype
        all_layers_tensor = F.softmax(all_layers_tensor, dim=-1, dtype=torch.float32).to(original_dtype)
        
        # Reshape for pooling: [num_layers * num_heads, k+1, key_len]
        flattened_tensor = all_layers_tensor.reshape(num_layers * num_heads, num_steps, key_len)

        if self.pool_kernel_size and self.pool_type != 'none':
            # Pool across the key_len dimension for each step
            # Input to pooling: [N, C, L_in], requires reshape
            # [num_layers * num_heads * k+1, 1, key_len]
            to_pool = flattened_tensor.reshape(-1, 1, key_len)

            padding = (self.pool_kernel_size - 1) // 2
            pool_fn = F.avg_pool1d if self.pool_type == 'avgpool' else F.max_pool1d
            pooled_tensor = pool_fn(to_pool, kernel_size=self.pool_kernel_size, stride=1, padding=padding)
            
            # Reshape back to [num_layers * num_heads, k+1, key_len]
            pooled_tensor = pooled_tensor.reshape(num_layers * num_heads, num_steps, key_len)
        else:
            pooled_tensor = flattened_tensor
        
        # Aggregate scores: max over layers and heads, then mean over lookahead steps
        step_final_scores = pooled_tensor.max(dim=0).values # -> [k+1, key_len]
        self.token_importance_scores = step_final_scores.mean(dim=0).unsqueeze(0) # -> [1, key_len]

    def _compute_raw_qk_scores(self, speculator_prefill_cache_as_tuple) -> torch.Tensor:
        if not self.captured_qs or not any(self.captured_qs):
            raise RuntimeError("Speculator Q-vectors were not captured for score computation.")

        spec_config = self.speculator_model.config
        num_spec_layers = spec_config.num_hidden_layers
        spec_num_q_heads = spec_config.num_attention_heads
        spec_num_kv_heads = spec_config.num_key_value_heads
        spec_num_kv_groups = spec_num_q_heads // spec_num_kv_heads
        
        all_layer_scores = []
        for layer_idx in range(num_spec_layers):
            if not self.captured_qs[layer_idx] or speculator_prefill_cache_as_tuple[layer_idx][0].numel() == 0:
                continue
            
            # Key cache from initial prefill, shape: [bs, num_kv_heads, prompt_len, head_dim]
            key_prompt_layer_rep_spec = hf_repeat_kv(speculator_prefill_cache_as_tuple[layer_idx][0].detach(), spec_num_kv_groups)
            
            # Captured Q-vectors are a list of tensors, each [bs, num_q_heads, 1, head_dim]
            # Concatenate them along the sequence length dimension (dim=2)
            all_q_for_layer = torch.cat(self.captured_qs[layer_idx], dim=2) # -> [bs, num_q_heads, k+1, head_dim]
            
            head_dim_from_q = all_q_for_layer.shape[-1]
            attn_logits = torch.matmul(all_q_for_layer, key_prompt_layer_rep_spec.transpose(-1, -2)) / math.sqrt(head_dim_from_q)
            all_layer_scores.append(attn_logits)
        
        # Stack into a single tensor: [bs, num_layers, num_heads, k+1, prompt_len]
        return torch.stack(all_layer_scores, dim=1)
        
    def _select_tokens_by_chunk(self, scores: torch.Tensor, num_to_keep: int, original_seq_len: int) -> torch.Tensor:
        device = scores.device
        if original_seq_len <= num_to_keep:
            return torch.arange(original_seq_len, device=device)

        num_chunks = math.ceil(original_seq_len / self.chunk_size)
        padding_len = num_chunks * self.chunk_size - original_seq_len
        if padding_len > 0:
            padded_scores = F.pad(scores, (0, padding_len), value=float('-inf'))
        else:
            padded_scores = scores
        
        chunked_scores = padded_scores.view(num_chunks, self.chunk_size)
        avg_chunk_scores = chunked_scores.mean(dim=1)
        
        percentage_to_keep = num_to_keep / original_seq_len
        num_chunks_to_keep = min(math.ceil(num_chunks * percentage_to_keep), num_chunks)

        _, top_chunk_indices = torch.topk(avg_chunk_scores, k=num_chunks_to_keep)
        
        selected_indices = [
            torch.arange(
                idx * self.chunk_size, (idx + 1) * self.chunk_size, device=device
            ) for idx in top_chunk_indices
        ]
        final_indices = torch.cat(selected_indices)
        final_indices = final_indices[final_indices < original_seq_len]
        
        if len(final_indices) > num_to_keep:
            final_scores = scores[final_indices]
            _, top_k_in_chunks_indices = torch.topk(final_scores, k=num_to_keep)
            final_indices = final_indices[top_k_in_chunks_indices]

        return torch.sort(final_indices)[0]

    def _calculate_indices_to_keep(self, original_seq_len: int) -> torch.Tensor:
        num_to_keep = min(self.max_capacity_prompt, original_seq_len)
        if num_to_keep >= original_seq_len:
            return torch.arange(original_seq_len, device=self.device, dtype=torch.long)
        
        if self.token_importance_scores is None:
            raise RuntimeError("Token importance scores have not been computed.")
        
        scores_for_selection = self.token_importance_scores[0].clone()

        # --- Start of Major Fix 2 ---
        if self.use_chunk_selection:
            return self._select_tokens_by_chunk(scores_for_selection, num_to_keep, original_seq_len)
        else:
            # Simple top-k selection across the entire sequence
            _, indices = torch.topk(scores_for_selection, k=num_to_keep, dim=-1)
            return torch.sort(indices)[0]
        # --- End of Major Fix 2 ---

    def run(
        self, prompt_text: str, look_ahead_k: int, max_generation_length: int
    ) -> Tuple[str, Dict[str, Any]]: 
        run_metadata: Dict[str, Any] = {"max_capacity_prompt": self.max_capacity_prompt}
        overall_start_time = time.perf_counter()
        
        num_patched_layers = self._patch_speculator()
        if num_patched_layers == 0:
            raise RuntimeError("Speculator model could not be patched for Q-vector capture.")

        limit_due_to_base = self.base_config.max_position_embeddings - max_generation_length
        limit_due_to_speculator = self.speculator_model.config.max_position_embeddings - look_ahead_k
        max_prompt_len_calc = float(min(limit_due_to_base, limit_due_to_speculator) - self.POSITION_BUFFER)
        max_prompt_length = max(1, int(max_prompt_len_calc))
        
        messages = [{"role": "user", "content": prompt_text}]
        templated_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = self.tokenizer(
            templated_prompt, return_tensors="pt", padding=False, truncation=True, max_length=max_prompt_length
        ).to(self.device)
        prompt_input_ids = inputs.input_ids
        prompt_length = prompt_input_ids.shape[1]
        
        run_metadata["prompt_input_length"] = prompt_length 
        if prompt_length == 0:
            run_metadata["total_time"] = time.perf_counter() - overall_start_time
            return "", run_metadata
        
        # --- Stage 1: Speculator Prefill ---
        stage_start_time = time.perf_counter()
        self.is_prefilling = True # Major Fix 1: Set flag
        with torch.no_grad():
            spec_out = self.speculator_model(input_ids=prompt_input_ids, use_cache=True, cache_position=torch.arange(prompt_length, device=self.device))
        self.is_prefilling = False # Major Fix 1: Unset flag
        
        legacy_cache_tuple = spec_out.past_key_values
        if legacy_cache_tuple is None: raise RuntimeError("Speculator prefill did not return a KV cache.")
        speculator_prefill_cache = DynamicCache.from_legacy_cache(legacy_cache_tuple)
        
        speculator_next_token_ids = torch.argmax(spec_out.logits[:, -1, :], dim=-1, keepdim=True)
        speculation_prefill_time = time.perf_counter() - stage_start_time
        
        # --- Stage 2: Speculator Lookahead & Scoring ---
        stage_start_time = time.perf_counter()
        with torch.no_grad():
            current_spec_tokens, current_spec_cache = speculator_next_token_ids, speculator_prefill_cache
            for i in range(look_ahead_k):
                cache_len = current_spec_cache.get_seq_length(0)
                pos_ids = torch.tensor([[cache_len]], device=self.device)
                lookahead_out = self.speculator_model(
                    input_ids=current_spec_tokens, 
                    position_ids=pos_ids, 
                    past_key_values=current_spec_cache, 
                    use_cache=True, 
                    cache_position=torch.tensor([cache_len], device=self.device)
                )
                current_spec_cache = lookahead_out.past_key_values
                current_spec_tokens = torch.argmax(lookahead_out.logits[:, -1, :], dim=-1, keepdim=True)
                if current_spec_tokens.item() in self.eos_token_ids: break
        
        speculation_decode_time = time.perf_counter() - stage_start_time
        
        raw_qk_scores = self._compute_raw_qk_scores(legacy_cache_tuple)
        self._token_importance_from_attn_scores(raw_qk_scores)
        
        # --- Stage 3: Base Model Prefill ---
        base_model_prefill_start_time = time.perf_counter()
        
        indices_to_keep = self._calculate_indices_to_keep(prompt_length)
        prefill_input_ids = prompt_input_ids[:, indices_to_keep]
        prefill_position_ids = indices_to_keep.unsqueeze(0).to(torch.long)
        
        run_metadata["selective_prefill_original_len"] = prompt_length
        run_metadata["selective_prefill_kept_token_count"] = prefill_input_ids.shape[1]
        run_metadata["token_keep_rate"] = (prefill_input_ids.shape[1] / prompt_length * 100.0) if prompt_length > 0 else 100.0

        with torch.no_grad():
            base_out = self.base_model(input_ids=prefill_input_ids, position_ids=prefill_position_ids, use_cache=True, cache_position=torch.arange(prefill_input_ids.shape[1], device=self.device))
        
        base_model_next_token_ids = torch.argmax(base_out.logits[:, -1, :], dim=-1, keepdim=True)
        base_model_cache_after_prefill = base_out.past_key_values
        
        base_model_prefill_time = time.perf_counter() - base_model_prefill_start_time
        
        # --- Stage 4: Base Model Generation ---
        run_metadata["speculation_prefill"] = speculation_prefill_time
        run_metadata["speculation_decode"] = speculation_decode_time
        run_metadata["base_prefill"] = base_model_prefill_time
        run_metadata["base_ttft"] = speculation_prefill_time + speculation_decode_time + base_model_prefill_time
        
        gen_token_ids_list: List[int] = []
        if base_model_next_token_ids is not None and base_model_cache_after_prefill is not None:
            gen_token_ids_list.append(base_model_next_token_ids.item())
            current_decode_tokens, current_decode_kv_cache = base_model_next_token_ids, base_model_cache_after_prefill
            
            if gen_token_ids_list[-1] not in self.eos_token_ids:
                for i in range(max_generation_length - 1):
                    current_real_pos = prefill_input_ids.shape[1] + i
                    pos_ids = torch.tensor([[current_real_pos]], device=self.device)
                    with torch.no_grad():
                        decode_out = self.base_model(
                            current_decode_tokens, 
                            position_ids=pos_ids, 
                            past_key_values=current_decode_kv_cache, 
                            use_cache=True, 
                            cache_position=torch.tensor([current_real_pos], device=self.device)
                        )
                    
                    next_tokens = torch.argmax(decode_out.logits[:, -1, :], dim=-1, keepdim=True)
                    gen_token_ids_list.append(next_tokens.item())
                    current_decode_tokens, current_decode_kv_cache = next_tokens, decode_out.past_key_values
                    if gen_token_ids_list[-1] in self.eos_token_ids: break

        final_gen_text = self.tokenizer.decode(gen_token_ids_list, skip_special_tokens=True)
        run_metadata["total_time"] = time.perf_counter() - overall_start_time
        
        if final_gen_text.startswith("assistant\n\n"): final_gen_text = final_gen_text[len("assistant\n\n"):]
        elif final_gen_text.startswith(" assistant\n"): final_gen_text = final_gen_text[len(" assistant\n"):]
        return final_gen_text, run_metadata


def main():
    parser = argparse.ArgumentParser(description="Speculative Prefill Pipeline with Hugging Face Transformers")
    parser.add_argument("--base_model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Base model for generation.")
    parser.add_argument("--speculator_model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="Smaller speculator model.")
    parser.add_argument("--dataset_name", type=str, default="hotpotqa", help="Dataset from THUDM/LongBench to use for the prompt (e.g., 'hotpotqa').")
    parser.add_argument("--look_ahead_k", type=int, default=8, help="Number of lookahead steps for the speculator.")
    parser.add_argument("--max_capacity_prompt", type=int, default=512, help="Token budget for the pruned prompt.")
    parser.add_argument("--max_generation_length", type=int, default=32, help="Maximum number of tokens to generate.")
    parser.add_argument("--kernel_size", type=int, default=13, help="Kernel size for pooling importance scores. Must be odd. Use <=1 for no pooling.")
    parser.add_argument("--pooling", type=str, default="avgpool", choices=['avgpool', 'maxpool', 'none'], help="Type of pooling to apply to attention scores.")
    parser.add_argument("--no_chunk_selection", action='store_false', dest='use_chunk_selection', help="Disable chunk-based selection and use simple top-k instead.") # Major Fix 2
    parser.add_argument("--chunk_size", type=int, default=32, help="Size of chunks for chunk-based selection.")
    
    args = parser.parse_args()
    
    pipeline = SpeculativePrefillPipeline(
        base_model_name=args.base_model_name,
        speculator_model_name=args.speculator_model_name,
        max_capacity_prompt=args.max_capacity_prompt,
        pool_kernel_size=args.kernel_size,
        pool_type=args.pooling,
        use_chunk_selection=args.use_chunk_selection, # Major Fix 2
        chunk_size=args.chunk_size,
    )
    
    prompt_str: str
    try:
        dataset = load_dataset('THUDM/LongBench', args.dataset_name, split='test', trust_remote_code=True)
        sample = dataset[0]
        context = sample.get('context', '')
        input_text = sample.get('input', '')
        prompt_str = f"Context: {context}\nQuestion: {input_text}\nAnswer:"
    except Exception as e:
        print(f"Could not load dataset '{args.dataset_name}'. Using a default prompt. Error: {e}")
        prompt_str = "Explain the theory of relativity in simple terms."
    
    selection_strategy = f"Chunk-based (size={pipeline.chunk_size})" if pipeline.use_chunk_selection else "Simple Top-K"
    print(f"\n--- Running Speculative Prefill Pipeline ---")
    print(f"Base Model: {args.base_model_name}")
    print(f"Speculator Model: {args.speculator_model_name}")
    print(f"Max Prompt Capacity: {args.max_capacity_prompt}")
    print(f"Lookahead K: {args.look_ahead_k}")
    print(f"Pooling: type='{pipeline.pool_type}', kernel_size={pipeline.pool_kernel_size}")
    print(f"Token Selection: {selection_strategy}")
    print("-" * 43)

    generated_text, run_metadata = pipeline.run(
        prompt_text=prompt_str,
        look_ahead_k=args.look_ahead_k,
        max_generation_length=args.max_generation_length
    )
    
    print(f"\n--- Generated Text ---")
    print(generated_text)
    print("-" * 22)

    print("\n--- Performance Metrics ---")
    print(f"Prompt Length (Tokens): {run_metadata.get('prompt_input_length', 'N/A')}")
    print(f"Kept for Prefill (Tokens): {run_metadata.get('selective_prefill_kept_token_count', 'N/A')}")
    print(f"Token Keep Rate: {run_metadata.get('token_keep_rate', 0):.2f}%")
    print(f"Time to First Token (TTFT): {run_metadata.get('base_ttft', 0):.4f} seconds")
    print(f"  - Speculator Prefill: {run_metadata.get('speculation_prefill', 0):.4f} s")
    print(f"  - Speculator Decode/Scoring: {run_metadata.get('speculation_decode', 0):.4f} s")
    print(f"  - Base Model Prefill: {run_metadata.get('base_prefill', 0):.4f} s")
    print(f"Total Pipeline Time: {run_metadata.get('total_time', 0):.4f} seconds")
    print("-" * 27)


if __name__ == "__main__":
    main()
