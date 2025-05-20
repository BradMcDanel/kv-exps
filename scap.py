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
from typing import List, Dict, Tuple, Optional, Any
import math
import argparse
import time
from functools import partial
import random
from datasets import load_dataset

# Patched attention method for SCAP
def _scap_patched_attention_forward_method(
    self_attn: LlamaAttention,
    pipeline_instance: 'SlicedContextAggregationPipeline',
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs: Any
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

    bsz, q_len, _ = hidden_states.size()
    print(bsz)

    query_projection = self_attn.q_proj(hidden_states)
    key_projection = self_attn.k_proj(hidden_states)
    value_projection = self_attn.v_proj(hidden_states)

    query_states = query_projection.view(bsz, q_len, self_attn.config.num_attention_heads, self_attn.head_dim).transpose(1, 2)
    key_states_before_rope = key_projection.view(bsz, q_len, self_attn.config.num_key_value_heads, self_attn.head_dim).transpose(1, 2)
    value_states_for_cache = value_projection.view(bsz, q_len, self_attn.config.num_key_value_heads, self_attn.head_dim).transpose(1, 2)
    
    assert position_embeddings is not None, "position_embeddings must be provided to the patched attention"
    cos, sin = position_embeddings
    query_states_rotated, key_states_rotated = hf_apply_rotary_pos_emb(query_states, key_states_before_rope, cos, sin)

    if pipeline_instance.is_capturing_queries_for_slice_importance and q_len == 1: # q_len is 1 for lookahead token
        pipeline_instance.captured_lookahead_queries_buffer[self_attn.layer_idx] = query_states_rotated.detach().clone()


    if use_cache:
        assert past_key_value is not None, "past_key_value must be provided when use_cache is True"
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states_for_sdpa, value_states_for_sdpa = past_key_value.update(key_states_rotated, value_states_for_cache, self_attn.layer_idx, cache_kwargs)
    else:
        key_states_for_sdpa, value_states_for_sdpa = key_states_rotated, value_states_for_cache

    key_states_for_sdpa = hf_repeat_kv(key_states_for_sdpa, self_attn.num_key_value_groups)
    value_states_for_sdpa = hf_repeat_kv(value_states_for_sdpa, self_attn.num_key_value_groups)

    attn_mask_input = attention_mask
    is_sdpa_causal = False 
    
    if attn_mask_input is not None:
        actual_kv_seq_len = key_states_for_sdpa.shape[-2]
        mask_kv_seq_len = attn_mask_input.shape[-1]

        if mask_kv_seq_len > actual_kv_seq_len:
            attn_mask_input = attn_mask_input[..., :actual_kv_seq_len]
        elif mask_kv_seq_len < actual_kv_seq_len:
            padding_needed = actual_kv_seq_len - mask_kv_seq_len
            pad_value = torch.finfo(attn_mask_input.dtype).min
            attn_mask_input = F.pad(attn_mask_input, (0, padding_needed), mode='constant', value=pad_value)

    attention_output = F.scaled_dot_product_attention(
        query_states_rotated, 
        key_states_for_sdpa, 
        value_states_for_sdpa, 
        attn_mask=attn_mask_input, 
        dropout_p=self_attn.attention_dropout if self_attn.training else 0.0, 
        is_causal=is_sdpa_causal, 
        **kwargs 
    )
    
    attention_output = attention_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self_attn.o_proj.in_features)
    attention_output = self_attn.o_proj(attention_output)
    
    return attention_output, None


class SlicedContextAggregationPipeline:
    def __init__(self, base_model_name: str,
                 n_slices: int, k_percentage: float,
                 n_start_tokens: int, n_end_tokens: int,
                 chunk_size: int, 
                 share_kv_cache: bool = False):
        self.base_model_name = base_model_name
        self.n_slices = n_slices
        self.k_percentage = k_percentage 
        self.n_start_tokens = n_start_tokens
        self.n_end_tokens = n_end_tokens
        self.chunk_size = chunk_size 
        self.share_kv_cache = share_kv_cache

        self.base_config = AutoConfig.from_pretrained(base_model_name, trust_remote_code=True)
        self.tokenizer = self._load_tokenizer()
        self.base_model = self._load_model_with_config(base_model_name, "eager", self.base_config)
        self.device = self.base_model.device
        self.dtype = self.base_model.dtype
        
        self.eos_token_ids: List[int] = []
        if self.tokenizer.eos_token_id is not None:
            self.eos_token_ids.append(self.tokenizer.eos_token_id)
        # Handle cases where eos_token_id might be a list in config
        # (though AutoTokenizer usually resolves this to a single int if it's unique)
        # For safety, check if base_config also has eos_token_id list
        config_eos = getattr(self.base_config, 'eos_token_id', None)
        if isinstance(config_eos, list):
            for tid in config_eos:
                if tid not in self.eos_token_ids:
                    self.eos_token_ids.append(tid)
        elif isinstance(config_eos, int) and config_eos not in self.eos_token_ids:
             self.eos_token_ids.append(config_eos)


        self.captured_lookahead_queries_buffer: List[torch.Tensor] = [] 
        self.orig_base_attention_fwds: Dict[int, Any] = {}
        self.is_capturing_queries_for_slice_importance = False
        
        self.actual_n_start_for_run: int = 0 
        self.actual_n_end_for_run: int = 0   


    def _load_tokenizer(self) -> AutoTokenizer:
        tok = AutoTokenizer.from_pretrained(self.base_model_name, trust_remote_code=True)
        if tok.pad_token is None and tok.eos_token is not None:
            tok.pad_token = tok.eos_token
        assert tok.pad_token_id is not None, "Tokenizer must have a pad_token_id for batching."
        return tok

    def _load_model_with_config(self, model_name: str, attn_impl: Optional[str], config_obj: AutoConfig) -> AutoModelForCausalLM:
        load_kwargs: Dict[str, Any] = {
            "torch_dtype": "auto", 
            "config": config_obj,
            "trust_remote_code": True,
            "device_map": "auto"
        }
        if attn_impl is not None:
            load_kwargs["attn_implementation"] = attn_impl
        
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs) # type: ignore
        return model.eval() # type: ignore

    def _patch_base_model(self) -> int:
        assert hasattr(self.base_model, 'model') and hasattr(self.base_model.model, 'layers'), \
            "Base model does not have the expected structure for patching."
        
        num_patched_layers = 0
        layers_module = self.base_model.model.layers # type: ignore
        assert isinstance(layers_module, (torch.nn.ModuleList, list)), "Model layers are not an iterable list."

        for i, layer in enumerate(layers_module): 
            if hasattr(layer, 'self_attn') and isinstance(layer.self_attn, LlamaAttention):
                attention_module = layer.self_attn
                if i not in self.orig_base_attention_fwds:
                    self.orig_base_attention_fwds[i] = attention_module.forward
                
                partially_applied_func = partial(_scap_patched_attention_forward_method, pipeline_instance=self)
                attention_module.forward = types.MethodType(partially_applied_func, attention_module)
                num_patched_layers +=1
        return num_patched_layers

    def _unpatch_base_model(self):
        if not hasattr(self.base_model, 'model') or not hasattr(self.base_model.model, 'layers'): return
        layers_module = self.base_model.model.layers # type: ignore
        if not isinstance(layers_module, (torch.nn.ModuleList, list)): return

        for i, layer in enumerate(layers_module): 
            if hasattr(layer, 'self_attn') and isinstance(layer.self_attn, LlamaAttention):
                if i in self.orig_base_attention_fwds:
                    layer.self_attn.forward = self.orig_base_attention_fwds[i]
        self.orig_base_attention_fwds.clear()
        
    def _generate_slices(self, original_input_ids: torch.Tensor, prompt_length: int) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        all_slices_token_ids: List[torch.Tensor] = []
        all_slices_position_ids: List[torch.Tensor] = [] 

        self.actual_n_start_for_run = min(self.n_start_tokens, prompt_length)
        self.actual_n_end_for_run = min(self.n_end_tokens, max(0, prompt_length - self.actual_n_start_for_run))
        
        middle_start_original_idx = self.actual_n_start_for_run
        middle_end_original_idx = prompt_length - self.actual_n_end_for_run
        num_middle_tokens_prompt = max(0, middle_end_original_idx - middle_start_original_idx)

        if num_middle_tokens_prompt > 0:
            num_middle_chunks = math.ceil(num_middle_tokens_prompt / self.chunk_size)
            global_middle_chunk_start_indices = torch.arange(
                middle_start_original_idx, 
                middle_start_original_idx + num_middle_chunks * self.chunk_size, 
                self.chunk_size, 
                device=self.device
            )[:num_middle_chunks]
        else:
            num_middle_chunks = 0
            global_middle_chunk_start_indices = torch.empty(0, dtype=torch.long, device=self.device)
        
        for _ in range(self.n_slices): 
            current_slice_original_indices_set = set()
            for i in range(self.actual_n_start_for_run): current_slice_original_indices_set.add(i)
            
            if num_middle_chunks > 0:
                num_chunks_to_sample = math.ceil(self.k_percentage * num_middle_chunks)
                if num_chunks_to_sample > 0 and global_middle_chunk_start_indices.numel() > 0:
                    num_chunks_to_sample = min(num_chunks_to_sample, global_middle_chunk_start_indices.numel())
                    perm = torch.randperm(global_middle_chunk_start_indices.numel(), device=self.device)
                    selected_chunk_global_starts = global_middle_chunk_start_indices[perm[:num_chunks_to_sample]]
                    for chunk_start_idx_tensor in selected_chunk_global_starts:
                        chunk_start_idx = chunk_start_idx_tensor.item()
                        chunk_actual_end = min(chunk_start_idx + self.chunk_size, middle_end_original_idx)
                        for token_idx in range(chunk_start_idx, chunk_actual_end):
                            current_slice_original_indices_set.add(token_idx)
            
            for i in range(prompt_length - self.actual_n_end_for_run, prompt_length): current_slice_original_indices_set.add(i)
            
            if not current_slice_original_indices_set and prompt_length > 0 :
                 current_slice_original_indices_set.update(range(prompt_length)) 

            sorted_indices_for_slice_j = torch.tensor(sorted(list(current_slice_original_indices_set)), device=self.device, dtype=torch.long)
            if sorted_indices_for_slice_j.numel() == 0: continue

            all_slices_token_ids.append(original_input_ids[0, sorted_indices_for_slice_j]) 
            all_slices_position_ids.append(sorted_indices_for_slice_j) 
        
        assert all_slices_token_ids or prompt_length == 0, f"No slices generated for prompt_length {prompt_length}."
        return all_slices_token_ids, all_slices_position_ids

    def run(self, prompt_text: str, prompt_keep_percentage: float, max_generation_length: int) -> Tuple[str, Dict[str, float]]:
        timing_info: Dict[str, float] = {}
        if self.device.type == 'cuda': torch.cuda.synchronize()
        overall_start_time = time.perf_counter()

        self._patch_base_model()
        max_prompt_len_calculated = self.base_model.config.max_position_embeddings - max_generation_length - 20
        max_prompt_length = max(1, int(max_prompt_len_calculated))
        inputs = self.tokenizer(prompt_text, return_tensors="pt", padding=False, truncation=True, max_length=max_prompt_length).to(self.device)
        original_input_ids, prompt_length, batch_size = inputs.input_ids, inputs.input_ids.shape[1], inputs.input_ids.shape[0]
        assert batch_size == 1, "SCAP currently supports batch_size=1 for the overall prompt."

        if prompt_length == 0:
            self._unpatch_base_model()
            if self.device.type == 'cuda': torch.cuda.synchronize()
            timing_info["total_time"] = time.perf_counter() - overall_start_time
            return "", timing_info

        if self.device.type == 'cuda': torch.cuda.synchronize()
        slice_gen_start_time = time.perf_counter()
        all_slices_token_ids_unpadded, all_slices_position_ids_unpadded = self._generate_slices(original_input_ids, prompt_length)
        if self.device.type == 'cuda': torch.cuda.synchronize()
        timing_info["slice_generation"] = time.perf_counter() - slice_gen_start_time
        
        if not all_slices_token_ids_unpadded: 
            if prompt_length > 0: 
                 all_slices_token_ids_unpadded.append(original_input_ids.squeeze(0))
                 all_slices_position_ids_unpadded.append(torch.arange(prompt_length, device=self.device))
            else:
                 self._unpatch_base_model(); return "", {"total_time": time.perf_counter() - overall_start_time}
        
        num_actual_slices = len(all_slices_token_ids_unpadded)

        original_slice_lengths = torch.tensor([s.shape[0] for s in all_slices_token_ids_unpadded], device=self.device)
        max_slice_len = original_slice_lengths.max().item() if num_actual_slices > 0 else 0
        
        batched_slice_input_ids = torch.full((num_actual_slices, max_slice_len), self.tokenizer.pad_token_id, dtype=torch.long, device=self.device)
        batched_slice_pos_ids = torch.full((num_actual_slices, max_slice_len), 0, dtype=torch.long, device=self.device) 
        # For the prefill pass, the attention_mask should be based on padded tokens
        batched_2d_attention_mask_for_prefill = torch.zeros((num_actual_slices, max_slice_len), dtype=torch.bool, device=self.device)


        for i in range(num_actual_slices):
            s_len = original_slice_lengths[i].item()
            batched_slice_input_ids[i, :s_len] = all_slices_token_ids_unpadded[i]
            batched_slice_pos_ids[i, :s_len] = all_slices_position_ids_unpadded[i] 
            batched_2d_attention_mask_for_prefill[i, :s_len] = True # True means not masked
        
        slice_prefill_cache_pos_for_model_fwd = torch.arange(max_slice_len, device=self.device)
        
        if self.device.type == 'cuda': torch.cuda.synchronize()
        importance_calc_start_time = time.perf_counter()
        
        with torch.no_grad():
            batched_outputs_S_j = self.base_model(
                input_ids=batched_slice_input_ids, 
                attention_mask=batched_2d_attention_mask_for_prefill, # Pass 2D mask for padding
                position_ids=batched_slice_pos_ids, 
                use_cache=True, 
                cache_position=slice_prefill_cache_pos_for_model_fwd 
            )
        batched_kv_cache_s_j: Cache = batched_outputs_S_j.past_key_values # type: ignore
        
        last_token_indices = original_slice_lengths - 1
        actual_last_token_logits = batched_outputs_S_j.logits[torch.arange(num_actual_slices, device=self.device), last_token_indices, :]
        batched_lookahead_tok_ids = torch.argmax(actual_last_token_logits, dim=-1, keepdim=True) 

        last_orig_pos_in_slices = torch.tensor(
            [all_slices_position_ids_unpadded[i][original_slice_lengths[i].item()-1].item() for i in range(num_actual_slices)],
            device=self.device
        )
        batched_lookahead_pos_ids = (last_orig_pos_in_slices + 1).unsqueeze(-1) 
        
        self.is_capturing_queries_for_slice_importance = True
        self.captured_lookahead_queries_buffer = [torch.empty(0) for _ in range(self.base_model.config.num_hidden_layers)] 
        with torch.no_grad():
            current_max_len_in_cache = batched_kv_cache_s_j.get_seq_length(layer_idx=0) 
            lookahead_cache_pos_for_model_fwd = torch.tensor([current_max_len_in_cache], device=self.device, dtype=torch.long)

            # For lookahead, attention_mask should be None to let the model handle causality w.r.t. cache
            _ = self.base_model(
                input_ids=batched_lookahead_tok_ids, 
                attention_mask=None, # Crucial for single token decoding with KV cache
                position_ids=batched_lookahead_pos_ids, 
                past_key_values=batched_kv_cache_s_j, 
                use_cache=True, 
                cache_position=lookahead_cache_pos_for_model_fwd 
            )
        self.is_capturing_queries_for_slice_importance = False

        global_importance_raw_sum = torch.zeros(1, prompt_length, device=self.device, dtype=self.dtype) 
        global_token_counts_for_aggregation = torch.zeros(1, prompt_length, device=self.device, dtype=torch.int)

        example_attn_layer = self.base_model.model.layers[0].self_attn # type: ignore
        head_dim = example_attn_layer.head_dim
        num_kv_groups = example_attn_layer.num_key_value_groups
        
        batched_slice_importances = torch.zeros(num_actual_slices, max_slice_len, device=self.device, dtype=self.dtype)

        for layer_idx in range(self.base_model.config.num_hidden_layers):
            queries_for_layer = self.captured_lookahead_queries_buffer[layer_idx] 
            keys_for_layer_padded = batched_kv_cache_s_j.key_cache[layer_idx]    
            keys_for_layer_orig_slice_padded = keys_for_layer_padded[:, :, :max_slice_len, :]
            keys_repeated = hf_repeat_kv(keys_for_layer_orig_slice_padded, num_kv_groups) 
            layer_attn_logits = torch.matmul(queries_for_layer, keys_repeated.transpose(-1, -2)) / math.sqrt(head_dim)
            layer_score_contribution_batched = layer_attn_logits.sum(dim=1).squeeze(dim=1) 
            batched_slice_importances += layer_score_contribution_batched

        for i in range(num_actual_slices):
            s_len = original_slice_lengths[i].item()
            importances_for_this_slice = batched_slice_importances[i, :s_len] 
            original_indices_for_this_slice = all_slices_position_ids_unpadded[i] 
            global_importance_raw_sum.scatter_add_(1, original_indices_for_this_slice.unsqueeze(0), importances_for_this_slice.unsqueeze(0))
            global_token_counts_for_aggregation.scatter_add_(1, original_indices_for_this_slice.unsqueeze(0), torch.ones_like(importances_for_this_slice, dtype=torch.int).unsqueeze(0))
        
        global_importance_avg = torch.where(global_token_counts_for_aggregation > 0,
                                            global_importance_raw_sum / global_token_counts_for_aggregation,
                                            torch.zeros_like(global_importance_raw_sum))
        
        if self.device.type == 'cuda': torch.cuda.synchronize()
        timing_info["importance_calculation"] = time.perf_counter() - importance_calc_start_time

        if self.device.type == 'cuda': torch.cuda.synchronize()
        selection_start_time = time.perf_counter()
        num_tokens_to_keep_from_prompt_calc = max(1, math.ceil(prompt_length * prompt_keep_percentage))
        forced_indices_set = set(range(self.actual_n_start_for_run))
        for i in range(prompt_length - self.actual_n_end_for_run, prompt_length): forced_indices_set.add(i)
        num_distinct_forced_tokens = len(forced_indices_set)
        num_top_k_to_select_from_scores = max(0, num_tokens_to_keep_from_prompt_calc - num_distinct_forced_tokens)
        num_top_k_to_select_from_scores = min(num_top_k_to_select_from_scores, prompt_length - num_distinct_forced_tokens if prompt_length > num_distinct_forced_tokens else 0, prompt_length)

        top_k_indices_from_scores_list: List[int] = []
        if num_top_k_to_select_from_scores > 0 and global_importance_avg.sum().item() != 0 :
            mask_for_scoring = torch.ones_like(global_importance_avg[0], dtype=torch.bool)
            for idx_val in forced_indices_set: 
                if idx_val < prompt_length : mask_for_scoring[idx_val] = False 
            masked_scores = global_importance_avg[0].clone(); masked_scores[~mask_for_scoring] = -float('inf') 
            num_available_for_scoring = mask_for_scoring.sum().item()
            actual_k_for_topk = min(num_top_k_to_select_from_scores, num_available_for_scoring)
            if actual_k_for_topk > 0:
                _, top_k_indices_tensor = torch.topk(masked_scores, k=actual_k_for_topk)
                top_k_indices_from_scores_list = top_k_indices_tensor.tolist()
        
        final_selected_indices_set = forced_indices_set.copy(); final_selected_indices_set.update(top_k_indices_from_scores_list)
        if not final_selected_indices_set and prompt_length > 0:
            num_fallback_tokens = min(prompt_length, max(1, int(num_tokens_to_keep_from_prompt_calc)))
            final_selected_indices_set.update(range(num_fallback_tokens))

        sorted_top_k_indices_orig = torch.sort(torch.tensor(list(final_selected_indices_set), device=self.device, dtype=torch.long))[0]
        selected_tokens_input_ids = original_input_ids[:, sorted_top_k_indices_orig]
        selected_tokens_position_ids = sorted_top_k_indices_orig.unsqueeze(0) 
        if self.device.type == 'cuda': torch.cuda.synchronize()
        timing_info["token_selection"] = time.perf_counter() - selection_start_time
        
        assert selected_tokens_input_ids.shape[1] > 0 or prompt_length == 0, "Token selection resulted in zero tokens for a non-empty prompt."

        if self.device.type == 'cuda': torch.cuda.synchronize()
        final_prefill_start_time = time.perf_counter()
        base_model_cache_after_prefill: Optional[Cache] = None
        base_model_next_token_ids: Optional[torch.Tensor] = None

        if not self.share_kv_cache: 
            with torch.no_grad():
                final_prefill_cache_pos_for_model_fwd = torch.arange(selected_tokens_input_ids.shape[1], device=self.device)
                # For final prefill, if it's a "fresh" prefill, a 2D attention_mask for padding might be needed
                # if selected_tokens_input_ids could have padding. Here, it's derived from original_input_ids, so no padding.
                # Thus, attention_mask=None is appropriate to rely on causality.
                final_prefill_output = self.base_model(input_ids=selected_tokens_input_ids, 
                                                       attention_mask=None, # Relies on causality for prefill
                                                       position_ids=selected_tokens_position_ids, 
                                                       use_cache=True,
                                                       cache_position=final_prefill_cache_pos_for_model_fwd) 
            base_model_next_token_ids = torch.argmax(final_prefill_output.logits[:, -1:, :], dim=-1)
            base_model_cache_after_prefill = final_prefill_output.past_key_values
        else:
            raise NotImplementedError("share_kv_cache=True is not yet refactored for batched importance calculation.")

        if self.device.type == 'cuda': torch.cuda.synchronize()
        timing_info["final_prefill_ttft"] = time.perf_counter() - final_prefill_start_time
        
        if self.device.type == 'cuda': torch.cuda.synchronize()
        decode_start_time = time.perf_counter()
        generated_token_ids_list: List[int] = []
        final_generated_text = ""

        if base_model_next_token_ids is not None and base_model_cache_after_prefill is not None:
            current_decode_token_ids = base_model_next_token_ids
            current_decode_cache: Cache = base_model_cache_after_prefill
            last_input_pos = -1
            if selected_tokens_position_ids.shape[1] > 0: last_input_pos = selected_tokens_position_ids[0, -1].item()
            else: last_input_pos = prompt_length -1 
            
            current_real_position = last_input_pos + 1 
            
            for i in range(max_generation_length):
                assert current_decode_token_ids is not None
                generated_token_id = current_decode_token_ids.item()
                generated_token_ids_list.append(generated_token_id)
                if generated_token_id in self.eos_token_ids: break
                if i == max_generation_length - 1: break 
                
                decode_pos_ids_for_rope = torch.tensor([[current_real_position]], device=self.device, dtype=torch.long)
                current_cache_len = current_decode_cache.get_seq_length(layer_idx=0)
                decode_cache_pos_for_model_fwd = torch.tensor([current_cache_len], device=self.device, dtype=torch.long)

                with torch.no_grad():
                    decode_output = self.base_model(current_decode_token_ids, 
                                                    attention_mask=None, # Crucial for single token decoding with KV cache
                                                    position_ids=decode_pos_ids_for_rope, 
                                                    past_key_values=current_decode_cache, 
                                                    use_cache=True,
                                                    cache_position=decode_cache_pos_for_model_fwd)
                current_decode_token_ids = torch.argmax(decode_output.logits[:, -1:, :], dim=-1)
                current_decode_cache = decode_output.past_key_values # type: ignore
                current_real_position += 1
            final_generated_text = self.tokenizer.decode(generated_token_ids_list, skip_special_tokens=True)
        
        if self.device.type == 'cuda': torch.cuda.synchronize()
        timing_info["decoding"] = time.perf_counter() - decode_start_time
        
        self._unpatch_base_model()
        if self.device.type == 'cuda': torch.cuda.synchronize()
        timing_info["total_time"] = time.perf_counter() - overall_start_time
        
        return final_generated_text, {k: v for k, v in timing_info.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sliced Context Aggregation Prefill (SCAP) Pipeline")
    parser.add_argument("--base_model_name", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--dataset_name", type=str, default="hotpotqa", help="Dataset from THUDM/LongBench, or None.")
    parser.add_argument("--n_slices", type=int, default=20, help="Number of slices to generate.")
    parser.add_argument("--k_percentage", type=float, default=0.1, help="Percentage of middle tokens to include in each slice.")
    parser.add_argument("--n_start_tokens", type=int, default=100, help="Number of fixed tokens from the start of the prompt in each slice.")
    parser.add_argument("--n_end_tokens", type=int, default=100, help="Number of fixed tokens from the end of the prompt in each slice.")
    parser.add_argument("--chunk_size", type=int, default=8, help="Size of chunks for selecting middle tokens.") 
    parser.add_argument("--prompt_keep_percentage", type=float, default=0.2, help="Percentage of original prompt tokens to keep for final prefill.")
    parser.add_argument("--max_generation_length", type=int, default=32)
    parser.add_argument("--share_kv_cache", action="store_true", default=False, help="Enable KV cache aggregation from slices for final prefill.")
    args = parser.parse_args()

    pipeline = SlicedContextAggregationPipeline(
        base_model_name=args.base_model_name,
        n_slices=args.n_slices,
        k_percentage=args.k_percentage,
        n_start_tokens=args.n_start_tokens,
        n_end_tokens=args.n_end_tokens,
        chunk_size=args.chunk_size, 
        share_kv_cache=args.share_kv_cache
    )

    prompt_to_run_str: str
    if args.dataset_name:
        dataset = load_dataset('THUDM/LongBench', args.dataset_name, split='test') # type: ignore
        sample = dataset[0] # type: ignore
        context = sample.get('context', '')
        input_val = sample.get('input', '') 
        if not context and not input_val: 
                context_fields = [str(v) for k,v in sample.items() if isinstance(v, str)]
                context = " ".join(context_fields[:len(context_fields)//2])
                input_val = " ".join(context_fields[len(context_fields)//2:]) if len(context_fields) > 1 else "Summarize."
        messages = [{"role": "user", "content": f"Context: {context}\nQuestion: {input_val}\nAnswer:"}]
        prompt_to_run_str = pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) # type: ignore
    else:
        messages = [{"role": "user", "content": "Explain the theory of relativity in simple terms."}]
        prompt_to_run_str = pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) # type: ignore
    
    print(f"\n--- Running SCAP with prompt (first 100 chars): '{prompt_to_run_str[:100]}...' ---")
    
    generated_text, run_timing_info = pipeline.run(
        prompt_text=prompt_to_run_str,
        prompt_keep_percentage=args.prompt_keep_percentage,
        max_generation_length=args.max_generation_length
    )
    
    num_generated_tokens = len(pipeline.tokenizer.encode(generated_text))
    print(f"\n--- Generated Output ({num_generated_tokens} tokens) ---")
    print(f"{generated_text}")

    print(f"\n--- Run Timing Information (seconds) ---")
    for stage, duration in run_timing_info.items():
        print(f"  {stage}: {duration:.4f}")
