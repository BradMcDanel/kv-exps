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

# --- QTIP Imports ---
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'qtip'))
try:
    from lib.utils.unsafe_import import model_from_hf_path as qtip_model_from_hf_path
    from model.llama import LlamaAttention as QTIP_LlamaAttention
    QTIP_AVAILABLE = True
except ImportError:
    QTIP_AVAILABLE = False
    qtip_model_from_hf_path = None
    QTIP_LlamaAttention = None


def _patched_attention_forward(
    self_attn: Union[LlamaAttention, Any],
    pipeline_instance: 'EchoCachePipeline',
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
    assert position_embeddings is not None, "Patched attention requires 'position_embeddings' from the parent model."
    cos, sin = position_embeddings

    batch_size, query_length, _ = hidden_states.size()

    num_heads = getattr(self_attn, 'num_heads', self_attn.config.num_attention_heads)
    head_dim = getattr(self_attn, 'head_dim', self_attn.config.hidden_size // num_heads)
    num_key_value_heads = getattr(self_attn, 'num_key_value_heads', self_attn.config.num_key_value_heads)
    hidden_size = getattr(self_attn, 'hidden_size', self_attn.config.hidden_size)
    num_key_value_groups = getattr(self_attn, 'num_key_value_groups', num_heads // num_key_value_heads)

    query_projection = self_attn.q_proj(hidden_states)
    key_projection = self_attn.k_proj(hidden_states)
    value_projection = self_attn.v_proj(hidden_states)

    query_states = query_projection.view(batch_size, query_length, num_heads, head_dim).transpose(1, 2)
    key_states_before_rope = key_projection.view(batch_size, query_length, num_key_value_heads, head_dim).transpose(1, 2)
    value_states_for_cache = value_projection.view(batch_size, query_length, num_key_value_heads, head_dim).transpose(1, 2)

    query_states_rotated, key_states_rotated = hf_apply_rotary_pos_emb(query_states, key_states_before_rope, cos, sin, position_ids)

    if pipeline_instance.is_prefilling and query_length > 1:
        last_q_vector = query_states_rotated[:, :, -1:, :].detach().clone()
        pipeline_instance.captured_qs[self_attn.layer_idx].append(last_q_vector)
    elif not pipeline_instance.is_prefilling and query_length == 1:
        pipeline_instance.captured_qs[self_attn.layer_idx].append(query_states_rotated.detach().clone())

    key_states_for_sdpa, value_states_for_sdpa = past_key_value.update(
        key_states_rotated, value_states_for_cache, self_attn.layer_idx, {"cache_position": cache_position}
    )

    key_states_for_sdpa = hf_repeat_kv(key_states_for_sdpa, num_key_value_groups)
    value_states_for_sdpa = hf_repeat_kv(value_states_for_sdpa, num_key_value_groups)

    is_sdpa_causal = (query_length > 1) and (attention_mask is None)
    attention_output = F.scaled_dot_product_attention(
        query_states_rotated, key_states_for_sdpa, value_states_for_sdpa, attn_mask=attention_mask, dropout_p=0.0, is_causal=is_sdpa_causal
    )

    attention_output = attention_output.transpose(1, 2).contiguous().reshape(batch_size, query_length, hidden_size)
    attention_output = self_attn.o_proj(attention_output)

    return attention_output, None, past_key_value


class EchoCachePipeline:
    def __init__(
        self,
        base_model_name: str,
        speculator_model_name: str,
        tokenizer: AutoTokenizer,
        max_capacity_prompt: int = 512,
        tsp_len: int = 2048,
        cache_granularity: str = "head",
        pool_kernel_size: Optional[int] = 13,
        pool_type: str = 'avgpool',
        use_chunk_selection: bool = True,
        chunk_size: int = 32,
    ):
        self.base_model_name = base_model_name
        self.speculator_model_name = speculator_model_name
        self.max_capacity_prompt_arg = max_capacity_prompt
        self.tsp_len = tsp_len
        self.cache_granularity = cache_granularity.lower()
        if self.cache_granularity not in ["global", "layer", "head"]:
            raise ValueError(f"Invalid cache_granularity: {self.cache_granularity}. Must be 'global', 'layer', or 'head'.")

        self.pool_kernel_size = pool_kernel_size
        self.pool_type = pool_type.lower()
        self.use_chunk_selection = use_chunk_selection
        self.chunk_size = chunk_size

        self._configure_capacities()
        self._validate_config()

        self.tokenizer = tokenizer
        self.speculator_model = self._load_model(self.speculator_model_name)
        self.base_model = self._load_model(self.base_model_name)
        self.spec_config: AutoConfig = self.speculator_model.config
        self.base_config: AutoConfig = self.base_model.config
        self.device = self.speculator_model.device
        self.dtype = self.speculator_model.dtype
        self.eos_token_ids = self._extract_eos_token_ids()
        self._check_model_compatibility()
        self.captured_qs: List[List[torch.Tensor]] = []
        self.orig_spec_fwds: Dict[int, Any] = {}
        self.is_prefilling = False
        self.token_importance_scores: Optional[torch.Tensor] = None

    def _configure_capacities(self):
        if self.cache_granularity == "global":
            self.global_capacity_target = self.max_capacity_prompt_arg
            self.local_capacity_target_per_layer = self.max_capacity_prompt_arg
            self.local_capacity_target_per_head = self.max_capacity_prompt_arg
        elif self.cache_granularity == "layer":
            self.global_capacity_target = self.tsp_len
            self.local_capacity_target_per_layer = self.max_capacity_prompt_arg
        elif self.cache_granularity == "head":
            self.global_capacity_target = self.tsp_len
            self.local_capacity_target_per_head = self.max_capacity_prompt_arg

        if self.cache_granularity != "global" and self.tsp_len <= 0:
            raise ValueError("tsp_len must be positive for 'layer' or 'head' granularity.")

    def _validate_config(self):
        if self.pool_type not in ['avgpool', 'maxpool', 'none']:
            raise ValueError(f"pool_type must be 'avgpool', 'maxpool', or 'none', but got {self.pool_type}")
        if self.pool_kernel_size is not None:
            if self.pool_kernel_size <= 1:
                self.pool_kernel_size = None
            elif self.pool_kernel_size % 2 == 0:
                raise ValueError("pool_kernel_size must be an odd number.")
            if self.pool_type == 'none':
                raise ValueError("pool_kernel_size is specified, but pool_type is 'none'.")
        if self.pool_type != 'none' and self.pool_kernel_size is None:
            raise ValueError(f"pool_type is '{self.pool_type}', but pool_kernel_size is not specified.")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer.")

        if self.cache_granularity == "global":
            if self.global_capacity_target <= 0:
                raise ValueError("global_capacity_target must be positive for 'global' mode.")
        elif self.cache_granularity == "layer":
            if self.global_capacity_target <= 0:
                raise ValueError("tsp_len (global_capacity_target) must be positive for 'layer' mode.")
            if self.local_capacity_target_per_layer <= 0:
                raise ValueError("max_capacity_prompt (local_capacity_target_per_layer) must be positive for 'layer' mode.")
            if self.global_capacity_target < self.local_capacity_target_per_layer:
                print(f"Warning ('layer' mode): tsp_len ({self.global_capacity_target}) is less than per-layer target ({self.local_capacity_target_per_layer}). Effective local target will be smaller.")
        elif self.cache_granularity == "head":
            if self.global_capacity_target <= 0:
                raise ValueError("tsp_len (global_capacity_target) must be positive for 'head' mode.")
            if self.local_capacity_target_per_head <= 0:
                raise ValueError("max_capacity_prompt (local_capacity_target_per_head) must be positive for 'head' mode.")
            if self.global_capacity_target < self.local_capacity_target_per_head:
                print(f"Warning ('head' mode): tsp_len ({self.global_capacity_target}) is less than per-head target ({self.local_capacity_target_per_head}). Effective local target will be smaller.")

    def _extract_eos_token_ids(self) -> List[int]:
        config_eos = self.base_config.eos_token_id
        if isinstance(config_eos, int):
            return [config_eos]
        if isinstance(config_eos, list):
            return list(config_eos)
        if self.tokenizer.eos_token_id:
            return [self.tokenizer.eos_token_id]
        raise ValueError("eos_token_id not defined in config or tokenizer.")

    def _is_qtip_model(self, model_name: str) -> bool:
        return QTIP_AVAILABLE and ("relaxml" in model_name.lower() or "qtip" in model_name.lower())

    def _load_model(self, model_name: str) -> AutoModelForCausalLM:
        if self._is_qtip_model(model_name):
            if not QTIP_AVAILABLE:
                raise ImportError(f"QTIP model {model_name} requested but modules not available.")
            print(f"Loading QTIP model: {model_name}")
            model, _ = qtip_model_from_hf_path(model_name, max_mem_ratio=0.7, attn_implementation="sdpa")
        else:
            print(f"Loading standard HF model: {model_name}")
            model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float16, trust_remote_code=True, device_map="auto", attn_implementation="sdpa"
            )
        return model.eval()

    def _check_model_compatibility(self):
        spec_config = self.speculator_model.config
        base_config = self.base_model.config
        error_msg = "Speculator and base models are not compatible for KV cache sharing."

        def check_attr(attr, s, b):
            if s != b:
                raise ValueError(f"{error_msg} Mismatch in {attr}: {s} (spec) vs {b} (base).")

        check_attr('num_hidden_layers', spec_config.num_hidden_layers, base_config.num_hidden_layers)
        check_attr('hidden_size', spec_config.hidden_size, base_config.hidden_size)
        check_attr('num_attention_heads', spec_config.num_attention_heads, base_config.num_attention_heads)
        check_attr('num_key_value_heads', spec_config.num_key_value_heads, base_config.num_key_value_heads)
        print("Models are architecturally compatible for KV cache sharing.")

    def _patch_speculator(self) -> int:
        if not hasattr(self.speculator_model, 'model') or not hasattr(self.speculator_model.model, 'layers'):
            return 0
        num_layers = self.speculator_model.config.num_hidden_layers
        self.captured_qs = [[] for _ in range(num_layers)]
        num_patched_layers = 0
        for i, layer in enumerate(self.speculator_model.model.layers):
            if hasattr(layer, 'self_attn'):
                attn_module = layer.self_attn
                if isinstance(attn_module, LlamaAttention) or \
                   (QTIP_AVAILABLE and isinstance(attn_module, QTIP_LlamaAttention)):
                    if i not in self.orig_spec_fwds:
                        self.orig_spec_fwds[i] = attn_module.forward
                    attn_module.forward = types.MethodType(partial(_patched_attention_forward, pipeline_instance=self), attn_module)
                    num_patched_layers += 1
        return num_patched_layers

    def _token_importance_from_attn_scores(self, attention_scores: torch.Tensor):
        if attention_scores.numel() == 0:
            raise RuntimeError("Cannot calculate importance from empty attention scores.")
        bs, num_layers, num_heads, num_steps, key_len = attention_scores.shape
        if bs != 1:
            raise NotImplementedError("Batch size > 1 is not supported.")

        all_layers_tensor = F.softmax(attention_scores.squeeze(0), dim=-1, dtype=torch.float32).to(attention_scores.dtype)
        flattened_tensor = all_layers_tensor.reshape(num_layers * num_heads, num_steps, key_len)

        if self.pool_kernel_size and self.pool_type != 'none':
            padding = (self.pool_kernel_size - 1) // 2
            pool_fn = F.avg_pool1d if self.pool_type == 'avgpool' else F.max_pool1d
            pooled_tensor = pool_fn(flattened_tensor.reshape(-1, 1, key_len), kernel_size=self.pool_kernel_size, stride=1, padding=padding)
            pooled_tensor = pooled_tensor.reshape(num_layers * num_heads, num_steps, key_len)
        else:
            pooled_tensor = flattened_tensor
        self.token_importance_scores = pooled_tensor.max(dim=0).values.mean(dim=0).unsqueeze(0)

    def _compute_raw_qk_scores(self, speculator_prefill_cache_as_tuple) -> torch.Tensor:
        if not self.captured_qs or not any(self.captured_qs):
            raise RuntimeError("Speculator Q-vectors were not captured.")

        num_spec_layers = self.spec_config.num_hidden_layers
        spec_num_q_heads = self.spec_config.num_attention_heads
        spec_num_kv_heads = self.spec_config.num_key_value_heads
        spec_num_kv_groups = spec_num_q_heads // spec_num_kv_heads
        all_layer_scores = []

        for layer_idx in range(num_spec_layers):
            if not self.captured_qs[layer_idx] or speculator_prefill_cache_as_tuple[layer_idx][0].numel() == 0:
                continue
            key_prompt_layer_rep_spec = hf_repeat_kv(speculator_prefill_cache_as_tuple[layer_idx][0].detach(), spec_num_kv_groups)
            all_q_for_layer = torch.cat(self.captured_qs[layer_idx], dim=2)
            attn_logits = torch.matmul(all_q_for_layer, key_prompt_layer_rep_spec.transpose(-1, -2)) / math.sqrt(all_q_for_layer.shape[-1])
            all_layer_scores.append(attn_logits)
        return torch.stack(all_layer_scores, dim=1)

    def _select_tokens_by_chunk(self, scores: torch.Tensor, num_to_keep: int, original_seq_len: int) -> torch.Tensor:
        if original_seq_len <= num_to_keep:
            return torch.arange(original_seq_len, device=scores.device)

        num_chunks = math.ceil(original_seq_len / self.chunk_size)
        padding_len = num_chunks * self.chunk_size - original_seq_len
        padded_scores = F.pad(scores, (0, padding_len), value=float('-inf')) if padding_len > 0 else scores
        avg_chunk_scores = padded_scores.view(num_chunks, self.chunk_size).mean(dim=1)

        num_chunks_to_keep = min(max(1, math.ceil(num_chunks * (num_to_keep / original_seq_len))), num_chunks)
        if num_to_keep == 0:
            num_chunks_to_keep = 0
        if num_chunks_to_keep == 0:
            return torch.empty(0, device=scores.device, dtype=torch.long)

        _, top_chunk_indices = torch.topk(avg_chunk_scores, k=num_chunks_to_keep)
        selected_indices = torch.cat([torch.arange(idx * self.chunk_size, (idx + 1) * self.chunk_size, device=scores.device) for idx in top_chunk_indices])
        final_indices = selected_indices[selected_indices < original_seq_len]

        if len(final_indices) > num_to_keep:
            if scores[final_indices].numel() == 0:
                return torch.sort(final_indices)[0]
            _, top_k_in_chunks_indices = torch.topk(scores[final_indices], k=min(num_to_keep, len(final_indices)))
            final_indices = final_indices[top_k_in_chunks_indices]
        return torch.sort(final_indices)[0]

    def _calculate_indices_for_global_selection(self, original_seq_len: int) -> torch.Tensor:
        num_to_keep_globally = min(self.global_capacity_target, original_seq_len)
        if num_to_keep_globally >= original_seq_len:
            return torch.arange(original_seq_len, device=self.device, dtype=torch.long)
        if self.token_importance_scores is None:
            raise RuntimeError("Global token importance scores not computed.")

        scores_for_selection = self.token_importance_scores[0].clone()
        if self.use_chunk_selection:
            return self._select_tokens_by_chunk(scores_for_selection, num_to_keep_globally, original_seq_len)
        else:
            _, indices = torch.topk(scores_for_selection, k=num_to_keep_globally, dim=-1)
            return torch.sort(indices)[0]

    def _refine_kv_cache_locally_per_layer(
        self,
        k_globally_pruned: torch.Tensor,
        v_globally_pruned: torch.Tensor,
        layer_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        current_globally_pruned_len = k_globally_pruned.shape[2]
        num_to_keep_locally = min(self.local_capacity_target_per_layer, current_globally_pruned_len)

        if num_to_keep_locally >= current_globally_pruned_len:
            return k_globally_pruned, v_globally_pruned

        if not self.captured_qs[layer_idx]:
            print(f"Warning: No Q-vectors for layer {layer_idx} during local per-layer refinement. Selecting first N.")
            return k_globally_pruned[:, :, :num_to_keep_locally, :], v_globally_pruned[:, :, :num_to_keep_locally, :]

        layer_q_vectors = torch.cat(self.captured_qs[layer_idx], dim=2)
        spec_num_q_heads = self.spec_config.num_attention_heads
        spec_num_kv_heads = self.spec_config.num_key_value_heads
        spec_num_kv_groups = spec_num_q_heads // spec_num_kv_heads
        k_globally_pruned_repeated = hf_repeat_kv(k_globally_pruned, spec_num_kv_groups)
        attn_logits_layer = torch.matmul(layer_q_vectors, k_globally_pruned_repeated.transpose(-1, -2)) / math.sqrt(layer_q_vectors.shape[-1])

        if attn_logits_layer.shape[0] != 1:
            raise NotImplementedError("Per-layer local pruning for batch size > 1 not supported.")

        scores_squeezed = attn_logits_layer.squeeze(0)
        scores_softmaxed = F.softmax(scores_squeezed, dim=-1, dtype=torch.float32).to(scores_squeezed.dtype)
        num_lookahead_steps = layer_q_vectors.shape[2]
        flattened_scores = scores_softmaxed.reshape(spec_num_q_heads * num_lookahead_steps, current_globally_pruned_len)

        if self.pool_kernel_size and self.pool_type != 'none':
            padding = (self.pool_kernel_size - 1) // 2
            pool_fn = F.avg_pool1d if self.pool_type == 'avgpool' else F.max_pool1d
            pooled_scores_for_local = pool_fn(flattened_scores.unsqueeze(1), kernel_size=self.pool_kernel_size, stride=1, padding=padding).squeeze(1)
        else:
            pooled_scores_for_local = flattened_scores

        local_importance_scores = pooled_scores_for_local.max(dim=0).values
        if self.use_chunk_selection:
            local_top_indices = self._select_tokens_by_chunk(local_importance_scores, num_to_keep_locally, current_globally_pruned_len)
        else:
            _, local_top_indices = torch.topk(local_importance_scores, k=num_to_keep_locally, dim=-1)
            local_top_indices = torch.sort(local_top_indices)[0]

        k_final_layer = torch.index_select(k_globally_pruned, 2, local_top_indices)
        v_final_layer = torch.index_select(v_globally_pruned, 2, local_top_indices)
        return k_final_layer, v_final_layer

    def _refine_kv_cache_locally_per_head(
        self,
        k_globally_pruned_layer: torch.Tensor,
        v_globally_pruned_layer: torch.Tensor,
        layer_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_spec_kv_heads, current_globally_pruned_len, head_dim = k_globally_pruned_layer.shape
        num_to_keep_locally_per_head = min(self.local_capacity_target_per_head, current_globally_pruned_len)

        if num_to_keep_locally_per_head >= current_globally_pruned_len:
            return k_globally_pruned_layer, v_globally_pruned_layer

        if not self.captured_qs[layer_idx]:
            print(f"Warning: No Q-vectors for layer {layer_idx} during local per-head refinement. Selecting first N.")
            return k_globally_pruned_layer[:, :, :num_to_keep_locally_per_head, :], \
                   v_globally_pruned_layer[:, :, :num_to_keep_locally_per_head, :]

        layer_q_vectors_cat = torch.cat(self.captured_qs[layer_idx], dim=2)
        num_spec_q_heads = self.spec_config.num_attention_heads
        spec_num_kv_groups = num_spec_q_heads // num_spec_kv_heads
        final_k_for_layer_heads, final_v_for_layer_heads = [], []

        for kv_head_idx in range(num_spec_kv_heads):
            k_current_kv_head = k_globally_pruned_layer[:, kv_head_idx:kv_head_idx + 1, :, :]
            q_group_start_idx = kv_head_idx * spec_num_kv_groups
            q_group_end_idx = (kv_head_idx + 1) * spec_num_kv_groups
            q_vectors_for_kv_group = layer_q_vectors_cat[:, q_group_start_idx:q_group_end_idx, :, :]
            k_current_kv_head_repeated = k_current_kv_head.repeat_interleave(spec_num_kv_groups, dim=1)
            attn_logits_group = torch.matmul(q_vectors_for_kv_group, k_current_kv_head_repeated.transpose(-1, -2)) / math.sqrt(head_dim)

            if batch_size != 1:
                raise NotImplementedError("Per-head local pruning for batch size > 1 not supported.")

            scores_for_kv_head_squeezed = attn_logits_group.squeeze(0)
            scores_summed_over_q_group = scores_for_kv_head_squeezed.sum(dim=0)
            scores_softmaxed = F.softmax(scores_summed_over_q_group, dim=-1, dtype=torch.float32).to(scores_summed_over_q_group.dtype)

            if self.pool_kernel_size and self.pool_type != 'none':
                padding = (self.pool_kernel_size - 1) // 2
                pool_fn = F.avg_pool1d if self.pool_type == 'avgpool' else F.max_pool1d
                pooled_scores = pool_fn(scores_softmaxed.unsqueeze(1), kernel_size=self.pool_kernel_size, stride=1, padding=padding).squeeze(1)
            else:
                pooled_scores = scores_softmaxed
            local_importance_scores_for_kv_head = pooled_scores.max(dim=0).values

            if self.use_chunk_selection:
                local_top_indices_for_kv_head = self._select_tokens_by_chunk(
                    local_importance_scores_for_kv_head, num_to_keep_locally_per_head, current_globally_pruned_len
                )
            else:
                _, local_top_indices_for_kv_head = torch.topk(
                    local_importance_scores_for_kv_head, k=num_to_keep_locally_per_head, dim=-1
                )
                local_top_indices_for_kv_head = torch.sort(local_top_indices_for_kv_head)[0]

            k_selected_for_head = torch.index_select(k_current_kv_head, 2, local_top_indices_for_kv_head)
            v_current_kv_head = v_globally_pruned_layer[:, kv_head_idx:kv_head_idx + 1, :, :]
            v_selected_for_head = torch.index_select(v_current_kv_head, 2, local_top_indices_for_kv_head)
            final_k_for_layer_heads.append(k_selected_for_head)
            final_v_for_layer_heads.append(v_selected_for_head)

        final_k_for_layer = torch.cat(final_k_for_layer_heads, dim=1)
        final_v_for_layer = torch.cat(final_v_for_layer_heads, dim=1)
        return final_k_for_layer, final_v_for_layer

    def _slice_kv_cache(
        self,
        spec_cache_tuple: Tuple[Tuple[torch.Tensor, ...], ...],
        original_prompt_len: int
    ) -> Tuple[DynamicCache, int, int]:
        final_sliced_cache = DynamicCache()
        if not spec_cache_tuple or spec_cache_tuple[0][0].numel() == 0:
            return final_sliced_cache, 0, 0

        global_indices_to_keep = self._calculate_indices_for_global_selection(original_prompt_len)
        num_globally_kept = global_indices_to_keep.numel()

        if num_globally_kept == 0 and original_prompt_len > 0:
            print("Warning: Global pruning resulted in 0 tokens.")
            return final_sliced_cache, 0, 0

        avg_final_kept_metric = 0

        if self.cache_granularity == "global":
            for layer_idx, (k_l_spec_full, v_l_spec_full) in enumerate(spec_cache_tuple):
                if k_l_spec_full.numel() > 0:
                    k_final = torch.index_select(k_l_spec_full, 2, global_indices_to_keep)
                    v_final = torch.index_select(v_l_spec_full, 2, global_indices_to_keep)
                else:
                    k_final, v_final = k_l_spec_full, v_l_spec_full
                final_sliced_cache.update(
                    key_states=k_final.to(dtype=self.base_model.dtype),
                    value_states=v_final.to(dtype=self.base_model.dtype),
                    layer_idx=layer_idx
                )
            avg_final_kept_metric = num_globally_kept
        else:
            total_locally_kept_tokens_metric_numerator = 0
            metric_denominator = 0

            for layer_idx, (k_l_spec_full, v_l_spec_full) in enumerate(spec_cache_tuple):
                k_globally_pruned_layer, v_globally_pruned_layer = k_l_spec_full, v_l_spec_full
                if k_l_spec_full.numel() > 0 and num_globally_kept < original_prompt_len:
                    k_globally_pruned_layer = torch.index_select(k_l_spec_full, 2, global_indices_to_keep)
                    v_globally_pruned_layer = torch.index_select(v_l_spec_full, 2, global_indices_to_keep)

                if k_globally_pruned_layer.numel() == 0:
                    final_sliced_cache.update(key_states=k_globally_pruned_layer, value_states=v_globally_pruned_layer, layer_idx=layer_idx)
                    continue

                k_final_for_layer, v_final_for_layer = k_globally_pruned_layer, v_globally_pruned_layer

                if self.cache_granularity == "layer":
                    k_final_for_layer, v_final_for_layer = self._refine_kv_cache_locally_per_layer(
                        k_globally_pruned_layer, v_globally_pruned_layer, layer_idx
                    )
                    total_locally_kept_tokens_metric_numerator += k_final_for_layer.shape[2]
                    metric_denominator += 1
                elif self.cache_granularity == "head":
                    k_final_for_layer, v_final_for_layer = self._refine_kv_cache_locally_per_head(
                        k_globally_pruned_layer, v_globally_pruned_layer, layer_idx
                    )
                    total_locally_kept_tokens_metric_numerator += k_final_for_layer.shape[2] * k_final_for_layer.shape[1]
                    metric_denominator += k_final_for_layer.shape[1]

                final_sliced_cache.update(
                    key_states=k_final_for_layer.to(dtype=self.base_model.dtype),
                    value_states=v_final_for_layer.to(dtype=self.base_model.dtype),
                    layer_idx=layer_idx
                )

            if metric_denominator > 0:
                avg_final_kept_metric = total_locally_kept_tokens_metric_numerator / metric_denominator
            else:
                avg_final_kept_metric = 0

        return final_sliced_cache, num_globally_kept, int(avg_final_kept_metric)

    def run(self, input_ids: torch.Tensor, look_ahead_k: int, max_generation_length: int) -> Tuple[str, Dict[str, Any]]:
        run_metadata: Dict[str, Any] = {
            "max_capacity_prompt_arg": self.max_capacity_prompt_arg,
            "tsp_len_global_target": self.tsp_len if self.cache_granularity != "global" else "N/A (Global Mode)",
            "cache_granularity": self.cache_granularity
        }
        if self.cache_granularity == "global":
            run_metadata["effective_global_target"] = self.global_capacity_target
        elif self.cache_granularity == "layer":
            run_metadata["effective_local_layer_target"] = self.local_capacity_target_per_layer
        elif self.cache_granularity == "head":
            run_metadata["effective_local_head_target"] = self.local_capacity_target_per_head

        overall_start_time = time.perf_counter()

        num_patched_layers = self._patch_speculator()
        if num_patched_layers == 0 and look_ahead_k > 0:
            raise RuntimeError("Speculator could not be patched to capture Q-vectors.")

        prompt_length = input_ids.shape[1]
        run_metadata["prompt_input_length"] = prompt_length
        if prompt_length == 0:
            run_metadata["total_time"] = time.perf_counter() - overall_start_time
            run_metadata["global_token_keep_rate"] = 100.0
            run_metadata["local_token_keep_rate_from_global"] = 100.0
            return "", run_metadata

        stage_start_time = time.perf_counter()
        self.is_prefilling = True
        [q_list.clear() for q_list in self.captured_qs]
        self.token_importance_scores = None

        with torch.no_grad():
            spec_out = self.speculator_model(input_ids=input_ids, use_cache=True, cache_position=torch.arange(prompt_length, device=self.device))
        speculator_prefill_cache = spec_out.past_key_values
        if speculator_prefill_cache is None:
            raise RuntimeError("Speculator prefill did not return a KV cache.")

        speculator_prefill_cache_as_tuple = speculator_prefill_cache.to_legacy_cache() if isinstance(speculator_prefill_cache, DynamicCache) else speculator_prefill_cache
        current_spec_cache_for_lookahead = DynamicCache.from_legacy_cache(speculator_prefill_cache_as_tuple)
        run_metadata["speculation_prefill"] = time.perf_counter() - stage_start_time

        stage_start_time = time.perf_counter()
        self.is_prefilling = False
        with torch.no_grad():
            current_spec_tokens = torch.argmax(spec_out.logits[:, -1, :], dim=-1, keepdim=True)
            for _i in range(look_ahead_k):
                cache_len = current_spec_cache_for_lookahead.get_seq_length(0)
                pos_ids_lookahead = torch.tensor([[cache_len]], device=self.device, dtype=torch.long)
                lookahead_out = self.speculator_model(
                    input_ids=current_spec_tokens,
                    position_ids=pos_ids_lookahead,
                    past_key_values=current_spec_cache_for_lookahead,
                    use_cache=True,
                    cache_position=pos_ids_lookahead[0]
                )
                current_spec_cache_for_lookahead = lookahead_out.past_key_values
                if isinstance(current_spec_cache_for_lookahead, tuple):
                    current_spec_cache_for_lookahead = DynamicCache.from_legacy_cache(current_spec_cache_for_lookahead)
                current_spec_tokens = torch.argmax(lookahead_out.logits[:, -1, :], dim=-1, keepdim=True)
                if current_spec_tokens.item() in self.eos_token_ids:
                    break
        run_metadata["speculation_decode_and_q_capture"] = time.perf_counter() - stage_start_time

        raw_qk_scores_global = self._compute_raw_qk_scores(speculator_prefill_cache_as_tuple)
        self._token_importance_from_attn_scores(raw_qk_scores_global)

        base_model_first_token_gen_start_time = time.perf_counter()

        injected_cache, num_globally_kept, avg_final_kept_metric_val = self._slice_kv_cache(
            speculator_prefill_cache_as_tuple, prompt_length
        )
        run_metadata["shared_cache_original_len"] = prompt_length
        run_metadata["shared_cache_globally_kept_token_count"] = num_globally_kept
        run_metadata["shared_cache_avg_final_kept_metric"] = avg_final_kept_metric_val

        num_kept_tokens_for_base_knockout = injected_cache.get_seq_length(layer_idx=0)
        knockout_tokens = input_ids[:, -1:]
        knockout_pos_ids = torch.tensor([[prompt_length - 1]], device=self.device, dtype=torch.long)
        knockout_cache_pos = torch.tensor([num_kept_tokens_for_base_knockout], device=self.device)

        with torch.no_grad():
            knockout_out = self.base_model(
                knockout_tokens,
                position_ids=knockout_pos_ids,
                past_key_values=injected_cache,
                use_cache=True,
                cache_position=knockout_cache_pos,
            )
        current_tokens = torch.argmax(knockout_out.logits[:, -1, :], dim=-1, keepdim=True)
        current_cache = knockout_out.past_key_values
        if isinstance(current_cache, tuple):
            current_cache = DynamicCache.from_legacy_cache(current_cache)

        run_metadata["global_token_keep_rate"] = (num_globally_kept / prompt_length * 100.0) if prompt_length > 0 else 100.0
        if self.cache_granularity != "global" and num_globally_kept > 0:
            run_metadata["local_token_keep_rate_from_global"] = (avg_final_kept_metric_val / num_globally_kept * 100.0)
        elif self.cache_granularity == "global":
            run_metadata["local_token_keep_rate_from_global"] = "N/A (Global Mode)"
        else:
            run_metadata["local_token_keep_rate_from_global"] = 100.0

        base_model_first_token_time = time.perf_counter() - base_model_first_token_gen_start_time
        run_metadata["base_knockout_pass"] = base_model_first_token_time
        run_metadata["base_ttft"] = run_metadata["speculation_prefill"] + run_metadata["speculation_decode_and_q_capture"] + base_model_first_token_time

        gen_token_ids_list = [current_tokens.item()]
        decode_total_time = 0.0

        if gen_token_ids_list[0] not in self.eos_token_ids:
            for i in range(max_generation_length - 1):
                decode_step_start_time = time.perf_counter()
                current_real_position = prompt_length + i
                pos_ids_decode = torch.tensor([[current_real_position]], device=self.device)
                cache_pos_decode = torch.tensor([current_cache.get_seq_length(0)], device=self.device)

                with torch.no_grad():
                    decode_out = self.base_model(
                        current_tokens,
                        position_ids=pos_ids_decode,
                        past_key_values=current_cache,
                        use_cache=True,
                        cache_position=cache_pos_decode
                    )
                decode_total_time += time.perf_counter() - decode_step_start_time

                next_tokens = torch.argmax(decode_out.logits[:, -1, :], dim=-1, keepdim=True)
                gen_token_ids_list.append(next_tokens.item())

                current_cache = decode_out.past_key_values
                if isinstance(current_cache, tuple):
                    current_cache = DynamicCache.from_legacy_cache(current_cache)

                current_tokens = next_tokens
                if gen_token_ids_list[-1] in self.eos_token_ids:
                    break

        final_gen_text = self.tokenizer.decode(gen_token_ids_list, skip_special_tokens=True)
        run_metadata["decode_time"] = decode_total_time
        run_metadata["generated_tokens"] = len(gen_token_ids_list)
        run_metadata["total_time"] = time.perf_counter() - overall_start_time
        if final_gen_text.startswith("assistant\n\n"):
            final_gen_text = final_gen_text[len("assistant\n\n"):]
        elif final_gen_text.startswith(" assistant\n"):
            final_gen_text = final_gen_text[len(" assistant\n"):]
        return final_gen_text, run_metadata


def main():
    parser = argparse.ArgumentParser(description="Hierarchical EchoCache Pipeline with Selectable Granularity")
    parser.add_argument("--base_model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--speculator_model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--dataset_name", type=str, default="qasper")
    parser.add_argument("--look_ahead_k", type=int, default=8)

    parser.add_argument("--tsp_len", type=int, default=2048, help="Target for GLOBAL pruning, or N/A if granularity is 'global'.")
    parser.add_argument("--max_capacity_prompt", type=int, default=512, help="Target for LOCAL pruning (per-layer or per-head based on granularity), or global target if granularity is 'global'.")
    parser.add_argument("--cache_granularity", type=str, default="head", choices=["global", "layer", "head"], help="Granularity of cache pruning: 'global', 'layer', or 'head'.")

    parser.add_argument("--max_generation_length", type=int, default=128)
    parser.add_argument("--kernel_size", type=int, default=13)
    parser.add_argument("--pooling", type=str, default="avgpool", choices=['avgpool', 'maxpool', 'none'])
    parser.add_argument("--use_chunk_selection", action='store_true', default=True)
    parser.add_argument("--no_chunk_selection", dest="use_chunk_selection", action="store_false")
    parser.add_argument("--chunk_size", type=int, default=32)

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    pipeline = EchoCachePipeline(
        base_model_name=args.base_model_name,
        speculator_model_name=args.speculator_model_name,
        tokenizer=tokenizer,
        max_capacity_prompt=args.max_capacity_prompt,
        tsp_len=args.tsp_len,
        cache_granularity=args.cache_granularity,
        pool_kernel_size=args.kernel_size if args.pooling != 'none' else None,
        pool_type=args.pooling,
        use_chunk_selection=args.use_chunk_selection,
        chunk_size=args.chunk_size,
    )

    prompt_str: str
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
        prompt_config_path = os.path.join(project_root, 'eval', 'longbench', 'config', 'dataset2prompt.json')
        if not os.path.exists(prompt_config_path):
            raise FileNotFoundError(f"Could not find LongBench prompt config at: {prompt_config_path}")
        import json
        with open(prompt_config_path, "r") as f:
            dataset2prompt = json.load(f)
        dataset = load_dataset('THUDM/LongBench', args.dataset_name, split='test', trust_remote_code=True)
        sample = dataset[0]
        prompt_format = dataset2prompt.get(args.dataset_name)
        if prompt_format is None:
            raise KeyError(f"Prompt format for '{args.dataset_name}' not found.")
        prompt_str = prompt_format.format(context=sample['context'], input=sample['input'])
    except Exception as e:
        print(f"Could not load dataset/prompt format. Using default. Error: {e}")
        prompt_str = "Explain the theory of relativity in simple terms."

    selection_strategy = f"Chunk-based (size={pipeline.chunk_size})" if pipeline.use_chunk_selection else "Top-K"
    print(f"\n--- Running Hierarchical EchoCache Pipeline ---")
    print(f"Cache Granularity: {args.cache_granularity}")
    print(f"Base Model: {args.base_model_name}")
    print(f"Speculator Model: {args.speculator_model_name}")
    print(f"Global Pruning Target (tsp_len / effective global): {pipeline.global_capacity_target if args.cache_granularity == 'global' else args.tsp_len}")
    if args.cache_granularity == "layer":
        print(f"Per-Layer Local Pruning Target: {pipeline.local_capacity_target_per_layer}")
    elif args.cache_granularity == "head":
        print(f"Per-KV-Head Local Pruning Target: {pipeline.local_capacity_target_per_head}")
    print(f"Lookahead K: {args.look_ahead_k}")
    print(f"Pooling: type='{pipeline.pool_type}', kernel_size={pipeline.pool_kernel_size}")
    print(f"Token Selection: {selection_strategy}")
    print(f"Dataset: {args.dataset_name}")
    print("-" * 34)

    messages = [{"role": "user", "content": prompt_str}]
    templated_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(templated_prompt, return_tensors="pt", truncation=True, max_length=4096).to(pipeline.device)

    generated_text, run_metadata = pipeline.run(
        input_ids=inputs.input_ids,
        look_ahead_k=args.look_ahead_k,
        max_generation_length=args.max_generation_length,
    )

    print(f"\n--- Generated Text ---")
    print(generated_text)
    print("-" * 22)
    print("\n--- Performance Metrics ---")
    print(f"Cache Granularity: {run_metadata.get('cache_granularity', 'N/A')}")
    print(f"Prompt Length (Tokens): {run_metadata.get('prompt_input_length', 'N/A')}")
    print(f"Globally Kept Tokens: {run_metadata.get('shared_cache_globally_kept_token_count', 'N/A')} "
          f"(target: {run_metadata.get('tsp_len_global_target', run_metadata.get('effective_global_target'))})")

    local_kept_metric_key = "shared_cache_avg_final_kept_metric"
    local_kept_desc = "Avg Final Kept Tokens"
    local_target_desc = "N/A"
    if run_metadata.get('cache_granularity') == "layer":
        local_kept_desc = "Avg Locally Kept Tokens (per layer, from global set)"
        local_target_desc = f"(target: {run_metadata.get('effective_local_layer_target')})"
    elif run_metadata.get('cache_granularity') == "head":
        local_kept_desc = "Avg Locally Kept Tokens (per KV head, from global set)"
        local_target_desc = f"(target: {run_metadata.get('effective_local_head_target')})"
    elif run_metadata.get('cache_granularity') == "global":
        local_kept_desc = "Final Kept Tokens (Global strategy)"
        local_target_desc = f"(target: {run_metadata.get('effective_global_target')})"

    print(f"{local_kept_desc}: {run_metadata.get(local_kept_metric_key, 'N/A')} {local_target_desc}")
    print(f"Global Token Keep Rate: {run_metadata.get('global_token_keep_rate', 0):.2f}%")

    local_keep_rate_key = "local_token_keep_rate_from_global"
    local_keep_rate_desc = "Local Token Keep Rate (from global set)"
    if run_metadata.get('cache_granularity') == "global":
        print(f"{local_keep_rate_desc}: N/A (Global Mode)")
    else:
        print(f"{local_keep_rate_desc}: {run_metadata.get(local_keep_rate_key, 0):.2f}%")

    print(f"Time to First Token (TTFT): {run_metadata.get('base_ttft', 0):.4f} seconds")
    print(f"  - Speculator Prefill: {run_metadata.get('speculation_prefill', 0):.4f} s")
    print(f"  - Speculator Decode/Q-Capture & Global Scoring: {run_metadata.get('speculation_decode_and_q_capture', 0):.4f} s")
    print(f"  - Base Model Knockout Pass (incl. hierarchical slicing): {run_metadata.get('base_knockout_pass', 0):.4f} s")
    print(f"Total Pipeline Time: {run_metadata.get('total_time', 0):.4f} seconds")
    print("-" * 27)


if __name__ == "__main__":
    main()
