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
import os
import sys


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
    """
    Patched forward method for LlamaAttention to capture Q-vectors from the speculator model.
    This is used to calculate attention scores for token selection.
    """
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


class SpeculativePrefillPipeline:
    POSITION_BUFFER = 20

    def __init__(
        self,
        base_model_name: str,
        speculator_model_name: str,
        max_capacity_prompt: int = 256
    ):
        self.base_model_name = base_model_name
        self.speculator_model_name = speculator_model_name
        self.max_capacity_prompt = max_capacity_prompt

        if self.speculator_model_name is None:
            raise ValueError("SpeculativePrefillPipeline requires a `speculator_model_name` to be provided.")

        self.tokenizer = self._load_tokenizer()
        self.speculator_model: AutoModelForCausalLM = self._load_model(self.speculator_model_name)
        self.device: torch.device = self.speculator_model.device
        self.dtype: torch.dtype = self.speculator_model.dtype

        self.base_model = self._load_model(self.base_model_name)
        self.base_config: AutoConfig = self.base_model.config

        self.eos_token_ids = self._extract_eos_token_ids(self.base_config.eos_token_id)

        self.captured_qs: List[List[torch.Tensor]] = []
        self.orig_spec_fwds: Dict[int, Any] = {}
        self.is_generating_lookaheads = False
        self.first_step_prompt_qk_scores: Optional[torch.Tensor] = None

    def _extract_eos_token_ids(self, eos_token_id: Union[int, List[int], None]) -> List[int]:
        if isinstance(eos_token_id, int):
            return [eos_token_id]
        elif isinstance(eos_token_id, list):
            return list(eos_token_id)
        else:
            return []

    def _load_model(self, model_name: str) -> AutoModelForCausalLM:
        load_kwargs = {
            "torch_dtype": torch.float16,
            "trust_remote_code": True,
            "device_map": "auto",
            "attn_implementation": "sdpa",
        }
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        return model.eval()

    def _get_chat_template(self, model_name: str) -> str:
        if "llama-3" in model_name.lower():
            return "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
        elif "llama-2" in model_name.lower():
            return "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 and system_message != false %}{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}"
        else:
            return "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

    def _load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_name, trust_remote_code=True)
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.chat_template is None:
            tokenizer.chat_template = self._get_chat_template(self.base_model_name)
        return tokenizer

    def _patch_speculator(self):
        if not hasattr(self.speculator_model, 'model') or not hasattr(self.speculator_model.model, 'layers'):
            return 0
        num_layers = self.speculator_model.config.num_hidden_layers
        self.captured_qs = [[] for _ in range(num_layers)]
        self.first_step_prompt_qk_scores = None
        num_patched_layers = 0
        for i, layer in enumerate(self.speculator_model.model.layers):
            if hasattr(layer, 'self_attn'):
                attn = layer.self_attn
                if isinstance(attn, LlamaAttention):
                    if i not in self.orig_spec_fwds:
                        self.orig_spec_fwds[i] = attn.forward
                    attn.forward = types.MethodType(partial(_hf_patched_attention_forward_method, pipeline_instance=self), attn)
                    num_patched_layers += 1
        return num_patched_layers

    def _compute_qk_scores(
        self,
        speculator_prefill_cache_as_tuple: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
        prompt_length: int,
        batch_size: int
    ):
        if not self.captured_qs or not any(self.captured_qs):
             raise RuntimeError("Cannot compute QK scores because no Q-vectors were captured from the speculator.")
        spec_config = self.speculator_model.config
        num_spec_layers = spec_config.num_hidden_layers
        spec_num_q_heads = getattr(spec_config, 'num_attention_heads', getattr(spec_config, 'num_heads', None))
        spec_num_kv_heads = getattr(spec_config, 'num_key_value_heads', spec_num_q_heads)
        if spec_num_q_heads is None or spec_num_kv_heads is None:
            raise RuntimeError("Could not determine head counts for speculator. Cannot compute QK scores.")
        spec_num_kv_groups = spec_num_q_heads // spec_num_kv_heads
        first_step_layer_scores_list = [
            torch.zeros(batch_size, prompt_length, device=self.device, dtype=self.dtype)
            for _ in range(num_spec_layers)
        ]
        for layer_idx in range(num_spec_layers):
            if not self.captured_qs[layer_idx] or speculator_prefill_cache_as_tuple[layer_idx][0].numel() == 0:
                continue
            key_prompt_layer_spec = speculator_prefill_cache_as_tuple[layer_idx][0].detach()
            key_prompt_layer_rep_spec = hf_repeat_kv(key_prompt_layer_spec, spec_num_kv_groups)
            head_dim_from_q = self.captured_qs[layer_idx][0].shape[-1]
            query_first_lookahead_layer = self.captured_qs[layer_idx][0]
            attn_logits_first_step = torch.matmul(
                query_first_lookahead_layer,
                key_prompt_layer_rep_spec.transpose(-1, -2)
            ) / math.sqrt(head_dim_from_q)
            attn_scores_first_step = attn_logits_first_step.squeeze(2)
            first_step_layer_scores_list[layer_idx] = attn_scores_first_step.sum(dim=1)
        valid_first_step_layer_scores = [
            fsls for fsls in first_step_layer_scores_list
            if fsls.numel() > 0 and fsls.shape[0] == batch_size and fsls.shape[-1] == prompt_length
        ]
        if not valid_first_step_layer_scores:
            raise RuntimeError(
                "Failed to compute QK scores. No valid layer scores were generated. "
                "This could be due to an issue with Q-vector capture or empty caches."
            )
        self.first_step_prompt_qk_scores = torch.sum(torch.stack(valid_first_step_layer_scores), dim=0)

    def _get_sorted_top_k_indices(
        self, scores: torch.Tensor, num_to_keep: int, current_seq_len: int, device: torch.device
    ) -> torch.Tensor:
        num_to_keep = min(num_to_keep, current_seq_len)
        if num_to_keep <= 0:
            return torch.empty(0, dtype=torch.long, device=device)
        if num_to_keep >= current_seq_len:
            return torch.arange(current_seq_len, device=device, dtype=torch.long)
        if scores.ndim > 1:
            scores = scores.squeeze()
        if scores.shape[0] != current_seq_len:
            raise ValueError(f"Score length {scores.shape[0]} does not match current_seq_len {current_seq_len}")
        _, top_k_indices = torch.topk(scores, k=num_to_keep)
        return torch.sort(top_k_indices)[0]

    def _get_indices_for_selective_prefill(
        self, original_seq_len: int, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        if not (self.first_step_prompt_qk_scores is not None and
                self.first_step_prompt_qk_scores.shape[0] == batch_size and
                self.first_step_prompt_qk_scores.shape[-1] == original_seq_len):
            raise RuntimeError("Cannot select indices without valid QK scores from speculator.")
        scores_for_selection = self.first_step_prompt_qk_scores[0].clone()
        num_to_keep = min(self.max_capacity_prompt, original_seq_len)
        final_indices = self._get_sorted_top_k_indices(
            scores_for_selection, num_to_keep, original_seq_len, device
        )
        return final_indices

    def run(
        self, prompt_text: str, look_ahead_k: int, max_generation_length: int
    ) -> Tuple[str, Dict[str, Any]]:
        run_metadata: Dict[str, Any] = {}
        overall_start_time = time.perf_counter()
        run_metadata["max_capacity_prompt"] = self.max_capacity_prompt

        num_patched_layers = self._patch_speculator()
        if num_patched_layers == 0 and look_ahead_k > 0:
            raise RuntimeError("Speculator model could not be patched, but look_ahead_k > 0. Cannot capture Q-vectors.")

        limit_due_to_base = self.base_config.max_position_embeddings - max_generation_length
        limit_due_to_speculator = self.speculator_model.config.max_position_embeddings - look_ahead_k
        max_prompt_len_calc = float(min(limit_due_to_base, limit_due_to_speculator) - self.POSITION_BUFFER)
        
        max_prompt_length = max(1, int(max_prompt_len_calc))
        inputs = self.tokenizer(
            prompt_text, return_tensors="pt", padding=False, truncation=True, max_length=max_prompt_length
        ).to(self.device)
        prompt_input_ids = inputs.input_ids
        prompt_length = inputs.input_ids.shape[1]
        batch_size = inputs.input_ids.shape[0]
        run_metadata["prompt_input_length"] = prompt_length
        if prompt_length == 0:
            run_metadata["total_time"] = time.perf_counter() - overall_start_time
            run_metadata["token_keep_rate"] = 100.0
            return "", run_metadata

        # --- Speculator Prefill and Scoring ---
        stage_start_time = time.perf_counter()
        self.is_generating_lookaheads = False
        spec_prompt_cache_pos = torch.arange(prompt_length, device=self.device)
        with torch.no_grad():
            spec_out = self.speculator_model(
                input_ids=prompt_input_ids, use_cache=True, cache_position=spec_prompt_cache_pos
            )
        
        legacy_cache_tuple = spec_out.past_key_values
        if legacy_cache_tuple is None:
            raise RuntimeError("Speculator prefill did not return a past_key_values cache.")
        speculator_prefill_cache_as_tuple = legacy_cache_tuple
        speculator_prefill_cache = DynamicCache.from_legacy_cache(legacy_cache_tuple)
        
        speculator_next_token_ids = torch.argmax(spec_out.logits[:, -1, :], dim=-1, keepdim=True)
        speculation_prefill_time = time.perf_counter() - stage_start_time
        run_metadata["speculation_prefill"] = speculation_prefill_time

        stage_start_time = time.perf_counter()
        self.is_generating_lookaheads = True
        [q_list.clear() for q_list in self.captured_qs]
        with torch.no_grad():
            self.speculator_model(
                input_ids=speculator_next_token_ids,
                position_ids=torch.tensor([[prompt_length]], device=self.device, dtype=torch.long),
                past_key_values=speculator_prefill_cache,
                use_cache=False,
                cache_position=torch.tensor([prompt_length], device=self.device, dtype=torch.long)
            )
        self.is_generating_lookaheads = False
        speculation_decode_time = time.perf_counter() - stage_start_time
        run_metadata["speculation_decode"] = speculation_decode_time
        
        self._compute_qk_scores(speculator_prefill_cache_as_tuple, prompt_length, batch_size)

        # --- Base Model Selective Prefill ---
        final_indices_for_selective_prefill = self._get_indices_for_selective_prefill(
            prompt_length, batch_size, self.device
        )
        
        if final_indices_for_selective_prefill.numel() == 0 and prompt_length > 0:
            raise ValueError(
                "Selective prefill resulted in zero tokens being kept. "
                "This can happen if `max_capacity_prompt` is too small or zero."
            )

        num_kept_tokens_for_base_prefill = final_indices_for_selective_prefill.numel()
        run_metadata["selective_prefill_original_len"] = prompt_length
        run_metadata["selective_prefill_kept_token_count"] = num_kept_tokens_for_base_prefill
        base_model_first_token_gen_start_time = time.perf_counter()
        
        base_model_cache_after_prefill: Optional[Cache] = None
        base_model_next_token_ids: Optional[torch.Tensor] = None
        is_full_contiguous_prefill = torch.equal(final_indices_for_selective_prefill, torch.arange(prompt_length, device=self.device))
        
        if is_full_contiguous_prefill:
            base_prefill_cache_pos = torch.arange(prompt_length, device=self.device)
            with torch.no_grad():
                base_out = self.base_model(
                    input_ids=prompt_input_ids, use_cache=True,
                    cache_position=base_prefill_cache_pos, past_key_values=DynamicCache()
                )
            base_model_next_token_ids = torch.argmax(base_out.logits[:, -1, :], dim=-1, keepdim=True)
            base_model_cache_after_prefill = base_out.past_key_values
        else: # Selective prefill case
            selected_ids = prompt_input_ids[:, final_indices_for_selective_prefill]
            selected_pos_ids = final_indices_for_selective_prefill.unsqueeze(0).to(torch.long)
            selective_cache_pos = torch.arange(selected_ids.shape[1], device=self.device)
            with torch.no_grad():
                selective_out = self.base_model(
                    selected_ids, position_ids=selected_pos_ids,
                    past_key_values=DynamicCache(), use_cache=True,
                    cache_position=selective_cache_pos
                )
            base_model_cache_after_sel_prefill = selective_out.past_key_values
            knockout_tokens = prompt_input_ids[:, -1:]
            knockout_pos_ids = torch.tensor([[prompt_length - 1]], device=self.device, dtype=torch.long)
            cache_len_after_sel = base_model_cache_after_sel_prefill.get_seq_length(0)
            knockout_cache_pos_base = torch.tensor([cache_len_after_sel], device=self.device, dtype=torch.long)
            with torch.no_grad():
                first_token_out = self.base_model(
                    knockout_tokens, position_ids=knockout_pos_ids,
                    past_key_values=base_model_cache_after_sel_prefill, use_cache=True,
                    cache_position=knockout_cache_pos_base
                )
            base_model_next_token_ids = torch.argmax(first_token_out.logits[:, -1, :], dim=-1, keepdim=True)
            base_model_cache_after_prefill = first_token_out.past_key_values
        
        if prompt_length > 0:
            run_metadata["token_keep_rate"] = (num_kept_tokens_for_base_prefill / prompt_length) * 100.0
        else:
            run_metadata["token_keep_rate"] = 100.0
        
        base_model_first_token_time = time.perf_counter() - base_model_first_token_gen_start_time
        run_metadata["base_prefill"] = base_model_first_token_time
        run_metadata["base_ttft"] = speculation_prefill_time + speculation_decode_time + base_model_first_token_time
        gen_token_ids_list: List[int] = []
        final_gen_text = ""
        
        decode_total_time = 0.0
        if base_model_next_token_ids is not None and base_model_cache_after_prefill is not None:
            first_gen_token_id = base_model_next_token_ids.item()
            gen_token_ids_list.append(first_gen_token_id)
            current_decode_tokens = base_model_next_token_ids
            current_decode_kv_cache: Cache = base_model_cache_after_prefill
            current_real_pos = prompt_length
            current_cache_write_pos = current_decode_kv_cache.get_seq_length(0)
            if first_gen_token_id not in self.eos_token_ids:
                for _ in range(max_generation_length - 1):
                    decode_step_start_time = time.perf_counter()
                    decode_pos_ids = torch.tensor([[current_real_pos]], device=self.device, dtype=torch.long)
                    decode_cache_pos = torch.tensor([current_cache_write_pos], device=self.device, dtype=torch.long)
                    with torch.no_grad():
                        decode_out = self.base_model(
                            current_decode_tokens, position_ids=decode_pos_ids,
                            past_key_values=current_decode_kv_cache, use_cache=True,
                            cache_position=decode_cache_pos
                        )
                    decode_total_time += time.perf_counter() - decode_step_start_time
                    
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

        run_metadata["decode_time"] = decode_total_time
        run_metadata["generated_tokens"] = len(gen_token_ids_list)
        run_metadata["total_time"] = time.perf_counter() - overall_start_time
        if final_gen_text.startswith("assistant\n\n"):
            final_gen_text = final_gen_text[len("assistant\n\n"):]
        elif final_gen_text.startswith(" assistant\n"):
            final_gen_text = final_gen_text[len(" assistant\n"):]
        return final_gen_text, run_metadata
