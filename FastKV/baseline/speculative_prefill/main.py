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
# --- End QTIP Imports ---

def _patched_attention_forward(
    self_attn: Union[LlamaAttention, Any],
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
    assert position_embeddings is not None, "Patched attention requires 'position_embeddings' to be passed."
    
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

    cos, sin = position_embeddings
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

    attn_mask_input = attention_mask
    is_sdpa_causal = (query_length > 1) and (attn_mask_input is None)

    if attn_mask_input is not None:
        is_sdpa_causal = False
        actual_kv_sequence_length = key_states_for_sdpa.shape[-2]
        if attn_mask_input.shape[-1] > actual_kv_sequence_length:
            attn_mask_input = attn_mask_input[:, :, :, :actual_kv_sequence_length]

    attention_output = F.scaled_dot_product_attention(
        query_states_rotated, key_states_for_sdpa, value_states_for_sdpa, attn_mask=attn_mask_input, dropout_p=0.0, is_causal=is_sdpa_causal
    )

    attention_output = attention_output.transpose(1, 2).contiguous().reshape(batch_size, query_length, hidden_size)
    attention_output = self_attn.o_proj(attention_output)

    return attention_output, None, past_key_value


class SpeculativePrefillPipeline:
    def __init__(
        self,
        base_model_name: str,
        speculator_model_name: str,
        tokenizer: AutoTokenizer,
        max_capacity_prompt: int = 512,
        max_capacity_prompt_percentage: Optional[float] = None,
        pool_kernel_size: Optional[int] = 13,
        pool_type: str = 'avgpool',
        use_chunk_selection: bool = True,
        chunk_size: int = 32,
        detailed_timing: bool = False,
    ):
        self.base_model_name = base_model_name
        self.speculator_model_name = speculator_model_name
        self.max_capacity_prompt = max_capacity_prompt
        self.max_capacity_prompt_percentage = max_capacity_prompt_percentage
        self.pool_kernel_size = pool_kernel_size
        self.pool_type = pool_type.lower()
        self.use_chunk_selection = use_chunk_selection
        self.chunk_size = chunk_size
        self.detailed_timing = detailed_timing
        self._validate_config()
        self.tokenizer = tokenizer
        self.speculator_model = self._load_model(self.speculator_model_name)
        self.base_model = self._load_model(self.base_model_name)
        self.base_config: AutoConfig = self.base_model.config
        self.device = self.speculator_model.device
        self.dtype = self.speculator_model.dtype
        self.eos_token_ids = self._extract_eos_token_ids()
        self.captured_qs: List[List[torch.Tensor]] = []
        self.orig_spec_fwds: Dict[int, Any] = {}
        self.is_prefilling = False
        self.token_importance_scores: Optional[torch.Tensor] = None

    def _validate_config(self):
        if self.max_capacity_prompt is not None and self.max_capacity_prompt_percentage is not None:
            raise ValueError("Only one of `max_capacity_prompt` and `max_capacity_prompt_percentage` can be specified.")
        if self.max_capacity_prompt_percentage is not None and not (0.0 < self.max_capacity_prompt_percentage <= 1.0):
            raise ValueError("`max_capacity_prompt_percentage` must be between 0.0 and 1.0.")

    def _validate_config(self):
        if self.pool_type not in ['avgpool', 'maxpool', 'none']:
            raise ValueError(f"pool_type must be 'avgpool', 'maxpool', or 'none', but got {self.pool_type}")
        if self.pool_kernel_size is not None:
            if self.pool_kernel_size <= 1: self.pool_kernel_size = None
            elif self.pool_kernel_size % 2 == 0: raise ValueError("pool_kernel_size must be an odd number.")
            if self.pool_type == 'none': raise ValueError("pool_kernel_size is specified, but pool_type is 'none'.")
        if self.pool_type != 'none' and self.pool_kernel_size is None:
            raise ValueError(f"pool_type is '{self.pool_type}', but pool_kernel_size is not specified.")
        if self.chunk_size <= 0: raise ValueError("chunk_size must be a positive integer.")

    def _extract_eos_token_ids(self) -> List[int]:
        config_eos = self.base_config.eos_token_id
        if isinstance(config_eos, int): return [config_eos]
        if isinstance(config_eos, list): return list(config_eos)
        if self.tokenizer.eos_token_id: return [self.tokenizer.eos_token_id]
        raise ValueError("eos_token_id not defined in config or tokenizer.")

    def _is_qtip_model(self, model_name: str) -> bool:
        return QTIP_AVAILABLE and ("relaxml" in model_name.lower() or "qtip" in model_name.lower())

    def _load_model(self, model_name: str) -> AutoModelForCausalLM:
        if self._is_qtip_model(model_name):
            if not QTIP_AVAILABLE: raise ImportError(f"QTIP model {model_name} requested but modules not available.")
            print(f"Loading QTIP model: {model_name}")
            model, _ = qtip_model_from_hf_path(model_name, max_mem_ratio=0.7, attn_implementation="sdpa")
        else:
            print(f"Loading standard HF model: {model_name}")
            model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float16, trust_remote_code=True, device_map="auto", attn_implementation="sdpa"
            )
        return model.eval()

    def _patch_speculator(self) -> int:
        if not hasattr(self.speculator_model, 'model') or not hasattr(self.speculator_model.model, 'layers'): return 0
        num_layers = self.speculator_model.config.num_hidden_layers
        self.captured_qs = [[] for _ in range(num_layers)]
        num_patched_layers = 0
        for i, layer in enumerate(self.speculator_model.model.layers):
            if hasattr(layer, 'self_attn'):
                attn_module = layer.self_attn
                if isinstance(attn_module, LlamaAttention) or (QTIP_AVAILABLE and isinstance(attn_module, QTIP_LlamaAttention)):
                    if i not in self.orig_spec_fwds: self.orig_spec_fwds[i] = attn_module.forward
                    attn_module.forward = types.MethodType(partial(_patched_attention_forward, pipeline_instance=self), attn_module)
                    num_patched_layers += 1
        return num_patched_layers

    def _token_importance_from_attn_scores(self, attention_scores: torch.Tensor):
        if attention_scores.numel() == 0: raise RuntimeError("Cannot calculate importance from empty attention scores.")
        
        attention_scores = attention_scores.permute(0, 3, 1, 2, 4)
        bs, num_steps, num_layers, num_heads, key_len = attention_scores.shape
        if bs != 1:
            raise NotImplementedError("Batch size > 1 is not supported.")

        attention_probs = F.softmax(attention_scores, dim=-1, dtype=torch.float32).to(attention_scores.dtype)

        if self.pool_kernel_size and self.pool_type != 'none':
            reshaped_for_pooling = attention_probs.squeeze(0).flatten(0, 2)
            
            padding = (self.pool_kernel_size - 1) // 2
            pool_fn = F.avg_pool1d if self.pool_type == 'avgpool' else F.max_pool1d
            
            pooled_tensor = pool_fn(reshaped_for_pooling, kernel_size=self.pool_kernel_size, stride=1, padding=padding)
            processed_scores = pooled_tensor.reshape(num_steps, num_layers, num_heads, key_len)
        else:
            processed_scores = attention_probs

        flattened_scores = processed_scores.flatten(1, 2)
        max_scores = flattened_scores.max(1).values
        final_token_importance = max_scores.mean(0)
        
        self.token_importance_scores = final_token_importance.unsqueeze(0)

    def _compute_raw_qk_scores(self, speculator_prefill_cache_as_tuple) -> torch.Tensor:
        if not self.captured_qs or not any(self.captured_qs): raise RuntimeError("Speculator Q-vectors were not captured.")
        spec_config = self.speculator_model.config
        num_spec_layers, spec_num_q_heads, spec_num_kv_heads = spec_config.num_hidden_layers, spec_config.num_attention_heads, spec_config.num_key_value_heads
        spec_num_kv_groups = spec_num_q_heads // spec_num_kv_heads
        all_layer_scores = []
        for layer_idx in range(num_spec_layers):
            if not self.captured_qs[layer_idx] or speculator_prefill_cache_as_tuple[layer_idx][0].numel() == 0: continue
            key_prompt_layer_rep_spec = hf_repeat_kv(speculator_prefill_cache_as_tuple[layer_idx][0].detach(), spec_num_kv_groups)
            all_q_for_layer = torch.cat(self.captured_qs[layer_idx], dim=2)
            attn_logits = torch.matmul(all_q_for_layer, key_prompt_layer_rep_spec.transpose(-1, -2)) / math.sqrt(all_q_for_layer.shape[-1])
            all_layer_scores.append(attn_logits)
        return torch.stack(all_layer_scores, dim=1)

    def _select_tokens_by_chunk(self, scores: torch.Tensor, num_to_keep: int, original_seq_len: int) -> torch.Tensor:
        if original_seq_len <= num_to_keep: return torch.arange(original_seq_len, device=scores.device)
        num_chunks = math.ceil(original_seq_len / self.chunk_size)
        padding_len = num_chunks * self.chunk_size - original_seq_len
        padded_scores = F.pad(scores, (0, padding_len), value=float('-inf')) if padding_len > 0 else scores
        avg_chunk_scores = padded_scores.view(num_chunks, self.chunk_size).mean(dim=1)
        num_chunks_to_keep = min(math.ceil(num_chunks * (num_to_keep / original_seq_len)), num_chunks)
        _, top_chunk_indices = torch.topk(avg_chunk_scores, k=num_chunks_to_keep)
        selected_indices = torch.cat([torch.arange(idx * self.chunk_size, (idx + 1) * self.chunk_size, device=scores.device) for idx in top_chunk_indices])
        final_indices = selected_indices[selected_indices < original_seq_len]
        if len(final_indices) > num_to_keep:
            _, top_k_in_chunks_indices = torch.topk(scores[final_indices], k=num_to_keep)
            final_indices = final_indices[top_k_in_chunks_indices]
        return torch.sort(final_indices)[0]

    def _calculate_indices_to_keep(self, original_seq_len: int) -> torch.Tensor:
        if self.max_capacity_prompt_percentage is not None:
            num_to_keep = int(original_seq_len * self.max_capacity_prompt_percentage)
        else:
            num_to_keep = min(self.max_capacity_prompt, original_seq_len)

        if num_to_keep >= original_seq_len: return torch.arange(original_seq_len, device=self.device, dtype=torch.long)
        if self.token_importance_scores is None: raise RuntimeError("Token importance scores not computed.")
        scores_for_selection = self.token_importance_scores[0].clone()
        if self.use_chunk_selection:
            return self._select_tokens_by_chunk(scores_for_selection, num_to_keep, original_seq_len)
        else:
            _, indices = torch.topk(scores_for_selection, k=num_to_keep, dim=-1)
            return torch.sort(indices)[0]

    def run(self, input_ids: torch.Tensor, look_ahead_k: int, max_generation_length: int) -> Tuple[str, Dict[str, Any]]:
        run_metadata: Dict[str, Any] = {"max_capacity_prompt": self.max_capacity_prompt}
        
        if self.detailed_timing:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

        num_patched_layers = self._patch_speculator()
        if num_patched_layers == 0 and look_ahead_k > 0:
            raise RuntimeError("Speculator model could not be patched to capture Q-vectors.")

        prompt_length = input_ids.shape[1]
        run_metadata["prompt_input_length"] = prompt_length
        if prompt_length == 0:
            run_metadata["token_keep_rate"] = 100.0
            return "", run_metadata

        # --- Stage 1: Speculator Prefill ---
        if self.detailed_timing: start_event.record()
        self.is_prefilling = True
        with torch.no_grad():
            spec_out = self.speculator_model(input_ids=input_ids, use_cache=True, cache_position=torch.arange(prompt_length, device=self.device))
        if self.detailed_timing:
            end_event.record()
            torch.cuda.synchronize()
            run_metadata["speculation_prefill"] = start_event.elapsed_time(end_event) / 1000.0
        
        speculator_prefill_cache = spec_out.past_key_values
        if speculator_prefill_cache is None: raise RuntimeError("Speculator prefill did not return a KV cache.")
        
        speculator_prefill_cache_as_tuple = speculator_prefill_cache.to_legacy_cache() if isinstance(speculator_prefill_cache, DynamicCache) else speculator_prefill_cache
        current_spec_cache = DynamicCache.from_legacy_cache(speculator_prefill_cache_as_tuple)
        speculator_next_token_ids = torch.argmax(spec_out.logits[:, -1, :], dim=-1, keepdim=True)

        # --- Stage 2: Speculator Lookahead & Scoring ---
        if self.detailed_timing: start_event.record()
        self.is_prefilling = False
        with torch.no_grad():
            current_spec_tokens = speculator_next_token_ids
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
                if isinstance(current_spec_cache, tuple):
                    current_spec_cache = DynamicCache.from_legacy_cache(current_spec_cache)
                current_spec_tokens = torch.argmax(lookahead_out.logits[:, -1, :], dim=-1, keepdim=True)
                if current_spec_tokens.item() in self.eos_token_ids: break

        raw_qk_scores = self._compute_raw_qk_scores(speculator_prefill_cache_as_tuple)
        self._token_importance_from_attn_scores(raw_qk_scores)
        if self.detailed_timing:
            end_event.record()
            torch.cuda.synchronize()
            run_metadata["speculation_decode"] = start_event.elapsed_time(end_event) / 1000.0

        # --- Stage 3: Base Model Selective Prefill ---
        if self.detailed_timing: start_event.record()
        indices_to_keep = self._calculate_indices_to_keep(prompt_length)
        selected_prompt_ids = input_ids[:, indices_to_keep]
        selective_cache_pos = torch.arange(selected_prompt_ids.shape[1], device=self.device)
        selective_pos_ids = indices_to_keep.unsqueeze(0).to(torch.long)
        run_metadata["selective_prefill_original_len"] = prompt_length
        run_metadata["selective_prefill_kept_token_count"] = selected_prompt_ids.shape[1]

        with torch.no_grad():
            base_out = self.base_model(
                selected_prompt_ids,
                position_ids=selective_pos_ids,
                use_cache=True,
                cache_position=selective_cache_pos
            )
        base_model_next_token_ids, base_model_cache_after_prefill = torch.argmax(base_out.logits[:, -1, :], dim=-1, keepdim=True), base_out.past_key_values
        if self.detailed_timing:
            end_event.record()
            torch.cuda.synchronize()
            run_metadata["base_prefill"] = start_event.elapsed_time(end_event) / 1000.0

        run_metadata["token_keep_rate"] = (selected_prompt_ids.shape[1] / prompt_length * 100.0) if prompt_length > 0 else 100.0
        if self.detailed_timing:
            run_metadata["base_ttft"] = run_metadata.get("speculation_prefill", 0) + run_metadata.get("speculation_decode", 0) + run_metadata.get("base_prefill", 0)

        # --- Stage 4: Base Model Generation ---
        gen_token_ids_list: List[int] = []
        if base_model_next_token_ids is not None and base_model_cache_after_prefill is not None:
            first_gen_token_id = base_model_next_token_ids.item()
            gen_token_ids_list.append(first_gen_token_id)
            current_decode_tokens = base_model_next_token_ids
            current_decode_kv_cache = base_model_cache_after_prefill
            if isinstance(current_decode_kv_cache, tuple):
                current_decode_kv_cache = DynamicCache.from_legacy_cache(current_decode_kv_cache)

            if first_gen_token_id not in self.eos_token_ids:
                for i in range(max_generation_length - 1):
                    current_cache_len = current_decode_kv_cache.get_seq_length(0)
                    pos_ids = torch.tensor([[selected_prompt_ids.shape[1] + i]], device=self.device)
                    decode_cache_pos = torch.tensor([current_cache_len], device=self.device)
                    
                    with torch.no_grad():
                        decode_out = self.base_model(
                            current_decode_tokens,
                            position_ids=pos_ids,
                            past_key_values=current_decode_kv_cache,
                            use_cache=True,
                            cache_position=decode_cache_pos
                        )
                    next_tokens = torch.argmax(decode_out.logits[:, -1, :], dim=-1, keepdim=True)
                    gen_token_ids_list.append(next_tokens.item())
                    current_decode_kv_cache = decode_out.past_key_values
                    if isinstance(current_decode_kv_cache, tuple):
                        current_decode_kv_cache = DynamicCache.from_legacy_cache(current_decode_kv_cache)
                    current_decode_tokens = next_tokens
                    if gen_token_ids_list[-1] in self.eos_token_ids: break

        final_gen_text = self.tokenizer.decode(gen_token_ids_list, skip_special_tokens=True)
        if final_gen_text.startswith("assistant\n\n"): final_gen_text = final_gen_text[len("assistant\n\n"):]
        elif final_gen_text.startswith(" assistant\n"): final_gen_text = final_gen_text[len(" assistant\n"):]
        return final_gen_text, run_metadata


def main():
    parser = argparse.ArgumentParser(description="Speculative Prefill Pipeline with QTIP Support")
    parser.add_argument("--base_model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--speculator_model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--dataset_name", type=str, default="qasper")
    parser.add_argument("--look_ahead_k", type=int, default=8)
    parser.add_argument("--max_capacity_prompt", type=int, default=512)
    parser.add_argument("--max_capacity_prompt_percentage", type=float, default=None)
    parser.add_argument("--max_generation_length", type=int, default=128)
    parser.add_argument("--kernel_size", type=int, default=13)
    parser.add_argument("--pooling", type=str, default="avgpool", choices=['avgpool', 'maxpool', 'none'])
    parser.add_argument("--use_chunk_selection", action='store_true', help="Use chunk-based token selection (it's enabled by default).")
    parser.add_argument("--no_chunk_selection", dest="use_chunk_selection", action="store_false", help="Disable chunk-based selection.")
    parser.add_argument("--chunk_size", type=int, default=32)
    
    args = parser.parse_args()
    
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
        
    pipeline = SpeculativePrefillPipeline(
        base_model_name=args.base_model_name,
        speculator_model_name=args.speculator_model_name,
        tokenizer=tokenizer,
        max_capacity_prompt=args.max_capacity_prompt,
        max_capacity_prompt_percentage=args.max_capacity_prompt_percentage,
        pool_kernel_size=args.kernel_size if args.pooling != 'none' else None,
        pool_type=args.pooling,
        use_chunk_selection=args.use_chunk_selection,
        chunk_size=args.chunk_size,
    )
    
    prompt_str: str
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.join(current_dir, '..', '..')
        prompt_config_path = os.path.join(project_root, 'eval', 'longbench', 'config', 'dataset2prompt.json')

        if not os.path.exists(prompt_config_path):
            raise FileNotFoundError(f"Could not find the LongBench prompt config at the expected path: {prompt_config_path}")

        import json
        with open(prompt_config_path, "r") as f:
            dataset2prompt = json.load(f)
        
        dataset = load_dataset('THUDM/LongBench', args.dataset_name, split='test', trust_remote_code=True)
        sample = dataset[0]
        
        prompt_format = dataset2prompt.get(args.dataset_name)
        if prompt_format is None:
            raise KeyError(f"Prompt format for dataset '{args.dataset_name}' not found in {prompt_config_path}")

        prompt_str = prompt_format.format(context=sample['context'], input=sample['input'])
        
    except Exception as e:
        print(f"Could not load dataset or LongBench prompt format. Using a default prompt. Error: {e}")
        prompt_str = "Explain the theory of relativity in simple terms."
    
    selection_strategy = f"Chunk-based (size={pipeline.chunk_size})" if pipeline.use_chunk_selection else "Top-K"
    print(f"\n--- Running Speculative Prefill Pipeline ---")
    print(f"Base Model: {args.base_model_name}")
    print(f"Speculator Model: {args.speculator_model_name}")
    if args.max_capacity_prompt_percentage:
        print(f"Max Prompt Capacity: {args.max_capacity_prompt_percentage:.1%}")
    else:
        print(f"Max Prompt Capacity: {args.max_capacity_prompt} tokens")
    print(f"Lookahead K: {args.look_ahead_k}")
    print(f"Pooling: type='{pipeline.pool_type}', kernel_size={pipeline.pool_kernel_size}")
    print(f"Token Selection: {selection_strategy}")
    print(f"Dataset: {args.dataset_name}")
    print("-" * 43)

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