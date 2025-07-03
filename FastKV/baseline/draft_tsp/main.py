# baseline/draft_tsp/main.py

import torch
import torch.nn.functional as F
import types
import math
import time
from functools import partial
from typing import List, Dict, Tuple, Optional, Any

from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from transformers.cache_utils import Cache, DynamicCache
from transformers.models.llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb, repeat_kv

from baseline.draft_tsp.monkeypatch import replace_llama_for_draft_tsp, set_speculator_data


class DraftTSPPipeline:
    def __init__(self, base_model_name: str, speculator_model_name: str, tokenizer: AutoTokenizer, args, detailed_timing: bool = False):
        self.tokenizer = tokenizer
        self.args = args
        self.detailed_timing = detailed_timing
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("Loading Speculator (Draft) Model...")
        self.speculator_model: PreTrainedModel = self._load_model(speculator_model_name)

        print("Loading Base Model...")
        self.base_model: PreTrainedModel = self._load_model(base_model_name)

        print("Patching for Draft TSP...")
        replace_llama_for_draft_tsp(self.base_model, args)
        
        self.captured_qs: List[List[torch.Tensor]] = []
        self.orig_spec_fwds: Dict[int, types.MethodType] = {}
        self.eos_token_ids = self._extract_eos_token_ids()

    def _extract_eos_token_ids(self) -> List[int]:
        config_eos = self.base_model.config.eos_token_id
        if isinstance(config_eos, int): return [config_eos]
        if isinstance(config_eos, list): return list(config_eos)
        if self.tokenizer.eos_token_id: return [self.tokenizer.eos_token_id]
        raise ValueError("eos_token_id not defined in config or tokenizer.")

    def _load_model(self, model_name: str) -> PreTrainedModel:
        """Loads a model with flash_attention_2."""
        return AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation="flash_attention_2"
        ).eval()

    def _patch_speculator(self):
        """Patches the speculator's attention layers to capture Q-vectors for scoring."""
        num_layers = self.speculator_model.config.num_hidden_layers
        self.captured_qs = [[] for _ in range(num_layers)]
        for i, layer in enumerate(self.speculator_model.model.layers):
            attn = layer.self_attn
            if i not in self.orig_spec_fwds:
                self.orig_spec_fwds[i] = attn.forward
            attn.forward = types.MethodType(
                partial(_patched_attention_forward_for_capture, pipeline_instance=self), 
                attn
            )

    def _unpatch_speculator(self):
        """Restores the original forward methods on the speculator."""
        for i, layer in enumerate(self.speculator_model.model.layers):
            if i in self.orig_spec_fwds:
                layer.self_attn.forward = self.orig_spec_fwds[i]

    def _get_token_importance_scores(self, prompt_input_ids: torch.Tensor, look_ahead_k: int, run_metadata: Dict) -> torch.Tensor:
        """Runs the speculator, generates lookaheads, and computes token importance scores."""
        [q_list.clear() for q_list in self.captured_qs]
        
        if self.detailed_timing:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

        # --- Stage 1a: Speculator Prefill ---
        if self.detailed_timing: start_event.record()
        with torch.no_grad():
            spec_out = self.speculator_model(input_ids=prompt_input_ids, use_cache=True)
        if self.detailed_timing:
            end_event.record()
            torch.cuda.synchronize()
            run_metadata["speculation_prefill"] = start_event.elapsed_time(end_event) / 1000.0
        
        if isinstance(spec_out.past_key_values, tuple):
            spec_prefill_cache_tuple = spec_out.past_key_values
            spec_lookahead_cache = DynamicCache.from_legacy_cache(spec_prefill_cache_tuple)
        else:
            spec_lookahead_cache = spec_out.past_key_values
            spec_prefill_cache_tuple = spec_lookahead_cache.to_legacy_cache()

        # --- Stage 1b: Speculator Lookahead & Scoring ---
        if self.detailed_timing: start_event.record()
        self._patch_speculator()
        with torch.no_grad():
            current_tokens = spec_out.logits[:, -1:, :].argmax(-1)
            for i in range(look_ahead_k):
                cache_len = spec_lookahead_cache.get_seq_length(0)
                pos_ids = torch.tensor([[cache_len]], device=self.device, dtype=torch.long)
                outputs = self.speculator_model(input_ids=current_tokens, past_key_values=spec_lookahead_cache, use_cache=True, position_ids=pos_ids)
                
                if isinstance(outputs.past_key_values, tuple): spec_lookahead_cache = DynamicCache.from_legacy_cache(outputs.past_key_values)
                else: spec_lookahead_cache = outputs.past_key_values
                current_tokens = outputs.logits[:, -1:, :].argmax(-1)
        self._unpatch_speculator()

        all_layer_logits = []
        spec_config = self.speculator_model.config
        for layer_idx in range(spec_config.num_hidden_layers):
            if not self.captured_qs[layer_idx] or not spec_prefill_cache_tuple[layer_idx][0].numel(): continue
            
            q_vecs = torch.cat(self.captured_qs[layer_idx], dim=2)
            k_cache = spec_prefill_cache_tuple[layer_idx][0]
            k_cache_rep = repeat_kv(k_cache, spec_config.num_attention_heads // spec_config.num_key_value_heads)
            logits = torch.matmul(q_vecs, k_cache_rep.transpose(2, 3)) / math.sqrt(q_vecs.shape[-1])
            all_layer_logits.append(logits)
        
        if not all_layer_logits: raise RuntimeError("Failed to capture any Q-vectors from the speculator.")
        attention_scores = torch.stack(all_layer_logits).permute(1, 3, 0, 2, 4)
        bs, num_steps, num_layers, num_heads, key_len = attention_scores.shape
        if bs != 1: raise NotImplementedError("Batch size > 1 is not supported.")

        attention_probs = F.softmax(attention_scores, dim=-1, dtype=torch.float32).to(attention_scores.dtype)
        
        if hasattr(self.args, 'kernel_size') and self.args.pooling in ['avgpool', 'maxpool'] and self.args.kernel_size > 1:
            reshaped_for_pooling = attention_probs.squeeze(0).flatten(0, 2)
            
            padding = (self.args.kernel_size - 1) // 2
            pool_fn = F.avg_pool1d if self.args.pooling == 'avgpool' else F.max_pool1d
            
            pooled_tensor = pool_fn(reshaped_for_pooling, kernel_size=self.args.kernel_size, stride=1, padding=padding)
            processed_scores = pooled_tensor.reshape(num_steps, num_layers, num_heads, key_len)
        else:
            processed_scores = attention_probs.squeeze(0)
        
        flattened_scores = processed_scores.flatten(1, 2)
        max_scores = flattened_scores.max(1).values
        final_token_importance = max_scores.mean(0)
        
        if self.detailed_timing:
            end_event.record()
            torch.cuda.synchronize()
            run_metadata["speculation_decode"] = start_event.elapsed_time(end_event) / 1000.0
        return torch.argsort(final_token_importance, descending=True)

    def run(self, input_ids: torch.Tensor, max_generation_length: int, look_ahead_k: int) -> Tuple[str, Dict]:
        run_metadata: Dict[str, Any] = {}
        prompt_len = input_ids.shape[1]

        sorted_indices = self._get_token_importance_scores(input_ids, look_ahead_k, run_metadata)

        if self.detailed_timing:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()

        set_speculator_data(sorted_indices, prompt_len)
        with torch.no_grad():
            prefill_outputs = self.base_model(input_ids=input_ids, use_cache=True)
            past_key_values = prefill_outputs.past_key_values
            next_token_logits = prefill_outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

        if self.detailed_timing:
            end_event.record()
            torch.cuda.synchronize()
            base_model_first_token_time = start_event.elapsed_time(end_event) / 1000.0
            run_metadata["base_prefill"] = base_model_first_token_time
            run_metadata["base_ttft"] = run_metadata.get("speculation_prefill", 0) + run_metadata.get("speculation_decode", 0) + base_model_first_token_time

        generated_ids = [next_token.item()]
        for _ in range(max_generation_length - 1):
            if generated_ids[-1] in self.eos_token_ids:
                break

            cache_len = past_key_values.get_seq_length(0)
            current_token_pos = torch.tensor([[cache_len]], device=self.device, dtype=torch.long)
            
            outputs = self.base_model(
                input_ids=next_token,
                position_ids=current_token_pos,
                cache_position=current_token_pos,
                past_key_values=past_key_values,
                use_cache=True
            )
            past_key_values = outputs.past_key_values

            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            generated_ids.append(next_token.item())
        
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return response, run_metadata


def _patched_attention_forward_for_capture(
    self_attn: LlamaAttention, 
    pipeline_instance: 'DraftTSPPipeline', 
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
    """
    A patched LlamaAttention forward pass that captures Q-vectors without re-running the original forward pass.
    This is achieved by re-implementing the attention logic here.
    """
    bsz, q_len, _ = hidden_states.size()

    query_states = self_attn.q_proj(hidden_states)
    key_states = self_attn.k_proj(hidden_states)
    value_states = self_attn.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self_attn.num_heads, self_attn.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self_attn.num_key_value_heads, self_attn.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self_attn.num_key_value_heads, self_attn.head_dim).transpose(1, 2)

    cos, sin = self_attn.rotary_emb(value_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
    
    if q_len == 1: 
        pipeline_instance.captured_qs[self_attn.layer_idx].append(query_states.detach().clone())

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self_attn.layer_idx, cache_kwargs)

    key_states = repeat_kv(key_states, self_attn.num_key_value_groups)
    value_states = repeat_kv(value_states, self_attn.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self_attn.head_dim)

    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = F.dropout(attn_weights, p=self_attn.attention_dropout, training=self_attn.training)
    
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self_attn.hidden_size)
    attn_output = self_attn.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value