import argparse
import json
import math
import os
import pickle
import types
import zipfile
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          set_seed)
from transformers.cache_utils import Cache, DynamicCache
from transformers.models.llama.modeling_llama import (
    LlamaAttention, apply_rotary_pos_emb as hf_apply_rotary_pos_emb,
    repeat_kv as hf_repeat_kv)


# --- Base Class ---

class BaseRankingGenerator:
    """A base class for generating token rankings using different strategies."""
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, args: argparse.Namespace):
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.config = self.model.config
        self.device = self.model.device
        self.orig_fwds: Dict[int, Any] = {}

    def generate_rankings(self, inputs: Dict[str, torch.Tensor]) -> Dict:
        raise NotImplementedError

    def _unpatch_model(self):
        for i, layer in enumerate(self.model.model.layers):
            if hasattr(layer, 'self_attn') and i in self.orig_fwds:
                layer.self_attn.forward = self.orig_fwds[i]
        self.orig_fwds.clear()

# --- Strategy 1: FastKV ---

class FastKVRankingGenerator(BaseRankingGenerator):
    def __init__(self, model, tokenizer, args):
        super().__init__(model, tokenizer, args)
        self.rankings: Dict[int, torch.Tensor] = {}
        
    def _patched_attention_forward(self, self_attn: LlamaAttention, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, position_ids: Optional[torch.LongTensor] = None, past_key_value: Optional[Cache] = None, use_cache: bool = False, **kwargs):
        bsz, q_len, _ = hidden_states.size()
        if self_attn.layer_idx not in self.orig_fwds or q_len <= self.args.window_size:
            original_forward = self.orig_fwds.get(self_attn.layer_idx, self_attn.__class__.forward)
            return original_forward(self_attn, hidden_states=hidden_states, attention_mask=attention_mask, position_ids=position_ids, past_key_value=past_key_value, use_cache=use_cache, **kwargs)

        cfg = self_attn.config
        num_heads, head_dim = cfg.num_attention_heads, cfg.hidden_size // cfg.num_attention_heads
        num_kv_heads, num_kv_groups = cfg.num_key_value_heads, cfg.num_attention_heads // cfg.num_key_value_heads

        query_states = self_attn.q_proj(hidden_states).view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
        key_states = self_attn.k_proj(hidden_states).view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
        value_states = self_attn.v_proj(hidden_states).view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
        
        cos, sin = self_attn.rotary_emb(value_states, position_ids=position_ids)
        query_states, key_states = hf_apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        with torch.no_grad():
            scoring_queries = query_states[..., -self.args.window_size:, :].detach()
            key_states_for_scoring = hf_repeat_kv(key_states, num_kv_groups).detach()
            attn_logits = torch.matmul(scoring_queries, key_states_for_scoring.transpose(2, 3)) / math.sqrt(head_dim)
            attn_probs = F.softmax(attn_logits, dim=-1, dtype=torch.bfloat16)
            token_scores = attn_probs[..., :, :-self.args.window_size].sum(dim=-2)

            if self.args.pooling != "none":
                pool_fn = F.avg_pool1d if self.args.pooling == "avgpool" else F.max_pool1d
                token_scores = pool_fn(token_scores, kernel_size=self.args.kernel_size, padding=self.args.kernel_size // 2, stride=1)

            final_scores = token_scores.sum(dim=1)
            padded_scores = torch.zeros(bsz, q_len, device=final_scores.device, dtype=final_scores.dtype)
            padded_scores[:, :-self.args.window_size] = final_scores
            padded_scores[:, -self.args.window_size:] = final_scores.max(dim=-1, keepdim=True)[0] + 1e-5
            self.rankings[self_attn.layer_idx] = padded_scores.squeeze(0).clone()

        attn_output = F.scaled_dot_product_attention(query_states, hf_repeat_kv(key_states, num_kv_groups), hf_repeat_kv(value_states, num_kv_groups), attn_mask=attention_mask, dropout_p=0.0, is_causal=(q_len > 1 and attention_mask is None))
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, cfg.hidden_size)
        return self_attn.o_proj(attn_output), None, past_key_value

    def _patch_model(self):
        for i, layer in enumerate(self.model.model.layers):
            if hasattr(layer, 'self_attn') and isinstance(layer.self_attn, LlamaAttention):
                setattr(layer.self_attn, "layer_idx", i)
                self.orig_fwds[i] = layer.self_attn.forward
                layer.self_attn.forward = types.MethodType(self._patched_attention_forward, layer.self_attn)

    @torch.no_grad()
    def generate_rankings(self, inputs: Dict[str, torch.Tensor]) -> Dict:
        self.rankings.clear()
        self._patch_model()
        self.model(input_ids=inputs['input_ids'], use_cache=False, position_ids=torch.arange(inputs['input_ids'].shape[1], device=self.device).unsqueeze(0))
        self._unpatch_model()
        return self.rankings


# --- Strategy 2: GemFilter ---

class GemFilterRankingGenerator(BaseRankingGenerator):
    def __init__(self, model, tokenizer, args):
        super().__init__(model, tokenizer, args)
        self.rankings: Dict[int, torch.Tensor] = {}

    def _patched_attention_forward(self, self_attn: LlamaAttention, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, position_ids: Optional[torch.LongTensor] = None, past_key_value: Optional[Cache] = None, use_cache: bool = False, **kwargs):
        bsz, q_len, _ = hidden_states.size()
        cfg = self_attn.config
        num_heads, head_dim = cfg.num_attention_heads, cfg.hidden_size // cfg.num_attention_heads
        num_kv_heads, num_kv_groups = cfg.num_key_value_heads, cfg.num_attention_heads // cfg.num_key_value_heads

        query_states = self_attn.q_proj(hidden_states).view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
        key_states = self_attn.k_proj(hidden_states).view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
        value_states = self_attn.v_proj(hidden_states).view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
        cos, sin = self_attn.rotary_emb(value_states, position_ids=position_ids)
        query_states, key_states = hf_apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        with torch.no_grad():
            query_last = query_states[:, :, -1:, :].detach()
            key_states_for_scoring = hf_repeat_kv(key_states, num_kv_groups).detach()
            inner_product = torch.matmul(query_last, key_states_for_scoring.transpose(-1, -2)).squeeze(2)
            token_scores = inner_product.sum(dim=1, keepdim=True)

            if self.args.pooling != "none":
                pool_fn = F.avg_pool1d if self.args.pooling == "avgpool" else F.max_pool1d
                token_scores = pool_fn(token_scores, kernel_size=self.args.kernel_size, padding=self.args.kernel_size // 2, stride=1)
            
            self.rankings[self_attn.layer_idx] = token_scores.squeeze().clone()

        attn_output = F.scaled_dot_product_attention(query_states, hf_repeat_kv(key_states, num_kv_groups), hf_repeat_kv(value_states, num_kv_groups), attn_mask=attention_mask, dropout_p=0.0, is_causal=(q_len > 1 and attention_mask is None))
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, cfg.hidden_size)
        return self_attn.o_proj(attn_output), None, past_key_value

    def _patch_model(self):
        for i, layer in enumerate(self.model.model.layers):
            if hasattr(layer, 'self_attn') and isinstance(layer.self_attn, LlamaAttention):
                setattr(layer.self_attn, "layer_idx", i)
                self.orig_fwds[i] = layer.self_attn.forward
                layer.self_attn.forward = types.MethodType(self._patched_attention_forward, layer.self_attn)

    @torch.no_grad()
    def generate_rankings(self, inputs: Dict[str, torch.Tensor]) -> Dict:
        self.rankings.clear()
        self._patch_model()
        self.model(input_ids=inputs['input_ids'], use_cache=False, position_ids=torch.arange(inputs['input_ids'].shape[1], device=self.device).unsqueeze(0))
        self._unpatch_model()
        return self.rankings


# --- Strategy 3: Speculative (1:1 with Oracle) ---

def _patched_attention_forward_speculative(
    self_attn: LlamaAttention,
    generator_obj: 'SpeculativeRankingGenerator',
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
    batch_size, query_length, _ = hidden_states.size()
    num_heads = self_attn.config.num_attention_heads
    head_dim = self_attn.config.hidden_size // num_heads
    num_key_value_heads = self_attn.config.num_key_value_heads
    hidden_size = self_attn.config.hidden_size
    num_key_value_groups = num_heads // num_key_value_heads
    
    query_projection = self_attn.q_proj(hidden_states)
    key_projection = self_attn.k_proj(hidden_states)
    value_projection = self_attn.v_proj(hidden_states)
    
    query_states = query_projection.view(batch_size, query_length, num_heads, head_dim).transpose(1, 2)
    key_states_before_rope = key_projection.view(batch_size, query_length, num_key_value_heads, head_dim).transpose(1, 2)
    value_states_for_cache = value_projection.view(batch_size, query_length, num_key_value_heads, head_dim).transpose(1, 2)
    
    cos, sin = self_attn.rotary_emb(value_states_for_cache, position_ids=position_ids)
    query_states_rotated, key_states_rotated = hf_apply_rotary_pos_emb(query_states, key_states_before_rope, cos, sin, position_ids)

    # Only capture Qs during the speculative decoding phase
    if generator_obj.is_generating_speculatively and query_length == 1:
        generator_obj.captured_qs[self_attn.layer_idx].append(query_states_rotated.detach().clone())

    # This past_key_value is updated IN-PLACE. The original `prompt_only_cache` is preserved outside.
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


class SpeculativeRankingGenerator(BaseRankingGenerator):
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, args: argparse.Namespace):
        super().__init__(model, tokenizer, args)
        self.eos_token_ids = self._extract_eos_token_ids()
        self.lookahead_k_values = args.lookahead_k_values
        self.max_k = max(self.lookahead_k_values) if self.lookahead_k_values else 0
        self.pool_kernel_size = args.kernel_size if args.pooling != 'none' else None
        self.pool_type = args.pooling.lower()
        self._validate_pooling_config()
        self.captured_qs: List[List[torch.Tensor]] = []
        self.is_generating_speculatively = False
        self.rankings: Dict[int, torch.Tensor] = {} # Keyed by k, value is the ranking tensor

    def _validate_pooling_config(self):
        if self.pool_type not in ['avgpool', 'maxpool', 'none']:
            raise ValueError(f"pool_type must be 'avgpool', 'maxpool', or 'none', but got {self.pool_type}")
        if self.pool_kernel_size is not None:
            if self.pool_kernel_size <= 1: self.pool_kernel_size = None
            elif self.pool_kernel_size % 2 == 0: raise ValueError("pool_kernel_size must be an odd number.")
            if self.pool_type == 'none' and self.pool_kernel_size is not None: raise ValueError("pool_kernel_size is specified, but pool_type is 'none'.")
        if self.pool_type != 'none' and self.pool_kernel_size is None:
            raise ValueError(f"pool_type is '{self.pool_type}', but pool_kernel_size is not specified.")

    def _extract_eos_token_ids(self) -> List[int]:
        config_eos = self.config.eos_token_id
        if isinstance(config_eos, int): return [config_eos]
        if isinstance(config_eos, list): return list(config_eos)
        if self.tokenizer.eos_token_id: return [self.tokenizer.eos_token_id]
        raise ValueError("eos_token_id not defined in config or tokenizer.")

    def _patch_model(self):
        num_layers = self.config.num_hidden_layers
        self.captured_qs = [[] for _ in range(num_layers)]
        for i, layer in enumerate(self.model.model.layers):
            if hasattr(layer, 'self_attn') and isinstance(layer.self_attn, LlamaAttention):
                attn_module = layer.self_attn
                if i not in self.orig_fwds: self.orig_fwds[i] = attn_module.forward
                setattr(attn_module, "layer_idx", i)
                attn_module.forward = types.MethodType(partial(_patched_attention_forward_speculative, generator_obj=self), attn_module)

    def _compute_raw_qk_scores(self, prompt_only_cache_tuple: Tuple[Tuple[torch.Tensor, torch.Tensor], ...]) -> torch.Tensor:
        if not self.captured_qs or not any(self.captured_qs): return torch.empty(0)
        
        num_layers = self.config.num_hidden_layers
        num_q_heads = self.config.num_attention_heads
        num_kv_heads = self.config.num_key_value_heads
        num_kv_groups = num_q_heads // num_kv_heads
        head_dim = self.config.hidden_size // num_q_heads

        all_layer_scores = []
        for layer_idx in range(num_layers):
            if not self.captured_qs[layer_idx] or not prompt_only_cache_tuple[layer_idx][0].numel(): continue
            key_prompt_layer = prompt_only_cache_tuple[layer_idx][0].detach()
            key_prompt_layer_repeated = hf_repeat_kv(key_prompt_layer, num_kv_groups)
            all_q_for_layer = torch.cat(self.captured_qs[layer_idx], dim=2)
            attn_logits = torch.matmul(all_q_for_layer, key_prompt_layer_repeated.transpose(-1, -2)) / math.sqrt(head_dim)
            all_layer_scores.append(attn_logits)
        if not all_layer_scores: return torch.empty(0)

        # Shape: [batch_size, num_layers, num_heads, num_gen_tokens, prompt_len]
        return torch.stack(all_layer_scores, dim=1)

    def _token_importance_from_attn_scores(self, attn_scores: torch.Tensor):
        # This function now mirrors the logic from the corrected oracle script.
        if attn_scores.numel() == 0:
            return

        # Input shape: [batch_size, num_layers, num_heads, num_gen_tokens, prompt_len]
        bs, num_layers, num_heads, num_gen_tokens, prompt_len = attn_scores.shape

        if bs != 1:
            raise NotImplementedError("Batch size > 1 is not supported for speculative ranking.")
        
        # Softmax over the prompt length dimension
        probs = F.softmax(attn_scores, dim=-1, dtype=torch.bfloat16)

        # Permute to [B, N_gen, S_prompt, L, H] for easier aggregation
        probs = probs.permute(0, 3, 4, 1, 2)

        # Step 1: Max-aggregation over Layers and Heads
        peak_importance, _ = torch.max(torch.max(probs, dim=-1)[0], dim=-1)
        # Shape: [B, N_gen, S_prompt]

        # Now, calculate rankings for each 'k'
        for k in self.lookahead_k_values:
            if k > num_gen_tokens:
                continue
            
            # Step 2: Mean-aggregation over the first 'k' generated tokens
            scores_for_k = peak_importance[:, :k, :]
            mean_importance = torch.mean(scores_for_k, dim=1)
            # Shape: [B, S_prompt]

            # Step 3 (Optional): 1D pooling on the final aggregated scores
            final_scores = mean_importance
            kernel_size = self.pool_kernel_size
            if kernel_size and prompt_len >= kernel_size:
                scores_for_pooling = final_scores.unsqueeze(1) # Add channel dim
                
                pool_fn = F.avg_pool1d if self.pool_type == 'avgpool' else F.max_pool1d
                
                pooled_scores = pool_fn(
                    scores_for_pooling,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=kernel_size // 2
                )
                final_scores = pooled_scores.squeeze(1)
            
            self.rankings[k] = final_scores.squeeze(0).to(self.model.dtype)

    @torch.no_grad()
    def generate_rankings(self, inputs: Dict[str, torch.Tensor]) -> Dict[int, torch.Tensor]:
        self.rankings.clear()
        if self.max_k == 0: return {}
        self._patch_model()
        try:
            input_ids = inputs['input_ids']
            prompt_length = input_ids.shape[1]
            
            # --- Stage 1: Prompt Prefill ---
            self.is_generating_speculatively = False
            prefill_pos = torch.arange(prompt_length, device=self.device)
            prefill_out = self.model(input_ids=input_ids, use_cache=True, cache_position=prefill_pos)
            prefill_cache = prefill_out.past_key_values
            
            # **FIX**: Snapshot the prompt-only KV cache before speculative generation
            if isinstance(prefill_cache, tuple): prefill_cache = DynamicCache.from_legacy_cache(prefill_cache)
            prompt_only_cache_tuple = prefill_cache.to_legacy_cache()
            
            current_tokens = torch.argmax(prefill_out.logits[:, -1, :], dim=-1, keepdim=True)
            current_cache = prefill_cache

            # --- Stage 2: Speculative Token Generation ---
            self.is_generating_speculatively = True
            for _ in range(self.max_k):
                if current_tokens.item() in self.eos_token_ids: break
                
                cache_len = current_cache.get_seq_length(0)
                pos_ids = torch.tensor([[cache_len]], device=self.device)
                decode_cache_pos = torch.tensor([cache_len], device=self.device)
                
                decode_out = self.model(input_ids=current_tokens, position_ids=pos_ids, past_key_values=current_cache, use_cache=True, cache_position=decode_cache_pos)
                
                current_cache = decode_out.past_key_values
                if isinstance(current_cache, tuple): current_cache = DynamicCache.from_legacy_cache(current_cache)
                
                current_tokens = torch.argmax(decode_out.logits[:, -1, :], dim=-1, keepdim=True)
            
            # --- Stage 3: Score Calculation ---
            # **FIX**: Pass the pristine prompt-only cache to the scoring function
            raw_qk_scores = self._compute_raw_qk_scores(prompt_only_cache_tuple)
            self._token_importance_from_attn_scores(raw_qk_scores)

        finally:
            self._unpatch_model()
            self.captured_qs.clear()
        
        return self.rankings



# --- Main Execution Logic ---

def main():
    parser = argparse.ArgumentParser(description="Generate FastKV, GemFilter, and Speculative Prefill token rankings.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--datasets", nargs='+', required=True)
    parser.add_argument("--num_samples", type=int, default=15)
    parser.add_argument("--max_len", type=int, default=128000)
    parser.add_argument("--output_dir", type=str, default="analysis_results/approx_rankings")
    parser.add_argument("--window_size", type=int, default=8)
    parser.add_argument("--lookahead_k_values", type=int, nargs='+', default=[1, 2, 4, 8, 16, 32, 64, 128, 256])
    parser.add_argument("--kernel_size", type=int, default=13)
    parser.add_argument("--pooling", type=str, default="avgpool", choices=["avgpool", "maxpool", "none"])
    args = parser.parse_args()

    set_seed(42)
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, '..'))
        config_path = os.path.join(project_root, 'eval/longbench/config/dataset2prompt.json')
        with open(config_path, "r") as f:
            dataset2prompt = json.load(f)
    except FileNotFoundError:
        print(f"Warning: Could not find dataset2prompt.json at {config_path}. This might fail for some datasets.")
        dataset2prompt = {}

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa")

    generators = {
        'fastkv': FastKVRankingGenerator(model, tokenizer, args),
        'gemfilter': GemFilterRankingGenerator(model, tokenizer, args),
        'speculative': SpeculativeRankingGenerator(model, tokenizer, args)
    }
    print("Initialized all ranking generators.")

    model_name_sanitized = args.model.replace('/', '_')

    for dataset_name in args.datasets:
        print(f"\n--- Processing Dataset: {dataset_name.upper()} ---")
        if dataset_name not in dataset2prompt:
            print(f"Warning: No prompt format found for {dataset_name}. Skipping.")
            continue
            
        dataset_output_dir = os.path.join(args.output_dir, model_name_sanitized)
        os.makedirs(dataset_output_dir, exist_ok=True)
        output_path = os.path.join(dataset_output_dir, f"{dataset_name}.npz")

        try:
            # Added more specific exceptions for robustness
            if os.path.exists(output_path):
                existing_data = dict(np.load(output_path, allow_pickle=True))
            else:
                existing_data = {}
        except (FileNotFoundError, EOFError, zipfile.BadZipFile):
            print(f"Warning: Could not load existing data from {output_path}. Starting fresh for this dataset.")
            existing_data = {}

        data = load_dataset('THUDM/LongBench', dataset_name, split='test', trust_remote_code=True)
        processed_count = 0

        for i, sample in enumerate(data):
            if processed_count >= args.num_samples: break
            sample_key = f'sample_{i}'
            # Check if this specific sample and all its ranking types are already done.
            if sample_key in existing_data and all(f'{name}_rankings' in existing_data[sample_key].item() for name in generators):
                 print(f"Skipping already fully processed sample {i}.")
                 processed_count += 1
                 continue

            try:
                raw_prompt = dataset2prompt[dataset_name].format(**sample)
            except KeyError: continue

            datasets_without_chat_template = ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]
            if dataset_name not in datasets_without_chat_template:
                messages = [{"role": "user", "content": raw_prompt}]
                final_prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                final_prompt_text = raw_prompt

            inputs = tokenizer(final_prompt_text, return_tensors="pt").to(model.device)
            prompt_len = inputs.input_ids.shape[1]

            if prompt_len > args.max_len or prompt_len <= args.window_size: continue

            print(f"Processing sample index {i} ({processed_count + 1}/{args.num_samples}) for {dataset_name} ({prompt_len} tokens)...")
            
            all_rankings = {}
            for name, gen in generators.items():
                try:
                    all_rankings[name] = gen.generate_rankings(inputs)
                except Exception as e:
                    print(f"  -> ERROR generating ranking for '{name}' on sample {i}: {e}")
                    all_rankings[name] = {}

            if not any(all_rankings.values()):
                print(f"  -> Warning: Failed to generate any ranking types for sample index {i}. Skipping.")
                continue

            processed_count += 1
            to_numpy = lambda d: {k: v.cpu().to(torch.float16).numpy() for k, v in d.items()} if d else {}
            
            sample_data = {'input_ids': inputs.input_ids.squeeze(0).cpu().numpy()}
            for name, ranks in all_rankings.items():
                if ranks:
                    sample_data[f'{name}_rankings'] = pickle.dumps(to_numpy(ranks))

            existing_data[sample_key] = np.array(sample_data, dtype=object)
            
            try:
                np.savez_compressed(output_path, **existing_data)
            except Exception as e:
                print(f"  -> ERROR saving data to {output_path}: {e}")

            if torch.cuda.is_available(): torch.cuda.empty_cache()

        print(f"\nSaved/Updated ranking results for {dataset_name} to {output_path}")


if __name__ == "__main__":
    main()
