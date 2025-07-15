# analysis/generate_approx_rankings.py

import argparse
import json
import math
import os
import pickle
import types
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
    """Generates rankings based on the FastKV attention score approximation."""

    def __init__(self, model, tokenizer, args):
        super().__init__(model, tokenizer, args)
        self.rankings: Dict[int, torch.Tensor] = {}

    def _patched_attention_forward(self, self_attn: LlamaAttention, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, position_ids: Optional[torch.LongTensor] = None, past_key_value: Optional[Cache] = None, use_cache: bool = False, **kwargs):
        bsz, q_len, _ = hidden_states.size()
        if q_len <= self.args.window_size:
            return self.orig_fwds[self_attn.layer_idx](hidden_states=hidden_states, attention_mask=attention_mask, position_ids=position_ids, past_key_value=past_key_value, use_cache=use_cache, **kwargs)

        cfg = self_attn.config
        num_heads, head_dim = cfg.num_attention_heads, cfg.hidden_size // cfg.num_attention_heads
        num_kv_heads, num_kv_groups = cfg.num_key_value_heads, cfg.num_attention_heads // cfg.num_key_value_heads

        query_states = self_attn.q_proj(hidden_states).view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
        key_states = self_attn.k_proj(hidden_states).view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
        value_states = self_attn.v_proj(hidden_states).view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
        cos, sin = self_attn.rotary_emb(value_states, position_ids)
        query_states, key_states = hf_apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        with torch.no_grad():
            scoring_queries = query_states[..., -self.args.window_size:, :].detach()
            key_states_for_scoring = hf_repeat_kv(key_states, num_kv_groups).detach()
            attn_logits = torch.matmul(scoring_queries, key_states_for_scoring.transpose(2, 3)) / math.sqrt(head_dim)
            attn_probs = F.softmax(attn_logits, dim=-1, dtype=torch.float32)
            token_scores = attn_probs[..., :, :-self.args.window_size].sum(dim=-2)

            if self.args.pooling != "none":
                pool_fn = F.avg_pool1d if self.args.pooling == "avgpool" else F.max_pool1d
                token_scores = pool_fn(token_scores, kernel_size=self.args.kernel_size, padding=self.args.kernel_size // 2, stride=1)

            final_scores = token_scores.sum(dim=1)
            padded_scores = torch.zeros(bsz, q_len, device=final_scores.device, dtype=final_scores.dtype)
            padded_scores[:, :-self.args.window_size] = final_scores
            padded_scores[:, -self.args.window_size:] = final_scores.max(dim=-1, keepdim=True)[0] + 1e-5
            self.rankings[self_attn.layer_idx] = padded_scores.clone()

        if use_cache:
            key_states, value_states = past_key_value.update(key_states, value_states, self_attn.layer_idx, kwargs.get("cache_position"))

        attn_output = F.scaled_dot_product_attention(query_states, hf_repeat_kv(key_states, num_kv_groups), hf_repeat_kv(value_states, num_kv_groups), attn_mask=attention_mask, dropout_p=0.0, is_causal=(use_cache is False and q_len > 1 and attention_mask is None))
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
        self.model(input_ids=inputs['input_ids'], use_cache=False)
        self._unpatch_model()
        return self.rankings


# --- Strategy 2: GemFilter ---

class GemFilterRankingGenerator(BaseRankingGenerator):
    """Generates rankings based on the GemFilter inner-product approximation."""

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
        cos, sin = self_attn.rotary_emb(value_states, position_ids)
        query_states, key_states = hf_apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        with torch.no_grad():
            query_last = query_states[:, :, -1:, :].detach()
            key_states_for_scoring = hf_repeat_kv(key_states, num_kv_groups).detach()
            inner_product = torch.matmul(query_last, key_states_for_scoring.transpose(-1, -2)).squeeze(2)
            token_scores = inner_product.sum(dim=1, keepdim=True)

            if self.args.pooling != "none":
                pool_fn = F.avg_pool1d if self.args.pooling == "avgpool" else F.max_pool1d
                token_scores = pool_fn(token_scores, kernel_size=self.args.kernel_size, padding=self.args.kernel_size // 2, stride=1)
            self.rankings[self_attn.layer_idx] = token_scores.squeeze(1).clone()

        if use_cache:
            key_states, value_states = past_key_value.update(key_states, value_states, self_attn.layer_idx, kwargs.get("cache_position"))

        attn_output = F.scaled_dot_product_attention(query_states, hf_repeat_kv(key_states, num_kv_groups), hf_repeat_kv(value_states, num_kv_groups), attn_mask=attention_mask, dropout_p=0.0, is_causal=(use_cache is False and q_len > 1 and attention_mask is None))
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
        self.model(input_ids=inputs['input_ids'], use_cache=False)
        self._unpatch_model()
        return self.rankings


# --- Strategy 3: Speculative ---

class SpeculativeRankingGenerator(BaseRankingGenerator):
    """Generates rankings based on speculative decoding lookahead queries."""

    def __init__(self, model, tokenizer, args):
        super().__init__(model, tokenizer, args)
        self.is_lookahead = False
        self.lookahead_qs: List[List[torch.Tensor]] = []
        self.eos_token_ids = self._extract_eos_token_ids()

    def _extract_eos_token_ids(self) -> List[int]:
        """Gets EOS token IDs from model config or tokenizer."""
        config_eos = self.config.eos_token_id
        if isinstance(config_eos, int):
            return [config_eos]
        if isinstance(config_eos, list):
            return list(config_eos)
        
        tokenizer_eos = self.tokenizer.eos_token_id
        if isinstance(tokenizer_eos, int):
            return [tokenizer_eos]
        if isinstance(tokenizer_eos, list):
            return list(tokenizer_eos)
            
        raise ValueError("Could not determine eos_token_id from model config or tokenizer.")

    def _patched_attention_forward(self, self_attn: LlamaAttention, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, position_ids: Optional[torch.LongTensor] = None, past_key_value: Optional[Cache] = None, use_cache: bool = False, **kwargs):
        bsz, q_len, _ = hidden_states.size()
        cfg = self_attn.config
        num_heads, head_dim = cfg.num_attention_heads, cfg.hidden_size // cfg.num_attention_heads
        num_kv_heads, num_kv_groups = cfg.num_key_value_heads, cfg.num_attention_heads // cfg.num_key_value_heads

        query_states = self_attn.q_proj(hidden_states).view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
        key_states = self_attn.k_proj(hidden_states).view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
        value_states = self_attn.v_proj(hidden_states).view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
        cos, sin = self_attn.rotary_emb(value_states, position_ids)
        query_states_rot, key_states_rot = hf_apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if self.is_lookahead and q_len == 1:
            self.lookahead_qs[self_attn.layer_idx].append(query_states_rot.detach().clone())

        if use_cache:
            key_states_rot, value_states = past_key_value.update(key_states_rot, value_states, self_attn.layer_idx, kwargs.get("cache_position"))

        attn_output = F.scaled_dot_product_attention(query_states_rot, hf_repeat_kv(key_states_rot, num_kv_groups), hf_repeat_kv(value_states, num_kv_groups), attn_mask=attention_mask, dropout_p=0.0, is_causal=(use_cache is False and q_len > 1 and attention_mask is None))
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, cfg.hidden_size)
        return self_attn.o_proj(attn_output), None, past_key_value

    def _patch_model(self):
        self.lookahead_qs = [[] for _ in range(self.config.num_hidden_layers)]
        for i, layer in enumerate(self.model.model.layers):
            if hasattr(layer, 'self_attn') and isinstance(layer.self_attn, LlamaAttention):
                setattr(layer.self_attn, "layer_idx", i)
                self.orig_fwds[i] = layer.self_attn.forward
                layer.self_attn.forward = types.MethodType(self._patched_attention_forward, layer.self_attn)

    def _compute_importance(self, legacy_cache: tuple) -> torch.Tensor:
        cfg = self.config
        num_layers, num_q_heads, num_kv_heads, head_dim = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.hidden_size // cfg.num_attention_heads
        num_kv_groups = num_q_heads // num_kv_heads

        all_scores = [
            torch.matmul(torch.cat(self.lookahead_qs[i], dim=2), hf_repeat_kv(legacy_cache[i][0], num_kv_groups).transpose(-1, -2)) / math.sqrt(head_dim)
            for i in range(num_layers) if self.lookahead_qs[i]
        ]
        if not all_scores:
            return torch.empty(0)

        stacked_scores = torch.stack(all_scores, dim=0).permute(1, 3, 0, 2, 4)
        if self.args.pooling != 'none':
            bs, k, layers, heads, p_len = stacked_scores.shape
            probs = F.softmax(stacked_scores, dim=-1, dtype=torch.float32).flatten(0, 3)
            pool_fn = F.avg_pool1d if self.args.pooling == "avgpool" else F.max_pool1d
            pooled_probs = pool_fn(probs.unsqueeze(1), kernel_size=self.args.kernel_size, padding=self.args.kernel_size // 2, stride=1).squeeze(1)
            stacked_scores = pooled_probs.reshape(bs, k, layers, heads, p_len)
        else:
            stacked_scores = F.softmax(stacked_scores, dim=-1, dtype=torch.float32)

        return stacked_scores.flatten(2, 3).max(2).values.mean(1)

    @torch.no_grad()
    def generate_rankings(self, inputs: Dict[str, torch.Tensor]) -> Dict:
        """
        **MODIFICATION: Overhauled the generation loop to respect EOS tokens
        and correctly use `cache_position` for SDPA compatibility.**
        """
        input_ids = inputs['input_ids']
        prompt_len = input_ids.shape[1]
        rankings_by_k = {}
        self._patch_model()

        try:
            # --- Prefill Stage (Corrected) ---
            # Pass cache_position for SDPA prefill.
            prefill_out = self.model(
                input_ids=input_ids,
                use_cache=True,
                cache_position=torch.arange(prompt_len, device=self.device)
            )
            prompt_kv_cache_tuple = prefill_out.past_key_values
            # It's good practice to wrap the cache early.
            current_kv_cache = DynamicCache.from_legacy_cache(prompt_kv_cache_tuple)
            current_tokens = torch.argmax(prefill_out.logits[:, -1, :], dim=-1, keepdim=True)

            # --- Lookahead Stage (Now respects EOS and SDPA) ---
            self.is_lookahead = True
            max_k = max(self.args.lookahead_k_values)
            
            for k in range(1, max_k + 1):
                # Stop generation if the previously generated token was EOS.
                # This check happens at the top of the loop.
                if current_tokens.item() in self.eos_token_ids:
                    # Since we stopped, the last valid ranking applies to all future k's.
                    # The score computation must happen *outside* the loop after it breaks.
                    break

                # --- Decode Step (Corrected) ---
                cache_len = current_kv_cache.get_seq_length(0)
                pos_ids = torch.tensor([[cache_len]], device=self.device)
                decode_cache_pos = torch.tensor([cache_len], device=self.device)

                decode_out = self.model(
                    input_ids=current_tokens,
                    position_ids=pos_ids,
                    past_key_values=current_kv_cache,
                    use_cache=True,
                    cache_position=decode_cache_pos,
                )
                
                # Update state for the next iteration
                current_tokens = torch.argmax(decode_out.logits[:, -1, :], dim=-1, keepdim=True)
                current_kv_cache = decode_out.past_key_values # Already a DynamicCache
                
                # If this k is one we need to record, compute scores based on queries so far.
                if k in self.args.lookahead_k_values:
                    scores = self._compute_importance(prompt_kv_cache_tuple)
                    if scores.numel() > 0:
                        rankings_by_k[k] = scores.clone()
            
            # --- Post-Loop Score Finalization for EOS ---
            # If the loop broke early due to EOS, we need to compute the final ranking
            # and apply it to all remaining k-values that were not yet computed.
            final_scores = self._compute_importance(prompt_kv_cache_tuple)
            if final_scores.numel() > 0:
                for k_val in self.args.lookahead_k_values:
                    if k_val not in rankings_by_k:
                        rankings_by_k[k_val] = final_scores.clone()

        finally:
            self._unpatch_model()
            self.lookahead_qs.clear() # Good practice to clear captured data

        return rankings_by_k

# --- Main Execution Logic ---

def main():
    """Main execution function to generate and save approximate rankings."""
    parser = argparse.ArgumentParser(description="Generate FastKV, GemFilter, and Speculative Prefill token rankings.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--datasets", nargs='+', required=True)
    parser.add_argument("--num_samples", type=int, default=15)
    parser.add_argument("--max_len", type=int, default=128000)
    parser.add_argument("--output_dir", type=str, default="analysis_results/approx_rankings")
    parser.add_argument("--window_size", type=int, default=8)
    parser.add_argument("--lookahead_k_values", type=int, nargs='+', default=[1, 2, 4, 8, 16, 32, 64, 128, 256])
    parser.add_argument("--kernel_size", type=int, default=7)
    parser.add_argument("--pooling", type=str, default="avgpool", choices=["avgpool", "maxpool", "none"])
    args = parser.parse_args()

    set_seed(42)
    os.makedirs(args.output_dir, exist_ok=True)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    with open(os.path.join(project_root, 'eval/longbench/config/dataset2prompt.json'), "r") as f:
        dataset2prompt = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
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
            existing_data = dict(np.load(output_path, allow_pickle=True))
        except FileNotFoundError:
            existing_data = {}

        data = load_dataset('THUDM/LongBench', dataset_name, split='test', trust_remote_code=True)
        processed_count = 0

        for i, sample in enumerate(data):
            if processed_count >= args.num_samples:
                break

            sample_key = f'sample_{i}'
            if sample_key in existing_data:
                continue

            try:
                raw_prompt = dataset2prompt[dataset_name].format(**sample)
            except KeyError:
                continue

            # Handle special datasets that should not use a chat template
            datasets_without_chat_template = ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]
            if dataset_name not in datasets_without_chat_template:
                messages = [{"role": "user", "content": raw_prompt}]
                final_prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                final_prompt_text = raw_prompt

            inputs = tokenizer(final_prompt_text, return_tensors="pt").to(model.device)
            prompt_len = inputs.input_ids.shape[1]

            if prompt_len > args.max_len:
                continue

            if prompt_len <= args.window_size:
                continue

            print(f"Processing sample index {i} ({processed_count + 1}/{args.num_samples}) for {dataset_name} ({prompt_len} tokens)...")
            all_rankings = {name: gen.generate_rankings(inputs) for name, gen in generators.items()}

            if not all(all_rankings.values()):
                print(f"  -> Warning: Failed to generate one or more ranking types for sample index {i}. Skipping.")
                continue

            processed_count += 1
            to_numpy = lambda d: {k: v.squeeze(0).cpu().to(torch.float32).numpy() for k, v in d.items()}
            sample_data = {'input_ids': inputs.input_ids.squeeze(0).cpu().numpy()}
            for name, ranks in all_rankings.items():
                sample_data[f'{name}_rankings'] = pickle.dumps(to_numpy(ranks))

            existing_data[sample_key] = np.array(sample_data, dtype=object)
            np.savez_compressed(output_path, **existing_data)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print(f"\nSaved/Updated ranking results for {dataset_name} to {output_path}")


if __name__ == "__main__":
    main()
