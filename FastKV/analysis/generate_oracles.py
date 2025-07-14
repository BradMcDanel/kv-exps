# analysis/generate_oracles.py

import os
import sys
import argparse
import pickle
import json
from typing import List, Dict, Tuple, Optional, Any

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, AutoConfig
from transformers.cache_utils import Cache, DynamicCache
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    apply_rotary_pos_emb as hf_apply_rotary_pos_emb,
    repeat_kv as hf_repeat_kv,
)
import math
import types
from functools import partial

def _patched_attention_forward_oracle(
    self_attn: LlamaAttention,
    oracle_generator: 'OracleGenerator',
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
    if not oracle_generator.is_prefilling and query_length == 1:
        oracle_generator.captured_qs[self_attn.layer_idx].append(query_states_rotated.detach().clone())
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

class OracleGenerator:
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.config: AutoConfig = self.model.config
        self.device = self.model.device
        self.dtype = self.model.dtype
        self.eos_token_ids = self._extract_eos_token_ids()
        self.captured_qs: List[List[torch.Tensor]] = []
        self.orig_fwds: Dict[int, Any] = {}
        self.is_prefilling = False
        self.token_importance_scores: Optional[torch.Tensor] = None

    def _extract_eos_token_ids(self) -> List[int]:
        config_eos = self.config.eos_token_id
        if isinstance(config_eos, int): return [config_eos]
        if isinstance(config_eos, list): return list(config_eos)
        if self.tokenizer.eos_token_id: return [self.tokenizer.eos_token_id]
        raise ValueError("eos_token_id not defined in config or tokenizer.")

    def _patch_model(self) -> int:
        if not hasattr(self.model, 'model') or not hasattr(self.model.model, 'layers'): return 0
        num_layers = self.config.num_hidden_layers
        self.captured_qs = [[] for _ in range(num_layers)]
        num_patched_layers = 0
        for i, layer in enumerate(self.model.model.layers):
            if hasattr(layer, 'self_attn') and isinstance(layer.self_attn, LlamaAttention):
                attn_module = layer.self_attn
                if i not in self.orig_fwds: self.orig_fwds[i] = attn_module.forward
                attn_module.forward = types.MethodType(partial(_patched_attention_forward_oracle, oracle_generator=self), attn_module)
                num_patched_layers += 1
        return num_patched_layers

    def _unpatch_model(self):
        for i, layer in enumerate(self.model.model.layers):
            if hasattr(layer, 'self_attn') and i in self.orig_fwds:
                layer.self_attn.forward = self.orig_fwds[i]
        self.orig_fwds.clear()

    def _token_importance_from_attn_scores(self, attention_scores: torch.Tensor):
        if attention_scores.numel() == 0:
            self.token_importance_scores = None
            return
        bs, num_layers, num_heads, num_gen_tokens, prompt_len = attention_scores.shape
        if bs != 1: raise NotImplementedError("Batch size > 1 is not supported.")
        attention_scores = attention_scores.squeeze(0)
        total_max_scores = torch.zeros(prompt_len, device=self.device, dtype=torch.float32)
        for i in range(num_gen_tokens):
            gen_token_scores = attention_scores[:, :, i, :]
            attention_probs = F.softmax(gen_token_scores, dim=-1, dtype=torch.float32)
            flattened_probs = attention_probs.flatten(0, 1)
            max_scores_for_token = flattened_probs.max(0).values
            total_max_scores += max_scores_for_token
        if num_gen_tokens > 0:
            final_token_importance = total_max_scores / num_gen_tokens
        else:
            final_token_importance = total_max_scores
        self.token_importance_scores = final_token_importance.unsqueeze(0).to(self.dtype)

    def _compute_raw_qk_scores(self, prompt_only_cache_tuple: Tuple[Tuple[torch.Tensor, torch.Tensor], ...]) -> torch.Tensor:
        if not self.captured_qs or not any(self.captured_qs): return torch.empty(0)
        num_layers = self.config.num_hidden_layers
        num_q_heads = self.config.num_attention_heads
        num_kv_heads = self.config.num_key_value_heads
        num_kv_groups = num_q_heads // num_kv_heads
        head_dim = self.config.hidden_size // num_q_heads
        all_layer_scores = []
        for layer_idx in range(num_layers):
            if not self.captured_qs[layer_idx] or prompt_only_cache_tuple[layer_idx][0].numel() == 0: continue
            key_prompt_layer = prompt_only_cache_tuple[layer_idx][0].detach()
            key_prompt_layer_repeated = hf_repeat_kv(key_prompt_layer, num_kv_groups)
            all_q_for_layer = torch.cat(self.captured_qs[layer_idx], dim=2)
            attn_logits = torch.matmul(all_q_for_layer, key_prompt_layer_repeated.transpose(-1, -2)) / math.sqrt(head_dim)
            all_layer_scores.append(attn_logits)
        if not all_layer_scores: return torch.empty(0)
        return torch.stack(all_layer_scores, dim=1)

    @torch.no_grad()
    def generate(self, inputs: Dict[str, torch.Tensor], max_gen: int) -> Optional[torch.Tensor]:
        self._patch_model()
        input_ids = inputs['input_ids']
        prompt_length = input_ids.shape[1]
        self.is_prefilling = True
        prefill_out = self.model(input_ids=input_ids, use_cache=True, cache_position=torch.arange(prompt_length, device=self.device))
        prefill_cache = prefill_out.past_key_values
        if isinstance(prefill_cache, tuple): prefill_cache = DynamicCache.from_legacy_cache(prefill_cache)
        prompt_only_cache_tuple = prefill_cache.to_legacy_cache()
        next_token = torch.argmax(prefill_out.logits[:, -1, :], dim=-1, keepdim=True)
        self.is_prefilling = False
        current_tokens = next_token
        current_cache = prefill_cache
        if next_token.item() not in self.eos_token_ids:
            for i in range(max_gen):
                cache_len = current_cache.get_seq_length(0)
                pos_ids = torch.tensor([[cache_len]], device=self.device)
                decode_cache_pos = torch.tensor([cache_len], device=self.device)
                decode_out = self.model(input_ids=current_tokens, position_ids=pos_ids, past_key_values=current_cache, use_cache=True, cache_position=decode_cache_pos)
                current_cache = decode_out.past_key_values
                if isinstance(current_cache, tuple): current_cache = DynamicCache.from_legacy_cache(current_cache)
                next_token = torch.argmax(decode_out.logits[:, -1, :], dim=-1, keepdim=True)
                current_tokens = next_token
                if next_token.item() in self.eos_token_ids: break
        raw_qk_scores = self._compute_raw_qk_scores(prompt_only_cache_tuple)
        self._token_importance_from_attn_scores(raw_qk_scores)
        self._unpatch_model()
        self.captured_qs.clear()
        return self.token_importance_scores

def build_chat(tokenizer: AutoTokenizer, prompt: str, model_name: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def main():
    parser = argparse.ArgumentParser(description="Generate Answer-Informed Oracle Rankings.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--datasets", nargs='+', required=True)
    parser.add_argument("--num_samples", type=int, default=15)
    parser.add_argument("--max_len", type=int, default=128000, help="Skip prompts longer than this.")
    parser.add_argument("--output_dir", type=str, default="analysis_results/oracles")
    args = parser.parse_args()
    
    set_seed(42)
    os.makedirs(args.output_dir, exist_ok=True)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    dataset2prompt_path = os.path.join(project_root, 'eval/longbench/config/dataset2prompt.json')
    dataset2maxlen_path = os.path.join(project_root, 'eval/longbench/config/dataset2maxlen.json')

    with open(dataset2prompt_path, "r") as f: dataset2prompt = json.load(f)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa")
    generator = OracleGenerator(model, tokenizer)
    
    all_oracles = {}
    for dataset_name in args.datasets:
        print(f"\n--- Processing Oracle for Dataset: {dataset_name.upper()} ---")
        if dataset_name not in dataset2prompt:
            print(f"Warning: No prompt format found for {dataset_name}. Skipping.")
            continue
        
        data = load_dataset('THUDM/LongBench', dataset_name, split='test', trust_remote_code=True)
        max_gen = 256
        dataset_oracles = []
        processed_count = 0

        for i, sample in enumerate(data):
            if processed_count >= args.num_samples: break
            
            try:
                raw_prompt = dataset2prompt[dataset_name].format(**sample)
            except KeyError:
                continue

            datasets_without_chat_template = ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]
            if dataset_name not in datasets_without_chat_template:
                final_prompt_text = build_chat(tokenizer, raw_prompt, args.model)
            else:
                final_prompt_text = raw_prompt

            inputs = tokenizer(final_prompt_text, return_tensors="pt").to(model.device)
            prompt_len = inputs.input_ids.shape[1]

            if prompt_len > args.max_len:
                continue

            print(f"Processing sample {i} ({processed_count + 1}/{args.num_samples}) for {dataset_name} ({prompt_len} tokens)...")
            
            oracle_ranking_tensor = generator.generate(inputs, max_gen)
            
            if oracle_ranking_tensor is not None and oracle_ranking_tensor.numel() > 0:
                assert oracle_ranking_tensor.shape[1] == prompt_len, \
                    f"Sample {i}: Mismatch! Oracle ranking len {oracle_ranking_tensor.shape[1]} vs Input len {prompt_len}"

                dataset_oracles.append({
                    'sample_idx': i,
                    'ranking': oracle_ranking_tensor.squeeze(0).cpu().tolist(),
                    'input_ids': inputs.input_ids.squeeze(0).cpu().tolist(),
                })
                processed_count += 1
            else:
                print(f"  -> Warning: Failed to generate a valid oracle ranking for sample {i}.")

            if torch.cuda.is_available(): torch.cuda.empty_cache()
            
        all_oracles[dataset_name] = dataset_oracles

    model_name_sanitized = args.model.replace('/', '_')
    output_filename = f"oracles_model_{model_name_sanitized}.pkl"
    output_path = os.path.join(args.output_dir, output_filename)

    print(f"\nSaving oracle results to {output_path}")
    with open(output_path, "wb") as f:
        pickle.dump(all_oracles, f)
    
if __name__ == "__main__":
    main()
