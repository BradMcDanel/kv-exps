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
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    apply_rotary_pos_emb as hf_apply_rotary_pos_emb,
    repeat_kv as hf_repeat_kv,
)
import math
import types
from functools import partial
import numpy as np

def _patched_attention_forward_unified(
    self_attn: LlamaAttention,
    ranking_generator: "ApproxRankingGenerator",
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
    bsz, q_len, _ = hidden_states.size()
    if q_len <= ranking_generator.window_size:
        return ranking_generator.orig_fwds[self_attn.layer_idx](
            hidden_states, attention_mask, position_ids, past_key_value, 
            output_attentions, use_cache, cache_position, **kwargs
        )
    
    num_heads = self_attn.config.num_attention_heads
    head_dim = self_attn.config.hidden_size // num_heads
    num_kv_heads = self_attn.config.num_key_value_heads
    num_kv_groups = num_heads // num_kv_heads

    q_proj = self_attn.q_proj(hidden_states)
    k_proj = self_attn.k_proj(hidden_states)
    v_proj = self_attn.v_proj(hidden_states)
    
    query_states = q_proj.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
    key_states = k_proj.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
    value_states = v_proj.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
    
    cos, sin = self_attn.rotary_emb(value_states, position_ids)
    query_states, key_states = hf_apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
    
    with torch.no_grad():
        scoring_queries = query_states[..., -ranking_generator.window_size :, :].detach()
        key_states_for_scoring = hf_repeat_kv(key_states, num_kv_groups).detach()
        
        attn_logits = torch.matmul(scoring_queries, key_states_for_scoring.transpose(2, 3)) / math.sqrt(head_dim)
        
        ranking_generator.aggregate_ranking_logits[self_attn.layer_idx] = attn_logits.clone()
        
        attn_probs = F.softmax(attn_logits, dim=-1, dtype=torch.float32)

        attn_probs_for_scoring = attn_probs[..., :, :-ranking_generator.window_size]

        token_scores_per_head = attn_probs_for_scoring.sum(dim=-2)

        if ranking_generator.pooling != "none":
            if ranking_generator.pooling == "avgpool":
                pooled_scores = F.avg_pool1d(token_scores_per_head, kernel_size=ranking_generator.kernel_size, padding=ranking_generator.kernel_size // 2, stride=1)
            elif ranking_generator.pooling == "maxpool":
                pooled_scores = F.max_pool1d(token_scores_per_head, kernel_size=ranking_generator.kernel_size, padding=ranking_generator.kernel_size // 2, stride=1)
        else:
            pooled_scores = token_scores_per_head
        
        final_fastkv_scores = pooled_scores.sum(dim=1)
        
        padded_scores = torch.zeros(bsz, q_len, device=final_fastkv_scores.device, dtype=final_fastkv_scores.dtype)
        
        scorable_len = final_fastkv_scores.shape[1]
        padded_scores[:, :scorable_len] = final_fastkv_scores
        
        max_score = final_fastkv_scores.max(dim=-1, keepdim=True)[0]
        padded_scores[:, -ranking_generator.window_size:] = max_score + 1e-5

        ranking_generator.layerwise_rankings[self_attn.layer_idx] = padded_scores.clone()

    if past_key_value is not None:
         key_states, value_states = past_key_value.update(key_states, value_states, self_attn.layer_idx, {"cache_position": cache_position})
    
    key_states_repeated = hf_repeat_kv(key_states, num_kv_groups)
    value_states_repeated = hf_repeat_kv(value_states, num_kv_groups)
    
    attn_output = F.scaled_dot_product_attention(query_states, key_states_repeated, value_states_repeated, attn_mask=attention_mask, dropout_p=0.0, is_causal=(q_len > 1 and attention_mask is None))
    attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self_attn.config.hidden_size)
    attn_output = self_attn.o_proj(attn_output)
    
    return attn_output, None, past_key_value

class ApproxRankingGenerator:
    def __init__(self, model: AutoModelForCausalLM, window_size: int = 32, kernel_size: int = 7, pooling: str = 'avgpool'):
        self.model = model
        self.config: AutoConfig = self.model.config
        self.device = self.model.device
        self.window_size = window_size
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.layerwise_rankings: Dict[int, torch.Tensor] = {}
        self.aggregate_ranking_logits: Dict[int, torch.Tensor] = {}
        self.orig_fwds: Dict[int, Any] = {}

    def _patch_model(self):
        self.orig_fwds.clear()
        for i, layer in enumerate(self.model.model.layers):
            if hasattr(layer, 'self_attn') and isinstance(layer.self_attn, LlamaAttention):
                setattr(layer.self_attn, "layer_idx", i) # Ensure layer_idx is available
                self.orig_fwds[i] = layer.self_attn.forward
                layer.self_attn.forward = types.MethodType(partial(_patched_attention_forward_unified, ranking_generator=self), layer.self_attn)

    def _unpatch_model(self):
        for i, layer in enumerate(self.model.model.layers):
            if hasattr(layer, 'self_attn') and i in self.orig_fwds:
                layer.self_attn.forward = self.orig_fwds[i]
        self.orig_fwds.clear()

    def _compute_cumulative_sp_rankings(self) -> Dict[int, torch.Tensor]:
        if not self.aggregate_ranking_logits: return {}
        cumulative_rankings = {}
        sorted_layers = sorted(self.aggregate_ranking_logits.keys())
        for up_to_layer_idx in sorted_layers:
            current_layer_indices = [l for l in sorted_layers if l <= up_to_layer_idx]
            logits_to_process = torch.stack([self.aggregate_ranking_logits[i] for i in current_layer_indices])
            
            all_probs = F.softmax(logits_to_process, dim=-1, dtype=torch.float32)
            all_probs = all_probs.permute(1, 3, 0, 2, 4)
            flat_probs = all_probs.flatten(start_dim=2, end_dim=3)
            max_scores, _ = flat_probs.max(dim=2)
            final_scores = max_scores.mean(dim=1)
            
            scores_to_rank = final_scores[:, :-self.window_size]
            max_score = scores_to_rank.max(dim=-1, keepdim=True)[0]
            
            final_scores[:, -self.window_size:] = max_score + 1e-5
            
            cumulative_rankings[up_to_layer_idx] = final_scores.clone()
        return cumulative_rankings

    @torch.no_grad()
    def generate_rankings(self, inputs: Dict[str, torch.Tensor]) -> Tuple[Dict, Dict]:
        self.layerwise_rankings.clear()
        self.aggregate_ranking_logits.clear()
        
        try:
            self._patch_model()
            self.model(input_ids=inputs['input_ids'], use_cache=False, cache_position=torch.arange(inputs['input_ids'].shape[1], device=self.device))
        finally:
            self._unpatch_model()
            
        cumulative_sp_rankings = self._compute_cumulative_sp_rankings()
        return self.layerwise_rankings, cumulative_sp_rankings

def build_chat(tokenizer: AutoTokenizer, prompt: str, model_name: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def main():
    parser = argparse.ArgumentParser(description="Generate Layerwise and Cumulative (SP-style) token rankings.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--datasets", nargs='+', required=True)
    parser.add_argument("--num_samples", type=int, default=15)
    parser.add_argument("--max_len", type=int, default=128000, help="Skip prompts longer than this.")
    parser.add_argument("--output_dir", type=str, default="analysis_results/approx_rankings")
    parser.add_argument("--window_size", type=int, default=8, help="Number of recent tokens for perspective.")
    parser.add_argument("--kernel_size", type=int, default=7, help="Kernel size for pooling in layerwise ranking.")
    parser.add_argument("--pooling", type=str, default="avgpool", choices=["avgpool", "maxpool", "none"])
    args = parser.parse_args()
    
    set_seed(42)
    os.makedirs(args.output_dir, exist_ok=True)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    dataset2prompt_path = os.path.join(project_root, 'eval/longbench/config/dataset2prompt.json')
    with open(dataset2prompt_path, "r") as f: dataset2prompt = json.load(f)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa")
    ranking_generator = ApproxRankingGenerator(model, window_size=args.window_size, kernel_size=args.kernel_size, pooling=args.pooling)
    
    model_name_sanitized = args.model.replace('/', '_')

    for dataset_name in args.datasets:
        print(f"\n--- Processing Approx. Rankings for Dataset: {dataset_name.upper()} ---")
        if dataset_name not in dataset2prompt:
            print(f"Warning: No prompt format found for {dataset_name}. Skipping.")
            continue
        
        dataset_output_dir = os.path.join(args.output_dir, model_name_sanitized)
        os.makedirs(dataset_output_dir, exist_ok=True)
        output_path = os.path.join(dataset_output_dir, f"{dataset_name}.npz")
        
        try:
            existing_data = dict(np.load(output_path, allow_pickle=True))
            print(f"Loaded {len(existing_data)} existing samples from {output_path}. Will append new samples.")
        except FileNotFoundError:
            existing_data = {}

        data = load_dataset('THUDM/LongBench', dataset_name, split='test', trust_remote_code=True)
        processed_count = 0

        for i, sample in enumerate(data):
            if processed_count >= args.num_samples: break
            sample_key = f'sample_{i}'

            if sample_key in existing_data:
                print(f"Skipping already processed sample {i}.")
                processed_count += 1
                continue
            
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

            if prompt_len > args.max_len or prompt_len <= args.window_size:
                continue
            
            print(f"Processing sample {i} ({processed_count + 1}/{args.num_samples}) for {dataset_name} ({prompt_len} tokens)...")
            
            layerwise_rankings, cumulative_rankings = ranking_generator.generate_rankings(inputs)
            
            if not layerwise_rankings or not cumulative_rankings:
                print(f"  -> Warning: Failed to generate rankings for sample {i}. Skipping.")
                continue

            # Convert rankings to serializable numpy arrays
            layerwise_np = {l: s.squeeze(0).cpu().to(torch.float32).numpy() for l, s in layerwise_rankings.items()}
            cumulative_np = {l: s.squeeze(0).cpu().to(torch.float32).numpy() for l, s in cumulative_rankings.items()}
            
            sample_data = {
                'input_ids': inputs.input_ids.squeeze(0).cpu().numpy(),
                'layerwise_rankings': pickle.dumps(layerwise_np),
                'cumulative_rankings': pickle.dumps(cumulative_np),
            }
            existing_data[sample_key] = np.array(sample_data)

            # Save after each sample
            np.savez_compressed(output_path, **existing_data)
            processed_count += 1

            if torch.cuda.is_available(): torch.cuda.empty_cache()
            
        print(f"\nSaved/Updated approximate ranking results for {dataset_name} to {output_path}")

if __name__ == "__main__":
    main()
