# analysis/degradation_analyzer_lookahead.py

import os
import sys
import argparse
import json
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from datasets import load_dataset
from typing import List, Tuple, Any, Dict
from transformers.cache_utils import Cache, DynamicCache
import types
from functools import partial
import math

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Import known-good components ---
from transformers.models.llama.modeling_llama import LlamaAttention, repeat_kv, apply_rotary_pos_emb
from eval.longbench.evaluate import dataset2metric
from eval.longbench.main import build_chat
from baseline.fastkv.monkeypatch import replace_llama as replace_llama_fastkv
from baseline.fastkv.fastkv_utils import compress

def _minimal_sdpa_spy_forward(
    self_attn: LlamaAttention,
    profiler_instance: 'LookaheadProfiler',
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor = None,
    position_ids: torch.LongTensor = None,
    past_key_value: Cache = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: torch.LongTensor = None,
    **kwargs,
) -> Tuple[torch.Tensor, Any, Any]:
    """
    A minimal patch that spies on Q-vectors and then calls the original forward method.
    This is designed to be compatible with both eager and SDPA implementations.
    """
    # --- SPYING LOGIC ---
    if profiler_instance.is_capturing:
        # Perform the minimum calculations needed to get the query vector
        bsz, q_len, _ = hidden_states.size()
        query_states = self_attn.q_proj(hidden_states)
        query_states = query_states.view(bsz, q_len, self_attn.num_heads, self_attn.head_dim).transpose(1, 2)
        
        # Calculate cos and sin based on the current device and position
        kv_seq_len = q_len
        if past_key_value is not None:
             kv_seq_len += past_key_value.get_seq_length(self_attn.layer_idx)
        cos, sin = self_attn.rotary_emb(hidden_states, seq_len=kv_seq_len)
        
        # Apply RoPE to get the final Q-vector for capture
        query_states_rotated, _ = apply_rotary_pos_emb(query_states, torch.zeros_like(query_states), cos, sin, position_ids)
        profiler_instance.captured_qs[self_attn.layer_idx].append(query_states_rotated.detach().clone())

    # --- EXECUTION LOGIC ---
    # Call the original, un-patched method we saved during initialization.
    # This ensures the model executes exactly as intended by the library,
    # whether it's in eager or SDPA mode.
    return profiler_instance.orig_fwds[self_attn.layer_idx](
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        cache_position=cache_position,
    )

class LookaheadProfiler:
    def __init__(self, model_name, tokenizer):
        print(f"Loading profiler model: {model_name} with Eager attention for robust patching")
        # --- KEY CHANGE: We will now use 'eager' again. The new patch is designed for it.
        # This gives us access to the non-optimized, inspectable components.
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto",
            attn_implementation="eager"
        ).eval()
        self.tokenizer = tokenizer
        self.config = self.model.config
        self.device = self.model.device
        self.captured_qs = []
        self.orig_fwds = {}
        self.is_capturing = False
        self._patch_model_for_q_capture()

    def _patch_model_for_q_capture(self):
        """Saves the original forward and replaces it with our spy."""
        for i, layer in enumerate(self.model.model.layers):
            if hasattr(layer, 'self_attn'):
                # Save the original method
                self.orig_fwds[i] = layer.self_attn.forward
                # Apply the patch
                layer.self_attn.forward = types.MethodType(
                    partial(_minimal_sdpa_spy_forward, profiler_instance=self), layer.self_attn
                )
        print(f"Profiler: Patched {len(self.orig_fwds)} attention layers.")

    # The rest of the LookaheadProfiler class remains the same...
    @torch.no_grad()
    def calculate_lookahead_entropy(self, prompt_text: str, max_len: int, k: int) -> float:
        inputs = self.tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=max_len).to(self.device)
        prompt_length = inputs.input_ids.shape[1]

        # Use a try-finally block to ensure the patch is always restored
        try:
            # Prefill
            prefill_out = self.model(**inputs, use_cache=True)
            prefill_cache = prefill_out.past_key_values
            
            if isinstance(prefill_cache, tuple):
                prefill_cache_tuple = prefill_cache
                current_cache = DynamicCache.from_legacy_cache(prefill_cache)
            else:
                prefill_cache_tuple = prefill_cache.to_legacy_cache()
                current_cache = prefill_cache

            # Lookahead
            [q_list.clear() for q_list in self.captured_qs]
            self.is_capturing = True
            
            current_tokens = torch.argmax(prefill_out.logits[:, -1, :], dim=-1, keepdim=True)
            
            for i in range(k):
                cache_len = current_cache.get_seq_length(0)
                pos_ids = torch.tensor([[cache_len]], device=self.device)
                lookahead_out = self.model(
                    input_ids=current_tokens, past_key_values=current_cache, use_cache=True, position_ids=pos_ids
                )
                current_cache = lookahead_out.past_key_values
                if isinstance(current_cache, tuple):
                    current_cache = DynamicCache.from_legacy_cache(current_cache)
                current_tokens = torch.argmax(lookahead_out.logits[:, -1, :], dim=-1, keepdim=True)
                if self.tokenizer.eos_token_id and current_tokens.item() == self.tokenizer.eos_token_id: break
            
            self.is_capturing = False

            # Compute
            if not any(self.captured_qs): return 0.0
            all_layer_scores = self._compute_raw_qk_scores(prefill_cache_tuple)
            bs, num_layers, num_heads, num_steps, key_len = all_layer_scores.shape
            attn_probs = F.softmax(all_layer_scores.squeeze(0), dim=-1, dtype=torch.float32)
            importance_scores = attn_probs.view(num_layers * num_heads, num_steps, key_len).max(dim=0).values.mean(dim=0)
            
            scores_np = importance_scores.cpu().numpy()
            scores_sum = scores_np.sum()
            if scores_sum == 0: return 0.0
            probs = scores_np / scores_sum
            return -np.sum(probs * np.log2(probs + 1e-9))
        
        finally:
            # This is not strictly necessary if the object is destroyed, but it's good practice
            self.is_capturing = False


    def _compute_raw_qk_scores(self, prefill_cache_tuple: tuple) -> torch.Tensor:
        num_layers = self.config.num_hidden_layers
        num_q_heads = self.config.num_attention_heads
        num_kv_heads = self.config.num_key_value_heads
        num_kv_groups = num_q_heads // num_kv_heads
        
        all_logits = []
        for layer_idx in range(num_layers):
            if not self.captured_qs[layer_idx] or prefill_cache_tuple[layer_idx][0].numel() == 0: continue
            key_states_repeated = repeat_kv(prefill_cache_tuple[layer_idx][0].detach(), num_kv_groups)
            query_states = torch.cat(self.captured_qs[layer_idx], dim=2)
            attn_logits = torch.matmul(query_states, key_states_repeated.transpose(-1, -2)) / math.sqrt(query_states.shape[-1])
            all_logits.append(attn_logits)
        return torch.stack(all_logits, dim=1)

# The rest of the script (DegradationAnalyzer, main) is unchanged.
# ...
class DegradationAnalyzer:
    def __init__(self, args):
        self.args = args
        set_seed(args.seed)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    def run_analysis(self):
        # --- PHASE 1: Calculate all entropies first ---
        print("\n--- PHASE 1: Calculating Lookahead-Informed Entropies ---")
        profiler = LookaheadProfiler(self.args.model_name, self.tokenizer)
        
        dataset = load_dataset('THUDM/LongBench', self.args.dataset, split='test', trust_remote_code=True)
        num_samples = min(self.args.num_samples, len(dataset))
        samples_to_process = dataset.select(range(num_samples))
        
        prompt_format = json.load(open("eval/longbench/config/dataset2prompt.json", "r"))[self.args.dataset]
        model2maxlen = json.load(open("eval/longbench/config/model2maxlen.json", "r"))
        max_len = model2maxlen.get(self.args.model_name, self.args.max_prompt_len)

        entropy_results = {}
        for i, sample in enumerate(tqdm(samples_to_process, desc="Profiling samples")):
            prompt_text = prompt_format.format(**sample)
            tokenized_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
            if len(tokenized_ids) > max_len:
                half = int(max_len / 2)
                prompt_text = self.tokenizer.decode(tokenized_ids[:half]) + self.tokenizer.decode(tokenized_ids[-half:])
            
            entropy = profiler.calculate_lookahead_entropy(prompt_text, max_len, self.args.lookahead_k)
            entropy_results[i] = {"entropy": entropy, "prompt_text": prompt_text}
        
        del profiler
        torch.cuda.empty_cache()
        print("--- PHASE 1 COMPLETE ---\n")

        # --- PHASE 2: Evaluate performance with FastKV ---
        print("\n--- PHASE 2: Evaluating Performance Degradation with FastKV ---")
        if "llama" in self.args.model_name.lower():
            replace_llama_fastkv()
        
        fastkv_model = AutoModelForCausalLM.from_pretrained(
            self.args.model_name, device_map='auto',
            attn_implementation='flash_attention_2', torch_dtype=torch.float16
        ).eval()

        all_results_data = []
        for i, sample in enumerate(tqdm(samples_to_process, desc="Evaluating FastKV performance")):
            if i not in entropy_results: continue
            
            sample_data = {"sample_id": i, "entropy": entropy_results[i]["entropy"]}
            prompt_text = entropy_results[i]["prompt_text"]

            for budget in tqdm(self.args.budgets, desc=f"  Testing Budgets for sample {i}", leave=False):
                fastkv_args = argparse.Namespace(max_capacity_prompt=budget, window_size=8, kernel_size=7, pooling='avgpool', tsp_idx=15, tsp_len=2048)
                compress(fastkv_model, fastkv_args)
                
                final_prompt = build_chat(self.tokenizer, prompt_text, self.args.model_name)
                inputs = self.tokenizer(final_prompt, truncation=False, return_tensors="pt").to(fastkv_model.device)
                
                with torch.no_grad():
                    output_ids = fastkv_model.generate(**inputs, max_new_tokens=self.args.max_gen_len, num_beams=1, do_sample=False)[0]
                prediction = self.tokenizer.decode(output_ids[inputs.input_ids.shape[1]:], skip_special_tokens=True)
                
                score_fn = dataset2metric.get(self.args.dataset)
                if not score_fn: continue
                
                max_score = max(score_fn(prediction, gt, all_classes=sample.get('all_classes')) for gt in sample['answers'])
                sample_data[f'score_at_{budget}'] = max_score
            
            all_results_data.append(sample_data)

        if all_results_data:
            df = pd.DataFrame(all_results_data)
            output_path = os.path.join(self.args.output_dir, f"degradation_data_lookahead_{self.args.dataset}.csv")
            df.to_csv(output_path, index=False)
            print(f"\nAnalysis complete. Results saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Analyze degradation vs. lookahead-informed entropy.")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--dataset", type=str, default="qasper")
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--lookahead_k", type=int, default=8)
    parser.add_argument("--budgets", nargs='+', type=int, default=[4096, 2048, 1024, 512, 256, 128])
    parser.add_argument("--max_prompt_len", type=int, default=4096)
    parser.add_argument("--max_gen_len", type=int, default=128)
    parser.add_argument("--output_dir", type=str, default="analysis_results")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    analyzer = DegradationAnalyzer(args)
    analyzer.run_analysis()

if __name__ == "__main__":
    main()
