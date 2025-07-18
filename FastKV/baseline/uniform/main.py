# baseline/uniform/main.py
import os
import time
import argparse
import json
from typing import List, Dict, Tuple, Optional, Any

import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.cache_utils import Cache, DynamicCache

class UniformRandomPipeline:
    """
    Implements a baseline that performs selective prefill by anchoring the
    first K and last K tokens, and then uniformly sampling the rest of the
    tokens from the middle of the prompt to meet a total budget.
    """
    def __init__(
        self,
        base_model_name: str,
        tokenizer: AutoTokenizer,
        keep_percentage: float = 0.1,
        first_k: int = 64,
        last_k: int = 256,
        detailed_timing: bool = True,
        tsp_idx: Optional[int] = None,
    ):
        self.base_model_name = base_model_name
        self.tokenizer = tokenizer
        self.base_model = self._load_model(self.base_model_name)
        self.base_config: AutoConfig = self.base_model.config
        self.device = self.base_model.device
        self.dtype = self.base_model.dtype
        
        self.keep_percentage = keep_percentage
        self.first_k = first_k
        self.last_k = last_k
        self.detailed_timing = detailed_timing
        self.tsp_idx = tsp_idx
        
        self._validate_config()
        self.eos_token_ids = self._extract_eos_token_ids()

    def _validate_config(self):
        if not (0.0 < self.keep_percentage <= 1.0):
            raise ValueError("`keep_percentage` must be between 0.0 and 1.0.")
        if self.first_k < 0 or self.last_k < 0:
            raise ValueError("`first_k` and `last_k` must be non-negative.")

    def _load_model(self, model_name: str) -> AutoModelForCausalLM:
        print(f"Loading base model: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto", attn_implementation="sdpa"
        )
        return model.eval()

    def _extract_eos_token_ids(self) -> List[int]:
        config_eos = self.base_config.eos_token_id
        if isinstance(config_eos, int): return [config_eos]
        if isinstance(config_eos, list): return list(config_eos)
        if self.tokenizer.eos_token_id: return [self.tokenizer.eos_token_id]
        raise ValueError("eos_token_id not defined in config or tokenizer.")

    @torch.no_grad()
    def run(
        self,
        input_ids: torch.Tensor,
        max_generation_length: int,
        **kwargs # Accept other kwargs to match signature, but ignore them
    ) -> Tuple[str, Dict[str, Any]]:
        run_metadata: Dict[str, Any] = {}
        
        if self.detailed_timing:
            start_time = time.time()
            prefill_start_event = torch.cuda.Event(enable_timing=True)
            prefill_end_event = torch.cuda.Event(enable_timing=True)

        prompt_length = input_ids.shape[1]
        num_to_keep = int(prompt_length * self.keep_percentage)

        # --- Stage 1: Select Tokens with Anchoring and Uniform Sampling ---
        # 1. Define anchor indices (first K and last K)
        first_indices = torch.arange(min(self.first_k, prompt_length), device=self.device)
        last_indices = torch.arange(max(0, prompt_length - self.last_k), prompt_length, device=self.device)
        anchor_indices = torch.unique(torch.cat([first_indices, last_indices]))

        # 2. Define the pool of middle indices available for sampling
        all_indices_set = set(range(prompt_length))
        anchor_indices_set = set(anchor_indices.tolist())
        middle_pool_indices = torch.tensor(list(all_indices_set - anchor_indices_set), device=self.device)
        
        # 3. Calculate how many more tokens we need to sample
        num_to_sample = max(0, num_to_keep - len(anchor_indices))

        # 4. Uniformly sample from the middle pool
        if num_to_sample > 0 and len(middle_pool_indices) > 0:
            num_to_sample = min(num_to_sample, len(middle_pool_indices))
            perm = torch.randperm(len(middle_pool_indices), device=self.device)
            sampled_middle_indices = middle_pool_indices[perm[:num_to_sample]]
            indices_to_keep = torch.sort(torch.cat([anchor_indices, sampled_middle_indices]))[0]
        else:
            # If no sampling is needed or possible, just use the anchors (truncated to budget if needed)
            indices_to_keep = anchor_indices[:num_to_keep]

        selected_prompt_ids = input_ids[:, indices_to_keep]
        selective_pos_ids = indices_to_keep.unsqueeze(0).to(torch.long)
        
        run_metadata["prompt_input_length"] = prompt_length
        run_metadata["selective_prefill_kept_token_count"] = selected_prompt_ids.shape[1]
        run_metadata["token_keep_rate"] = (selected_prompt_ids.shape[1] / prompt_length * 100.0) if prompt_length > 0 else 100.0
        run_metadata["tsp_enabled"] = self.tsp_idx is not None

        # --- Stage 2: Base Model Selective Prefill ---
        if self.detailed_timing: prefill_start_event.record()
        
        selective_cache_pos = torch.arange(selected_prompt_ids.shape[1], device=self.device)
        base_out = self.base_model(
            selected_prompt_ids,
            position_ids=selective_pos_ids,
            use_cache=True,
            cache_position=selective_cache_pos
        )
        base_model_next_token_ids = torch.argmax(base_out.logits[:, -1, :], dim=-1, keepdim=True)
        base_model_cache_after_prefill = base_out.past_key_values

        if self.detailed_timing:
            prefill_end_event.record()
            torch.cuda.synchronize()
            run_metadata["ttft"] = prefill_start_event.elapsed_time(prefill_end_event) / 1000.0
        
        # --- Stage 3: Base Model Generation ---
        gen_token_ids_list: List[int] = []
        if base_model_next_token_ids is not None and base_model_cache_after_prefill is not None:
            first_gen_token_id = base_model_next_token_ids.item()
            gen_token_ids_list.append(first_gen_token_id)
            current_decode_tokens = base_model_next_token_ids
            current_decode_kv_cache = DynamicCache.from_legacy_cache(base_model_cache_after_prefill)
            if first_gen_token_id not in self.eos_token_ids:
                start_pos_for_generation = (selective_pos_ids[0, -1] + 1).item() if selective_pos_ids.numel() > 0 else 0
                for i in range(max_generation_length - 1):
                    current_cache_len = current_decode_kv_cache.get_seq_length(0)
                    pos_ids = torch.tensor([[start_pos_for_generation + i]], device=self.device, dtype=torch.long)
                    decode_cache_pos = torch.tensor([current_cache_len], device=self.device)
                    decode_out = self.base_model(current_decode_tokens, position_ids=pos_ids, past_key_values=current_decode_kv_cache, use_cache=True, cache_position=decode_cache_pos)
                    next_tokens = torch.argmax(decode_out.logits[:, -1, :], dim=-1, keepdim=True)
                    gen_token_ids_list.append(next_tokens.item())
                    current_decode_kv_cache = DynamicCache.from_legacy_cache(decode_out.past_key_values)
                    current_decode_tokens = next_tokens
                    if gen_token_ids_list[-1] in self.eos_token_ids:
                        break

        final_gen_text = self.tokenizer.decode(gen_token_ids_list, skip_special_tokens=True)
        return final_gen_text, run_metadata


def main():
    parser = argparse.ArgumentParser(description="Run uniform random baseline for KV cache compression.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--base_model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--dataset_name", type=str, default="qasper", help="Name of the LongBench dataset to test.")
    parser.add_argument("--sample_idx", type=int, default=0, help="The index of the sample to use from the dataset.")
    parser.add_argument("--keep_percentage", type=float, default=0.05, help="Percentage of prompt tokens to keep for prefill (e.g., 0.05 for 5%).")
    parser.add_argument("--first_k", type=int, default=64, help="Number of first tokens to always keep.")
    parser.add_argument("--last_k", type=int, default=256, help="Number of last tokens to always keep.")
    parser.add_argument("--max_generation_length", type=int, default=128, help="Maximum number of tokens to generate.")
    parser.add_argument("--max_prompt_len", type=int, default=8192, help="Maximum prompt length to consider.")
    parser.add_argument("--tsp_idx", type=int, default=None, help="Layer index for TSP token selection. If None, TSP is disabled.")
    
    args = parser.parse_args()
    
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
        
    pipeline = UniformRandomPipeline(
        base_model_name=args.base_model_name,
        tokenizer=tokenizer,
        keep_percentage=args.keep_percentage,
        first_k=args.first_k,
        last_k=args.last_k,
        tsp_idx=args.tsp_idx,
    )

    # --- Load Prompt ---
    try:
        # Construct path relative to this script's location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
        prompt_config_path = os.path.join(project_root, 'eval', 'longbench', 'config', 'dataset2prompt.json')
        
        with open(prompt_config_path, "r") as f:
            dataset2prompt = json.load(f)
        
        data = load_dataset('THUDM/LongBench', args.dataset_name, split='test', trust_remote_code=True)
        sample = data[args.sample_idx]

        raw_prompt = dataset2prompt[args.dataset_name].format(**sample)
        messages = [{"role": "user", "content": raw_prompt}]
        templated_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    except (FileNotFoundError, IndexError, KeyError) as e:
        print(f"Error loading data or prompt: {e}")
        return

    inputs = tokenizer(templated_prompt, return_tensors="pt", truncation=True, max_length=args.max_prompt_len).to(pipeline.device)

    print(f"\n--- Running Uniform Random Baseline ---")
    print(f"Base Model: {args.base_model_name}")
    print(f"Dataset: {args.dataset_name} | Sample: {args.sample_idx}")
    print(f"Keep Percentage: {args.keep_percentage:.1%}")
    print(f"First K: {args.first_k} | Last K: {args.last_k}")
    print(f"TSP Layer: {args.tsp_idx if args.tsp_idx is not None else 'Disabled'}")
    print("-" * 43)

    generated_text, run_metadata = pipeline.run(
        input_ids=inputs.input_ids,
        max_generation_length=args.max_generation_length,
    )
    
    print(f"\n--- Generated Text ---")
    print(generated_text)
    print("-" * 22)

    print("\n--- Performance Metrics ---")
    print(f"Prompt Length (Original): {run_metadata.get('prompt_input_length', 'N/A')} tokens")
    print(f"Kept for Prefill (Compressed): {run_metadata.get('selective_prefill_kept_token_count', 'N/A')} tokens")
    print(f"Token Keep Rate: {run_metadata.get('token_keep_rate', 0):.2f}%")
    print(f"TSP Enabled: {run_metadata.get('tsp_enabled', False)}")
    print(f"Time to First Token (TTFT): {run_metadata.get('ttft', 0):.4f} seconds")
    print("-" * 27)

if __name__ == "__main__":
    main()
