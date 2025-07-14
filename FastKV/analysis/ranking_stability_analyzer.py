import os
import sys
import argparse
import pickle
import types
import json
from functools import partial
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaFlashAttention2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from analysis.hijack_for_ranking_analysis import analysis_attention_forward
from baseline.fastkv.fastkv_utils import FastKVCluster

class RankingStabilityProfiler:
    """(This class remains unchanged)"""
    def __init__(self, model_name: str, tokenizer: AutoTokenizer):
        print(f"--- Initializing Profiler for {model_name} ---")
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2"
        )
        self.model.eval()
        self.original_forwards = {}
        print("Model loaded successfully.")

    def _patch_model_for_analysis(self, storage_dict: dict):
        num_patched = 0
        for i, layer in enumerate(self.model.model.layers):
            if not hasattr(layer, 'self_attn'): continue
            attn_module = layer.self_attn
            if not isinstance(attn_module, LlamaFlashAttention2): continue
            if not hasattr(attn_module, 'kv_cluster'):
                attn_module.kv_cluster = FastKVCluster()
            if i not in self.original_forwards:
                self.original_forwards[i] = attn_module.forward
            new_forward_partial = partial(analysis_attention_forward, self=attn_module, all_layer_rankings_storage=storage_dict)
            attn_module.forward = new_forward_partial
            num_patched += 1
        if num_patched == 0: raise RuntimeError("Failed to patch any attention layers.")
        print(f"Patched {num_patched} layers.")

    def _unpatch_model(self):
        num_unpatched = 0
        for i, layer in enumerate(self.model.model.layers):
            if i in self.original_forwards and hasattr(layer, 'self_attn'):
                layer.self_attn.forward = self.original_forwards[i]
                num_unpatched += 1
        print(f"Unpatched {num_unpatched} layers.")
        self.original_forwards = {}

    def profile_single_prompt(self, prompt_text: str, max_len: int) -> list:
        """
        Original method that takes text, tokenizes, and profiles.
        """
        # We can actually simplify this to just call the new method
        inputs = self.tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=max_len).to(self.model.device)
        return self.profile_single_prompt_from_ids(inputs.input_ids)

    def profile_single_prompt_from_ids(self, input_ids: torch.Tensor) -> list:
        """
        Runs a single prompt through the patched model using pre-tokenized IDs.
        """
        num_layers = self.model.config.num_hidden_layers
        storage_dict = {}
        all_layer_rankings_list = [None] * num_layers
        
        try:
            # The patching function needs to be told where to store the data
            self._patch_model_for_analysis(storage_dict=storage_dict)
            
            print(f"Profiling prompt with {input_ids.shape[1]} tokens...")
            with torch.no_grad():
                # Pass the input_ids tensor directly to the model
                self.model(input_ids=input_ids)
        finally:
            # Always ensure the model is unpatched
            self._unpatch_model()
            
        # Convert the populated dictionary to a correctly ordered list
        for i in range(num_layers):
            if i in storage_dict:
                all_layer_rankings_list[i] = storage_dict[i]

        return all_layer_rankings_list

def main():
    parser = argparse.ArgumentParser(description="Analyze and aggregate token ranking stability across multiple datasets.")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    # MODIFIED: Takes a list of datasets
    parser.add_argument("--datasets", nargs='+', required=True, help="Space-separated list of LongBench datasets to analyze (e.g., qasper narrativeqa trec).")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of valid samples to process PER dataset.")
    parser.add_argument("--max_len", type=int, default=8192, help="The MAXIMUM prompt length to consider. Samples longer than this are skipped.")
    parser.add_argument("--scan_limit", type=int, default=200, help="Scan up to this many entries from each dataset to find valid samples.")
    parser.add_argument("--output_dir", type=str, default="analysis_results/ranking_stability")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    profiler = RankingStabilityProfiler(args.model_name, tokenizer)

    # --- NEW: Top-level dictionary to hold all results ---
    multi_dataset_results = {}

    try:
        with open("eval/longbench/config/dataset2prompt.json", "r") as f:
            dataset2prompt = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: Could not load LongBench prompt format ({e}). Exiting.")
        sys.exit(1)

    # --- NEW: Loop over all requested datasets ---
    for dataset_name in args.datasets:
        print(f"\n{'='*20} Processing Dataset: {dataset_name.upper()} {'='*20}")
        if dataset_name not in dataset2prompt:
            print(f"Warning: No prompt format found for '{dataset_name}'. Skipping.")
            continue
        
        prompt_format = dataset2prompt[dataset_name]
        dataset = load_dataset('THUDM/LongBench', dataset_name, split='test', trust_remote_code=True)
        
        valid_prompts = []
        for i in tqdm(range(min(args.scan_limit, len(dataset))), desc=f"Scanning {dataset_name}"):
            sample = dataset[i]
            try:
                prompt_text = prompt_format.format(**sample)
                tokenized_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
                if len(tokenized_ids) <= args.max_len:
                    valid_prompts.append(prompt_text)
            except KeyError: # Handles cases where a sample is missing a key in the format string
                continue

        if len(valid_prompts) < args.num_samples:
            print(f"Warning: Requested {args.num_samples}, but only found {len(valid_prompts)} valid samples for '{dataset_name}'.")
        
        samples_to_process = valid_prompts[:args.num_samples]
        if not samples_to_process:
            print(f"No valid samples found for '{dataset_name}'.")
            continue

        dataset_rankings = []
        for i, prompt_text in enumerate(tqdm(samples_to_process, desc=f"Profiling {dataset_name}")):
            print(f"Profiling prompt {i+1}/{len(samples_to_process)}...")
            layer_rankings = profiler.profile_single_prompt(prompt_text)
            if layer_rankings:
                dataset_rankings.append(layer_rankings)
        
        multi_dataset_results[dataset_name] = dataset_rankings

    if not multi_dataset_results:
        print("\nNo data was captured for any dataset. Exiting.")
        sys.exit(1)

    # --- Save the single aggregated file ---
    output_filename = f"ranking_data_model_{args.model_name.replace('/', '_')}.pkl"
    output_path = os.path.join(args.output_dir, output_filename)
    
    with open(output_path, "wb") as f:
        pickle.dump(multi_dataset_results, f)
        
    print(f"\n--- Analysis Complete ---")
    print(f"Successfully aggregated and saved ranking data for {len(multi_dataset_results)} datasets to: {output_path}")

if __name__ == "__main__":
    main()
