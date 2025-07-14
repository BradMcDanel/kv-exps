# analysis/generate_all_rankings.py

import os
import sys
import argparse
import pickle
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from analysis.ranking_stability_analyzer import RankingStabilityProfiler # For FastKV-style
from analysis.generate_oracles import OracleGenerator # For Oracle

def main():
    parser = argparse.ArgumentParser(description="Generate all oracle and method rankings from a single input file.")
    parser.add_argument("--base_model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--spec_model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the .pkl file containing tokenized inputs.")
    parser.add_argument("--max_gen_len", type=int, default=256)
    parser.add_argument("--output_file", type=str, default="analysis_results/all_rankings.pkl")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    with open(args.input_file, "rb") as f:
        all_data = pickle.load(f)

    # --- Phase 1: Generate Oracle and Base Model Rankings (uses 8B model) ---
    print(f"--- Loading Base Model: {args.base_model_name} ---")
    base_tokenizer = AutoTokenizer.from_pretrained(args.base_model_name)
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model_name, torch_dtype=torch.bfloat16, device_map="auto")
    
    oracle_generator = OracleGenerator(base_model, base_tokenizer)
    base_profiler = RankingStabilityProfiler(base_model, base_tokenizer)

    for dataset_name, samples in all_data.items():
        print(f"\n--- Processing {dataset_name} with Base Model ---")
        for sample_idx, sample_data in tqdm(samples.items(), desc=f"Oracle & Base Rankings for {dataset_name}"):
            input_ids = sample_data["input_ids"].to(base_model.device)
            inputs = {'input_ids': input_ids}
            
            # Generate Oracle Ranking
            oracle_ranking = oracle_generator.generate_answer_informed_oracle(inputs, args.max_gen_len)
            sample_data["oracle_ranking"] = oracle_ranking
            
            # Generate Base Model Layer-wise Rankings
            base_rankings = base_profiler.profile_single_prompt_from_ids(input_ids)
            sample_data["base_model_rankings"] = base_rankings
    
    del oracle_generator, base_profiler, base_model, base_tokenizer
    gc.collect(); torch.cuda.empty_cache()

    # --- Phase 2: Generate Speculator Model Rankings (uses 1B model) ---
    print(f"\n--- Loading Speculator Model: {args.spec_model_name} ---")
    spec_tokenizer = AutoTokenizer.from_pretrained(args.spec_model_name)
    spec_model = AutoModelForCausalLM.from_pretrained(args.spec_model_name, torch_dtype=torch.bfloat16, device_map="auto")
    spec_profiler = RankingStabilityProfiler(spec_model, spec_tokenizer)

    for dataset_name, samples in all_data.items():
        print(f"\n--- Processing {dataset_name} with Speculator Model ---")
        for sample_idx, sample_data in tqdm(samples.items(), desc=f"Speculator Rankings for {dataset_name}"):
            # Tokenize with spec tokenizer to match what it would see. Lengths may differ, that's part of the analysis.
            raw_text = spec_tokenizer.decode(sample_data["input_ids"][0], skip_special_tokens=True)
            spec_inputs = spec_tokenizer(raw_text, return_tensors='pt', truncation=True, max_length=input_ids.shape[1]).to(spec_model.device)
            
            spec_rankings = spec_profiler.profile_single_prompt_from_ids(spec_inputs.input_ids)
            sample_data["spec_model_rankings"] = spec_rankings
    
    del spec_profiler, spec_model, spec_tokenizer
    gc.collect(); torch.cuda.empty_cache()

    # --- Save everything to a single, final file ---
    with open(args.output_file, "wb") as f:
        pickle.dump(all_data, f)
    
    print(f"\n--- All Rankings Generated ---")
    print(f"Saved comprehensive results to {args.output_file}")


if __name__ == "__main__":
    main()
