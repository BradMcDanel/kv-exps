# analysis/generate_method_rankings.py
import os
import sys
import argparse
import pickle
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from analysis.ranking_stability_analyzer import RankingStabilityProfiler

def main():
    parser = argparse.ArgumentParser(description="Generate layer-wise rankings for multiple models.")
    parser.add_argument("--base_model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--spec_model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--oracle_file", type=str, required=True, help="Path to the .pkl file containing the input_ids and oracle rankings.")
    parser.add_argument("--output_dir", type=str, default="analysis_results/method_rankings")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(args.oracle_file, "rb") as f:
        oracle_data = pickle.load(f)
    all_method_rankings = {}
    
    models_to_profile = {
        "base_model": args.base_model_name,
        "spec_model": args.spec_model_name
    }
    for model_key, model_name in models_to_profile.items():
        print(f"\n--- Profiling {model_key}: {model_name} ---")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        profiler = RankingStabilityProfiler(model_name, tokenizer)
        
        for dataset_name, samples_dict in oracle_data.items():
            if dataset_name not in all_method_rankings:
                all_method_rankings[dataset_name] = {}
            for sample_idx, sample_data in tqdm(samples_dict.items(), desc=f"Profiling {model_name} for {dataset_name}"):
                # --- DEBUG: Check token count ---
                input_ids = sample_data["final_input_ids"].to(profiler.model.device)
                print(f"Sample {sample_idx}: input_ids length = {input_ids.shape[1]}")
                
                # Call the profiler method that takes IDs directly
                layer_wise_rankings = profiler.profile_single_prompt_from_ids(input_ids)
                
                # --- DEBUG: Check ranking length ---
                if layer_wise_rankings:
                    print(f"Sample {sample_idx}: layer_wise_rankings type = {type(layer_wise_rankings)}")
                    if isinstance(layer_wise_rankings, list):
                        print(f"Sample {sample_idx}: number of layers = {len(layer_wise_rankings)}")
                        # Check if any layer has ranking data
                        non_none_layers = [i for i, layer_data in enumerate(layer_wise_rankings) if layer_data is not None]
                        if non_none_layers:
                            first_non_none_layer = layer_wise_rankings[non_none_layers[0]]
                            print(f"Sample {sample_idx}: first layer with data (layer {non_none_layers[0]}) type = {type(first_non_none_layer)}")
                            if hasattr(first_non_none_layer, '__len__'):
                                print(f"Sample {sample_idx}: first layer ranking length = {len(first_non_none_layer)}")
                    else:
                        print(f"Sample {sample_idx}: unexpected ranking format")
                
                if sample_idx not in all_method_rankings[dataset_name]:
                    all_method_rankings[dataset_name][sample_idx] = {}
                all_method_rankings[dataset_name][sample_idx][f"{model_key}_rankings"] = layer_wise_rankings
        
        del profiler, tokenizer
        gc.collect(); torch.cuda.empty_cache()
    
    output_filename = f"methods_basemodel_{args.base_model_name.replace('/', '_')}_specmodel_{args.spec_model_name.replace('/', '_')}.pkl"
    output_path = os.path.join(args.output_dir, output_filename)
    with open(output_path, "wb") as f:
        pickle.dump(all_method_rankings, f)
    print(f"\n--- Method Ranking Generation Complete ---")
    print(f"Saved all method rankings to {output_path}")

if __name__ == "__main__":
    main()
