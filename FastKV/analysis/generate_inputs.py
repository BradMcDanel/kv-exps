# analysis/generate_inputs.py

import os
import argparse
import pickle
import json
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer

def build_chat(tokenizer, prompt, model_name):
    if "Llama-2" in model_name:
        return f"[INST] {prompt} [/INST]"
    else:
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def main():
    parser = argparse.ArgumentParser(description="Generate and save tokenized inputs for analysis.")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="The tokenizer reference model.")
    parser.add_argument("--datasets", nargs='+', required=True)
    parser.add_argument("--num_samples", type=int, default=15)
    parser.add_argument("--max_len", type=int, default=4096)
    parser.add_argument("--scan_limit", type=int, default=200)
    parser.add_argument("--output_file", type=str, default="analysis_results/inputs.pkl")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    dataset2prompt = json.load(open("eval/longbench/config/dataset2prompt.json", "r"))
    
    all_inputs = {}
    for dataset_name in args.datasets:
        print(f"\n--- Scanning Dataset: {dataset_name.upper()} ---")
        prompt_format = dataset2prompt.get(dataset_name)
        if not prompt_format: continue
        
        data = load_dataset('THUDM/LongBench', dataset_name, split='test', trust_remote_code=True)
        
        dataset_inputs = {}
        processed_count = 0
        for i in range(min(args.scan_limit, len(data))):
            if processed_count >= args.num_samples: break
            sample = data[i]
            try:
                raw_prompt = prompt_format.format(**sample)
                
                # Check raw prompt length first for efficiency
                if len(tokenizer.encode(raw_prompt, add_special_tokens=False)) > args.max_len:
                    continue

                datasets_without_chat_template = ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]
                if dataset_name not in datasets_without_chat_template:
                    final_prompt_text = build_chat(tokenizer, raw_prompt, args.model_name)
                else:
                    final_prompt_text = raw_prompt

                inputs = tokenizer(final_prompt_text, return_tensors="pt", truncation=False)
                
                # Final check after adding chat template tokens
                if inputs.input_ids.shape[1] <= args.max_len:
                    dataset_inputs[i] = {"input_ids": inputs.input_ids.cpu()}
                    processed_count += 1
            except Exception as e:
                print(f"Skipping sample {i} for {dataset_name} due to error: {e}")
                continue
        
        all_inputs[dataset_name] = dataset_inputs

    with open(args.output_file, "wb") as f:
        pickle.dump(all_inputs, f)
    
    print(f"\n--- Input Generation Complete ---")
    print(f"Saved tokenized inputs for {sum(len(v) for v in all_inputs.values())} total samples to {args.output_file}")

if __name__ == "__main__":
    main()
