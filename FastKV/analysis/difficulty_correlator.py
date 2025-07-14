# analysis/difficulty_correlator.py

import os
import sys
import argparse
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add project root to sys.path to allow for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Import necessary components ---
from analysis.compressibility_profiler import CompressibilityProfiler, calculate_entropy

from eval.longbench.evaluate import dataset2metric

from eval.longbench.main import build_chat # Helper to format prompts

class DifficultyCorrelator:
    """
    Analyzes the correlation between speculator-derived prompt entropy
    and the final task performance of a full model.
    """
    def __init__(self, args):
        self.args = args
        
        print("--- Initializing Models and Tokenizer ---")
        # For this experiment, we use the SAME model as both speculator and evaluator
        # to establish a direct correlation without model mismatch as a confounding variable.
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
        self.profiler = CompressibilityProfiler(args.model_name, self.tokenizer)
        # The profiler's model is the one we will use for both tasks.
        self.model = self.profiler.model
        print("-" * 40)

    def run_correlation_analysis(self):
        """
        Main function to run the correlation analysis for a given dataset.
        """
        # --- Load Dataset and Prompt Formats ---
        try:
            with open("eval/longbench/config/dataset2prompt.json", "r") as f:
                dataset2prompt = json.load(f)
            # Use a default max length if the model is not in the config file
            model2maxlen_path = "eval/longbench/config/model2maxlen.json"
            if os.path.exists(model2maxlen_path):
                with open(model2maxlen_path, "r") as f:
                    model2maxlen = json.load(f)
            else:
                print(f"Warning: {model2maxlen_path} not found. Using default max length.")
                model2maxlen = {}
        except FileNotFoundError as e:
            print(f"Error: Config file not found. {e}. Please run from the project root.")
            sys.exit(1)

        print(f"Loading dataset: {self.args.dataset}")
        dataset = load_dataset('THUDM/LongBench', self.args.dataset, split='test', trust_remote_code=True)
        
        # Take a subset of samples if specified
        num_samples = min(self.args.num_samples, len(dataset))
        samples_to_process = dataset.select(range(num_samples))
        
        prompt_format = dataset2prompt[self.args.dataset]
        max_len = model2maxlen.get(self.args.model_name, self.args.max_prompt_len)

        results = []
        for i, sample in enumerate(tqdm(samples_to_process, desc=f"Analyzing {self.args.dataset}")):
            # --- 1. Generate the full prompt ---
            prompt_text = prompt_format.format(**sample)
            
            # Truncate prompt if it's too long
            tokenized_prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
            if len(tokenized_prompt_ids) > max_len:
                half = int(max_len / 2)
                prompt_text = self.tokenizer.decode(tokenized_prompt_ids[:half], skip_special_tokens=True) + \
                              self.tokenizer.decode(tokenized_prompt_ids[-half:], skip_special_tokens=True)
            
            # --- 2. Calculate Speculator-Derived Entropy ---
            try:
                # We need to re-tokenize after potential truncation
                final_tokenized_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
                importance_scores = self.profiler.profile_prompt(prompt_text, max_prompt_len=max_len)
                entropy = calculate_entropy(importance_scores) if importance_scores.size > 0 else 0.0
            except Exception as e:
                print(f"Warning: Failed to calculate entropy for sample {i}. Skipping. Error: {e}")
                entropy = None

            # --- 3. Calculate Full Model Score (Ground Truth Difficulty) ---
            final_prompt = build_chat(self.tokenizer, prompt_text, self.args.model_name)
            inputs = self.tokenizer(final_prompt, truncation=False, return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.args.max_gen_len,
                    num_beams=1,
                    do_sample=False
                )[0]
            
            prediction = self.tokenizer.decode(output_ids[inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            score_fn = dataset2metric.get(self.args.dataset)
            if not score_fn:
                print(f"Warning: No metric found for dataset '{self.args.dataset}'. Skipping score calculation.")
                continue

            ground_truths = sample['answers']
            all_classes = sample.get('all_classes')
            
            max_score = 0.0
            for gt in ground_truths:
                max_score = max(max_score, score_fn(prediction, gt, all_classes=all_classes))

            if entropy is not None:
                results.append({
                    "sample_id": i,
                    "dataset": self.args.dataset,
                    "entropy": entropy,
                    "score": max_score,
                    "prompt_length": len(final_tokenized_ids)
                })

        # --- 4. Save results to CSV ---
        if results:
            df = pd.DataFrame(results)
            output_path = os.path.join(self.args.output_dir, f"correlation_data_{self.args.dataset}.csv")
            df.to_csv(output_path, index=False)
            print(f"\nAnalysis complete. Results for {len(results)} samples saved to {output_path}")
            if 'entropy' in df.columns and 'score' in df.columns and len(df) > 1:
                print("\n--- Spearman Correlation Summary ---")
                print(df[['entropy', 'score']].corr(method='spearman'))
                print("-" * 34)
        else:
            print("No results were generated.")


def main():
    parser = argparse.ArgumentParser(description="Correlate prompt entropy with model performance.")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="The model to use for both speculation and evaluation.")
    parser.add_argument("--dataset", type=str, default="qasper", help="A single LongBench dataset to analyze.")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of samples to process from the dataset.")
    parser.add_argument("--max_prompt_len", type=int, default=4096, help="Maximum length to truncate prompts to for the profiler.")
    parser.add_argument("--max_gen_len", type=int, default=128, help="Max new tokens for generation.")
    parser.add_argument("--output_dir", type=str, default="analysis_results", help="Directory to save the output CSV.")
    
    args = parser.parse_args()
    
    # Simple check for the help flag
    if '--help' in sys.argv or '-h' in sys.argv:
        return
        
    os.makedirs(args.output_dir, exist_ok=True)
    
    correlator = DifficultyCorrelator(args)
    correlator.run_correlation_analysis()

if __name__ == "__main__":
    main()
