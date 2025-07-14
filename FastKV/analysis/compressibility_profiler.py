# analysis/compressibility_profiler.py

import os
import sys
import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from typing import List

# Add project root to sys.path to allow import from baseline/
try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from baseline.speculative_prefill.main import AdaptiveContextPruningPipeline
except ImportError:
    print("Error: Could not import AdaptiveContextPruningPipeline.")
    print("Please ensure you are running this script from the 'analysis' directory within your project root,")
    print("and that the main pipeline class is located at 'baseline/speculative_prefill/main.py'.")
    sys.exit(1)


def profile_and_visualize(profiler_pipeline: AdaptiveContextPruningPipeline, datasets_to_profile: List[str], num_samples: int, max_len: int, output_dir: str):
    """
    Profiles prompts from specified datasets and visualizes the score distributions.
    """
    # --- Load Prompt Formats ---
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.join(script_dir, '..')
        prompt_config_path = os.path.join(project_root, 'eval', 'longbench', 'config', 'dataset2prompt.json')
        if not os.path.exists(prompt_config_path):
             raise FileNotFoundError(f"Config file not found at {prompt_config_path}")
        with open(prompt_config_path, "r") as f:
            dataset2prompt = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the LongBench config is correctly placed at 'eval/longbench/config/dataset2prompt.json' relative to your project root.")
        sys.exit(1)

    all_scores_data = []

    for dataset_name in datasets_to_profile:
        print(f"\n--- Profiling dataset: {dataset_name} ---")
        try:
            dataset = load_dataset('THUDM/LongBench', dataset_name, split='test', trust_remote_code=True)
            prompt_format = dataset2prompt[dataset_name]
            
            for i in tqdm(range(min(num_samples, len(dataset))), desc=f"Profiling {dataset_name}"):
                sample = dataset[i]
                
                try:
                    prompt_text = prompt_format.format(**sample)
                except KeyError as e:
                    print(f"Warning: Skipping sample {i} in {dataset_name} due to missing key for formatting: {e}")
                    continue

                inputs = profiler_pipeline.tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=max_len).to(profiler_pipeline.device)
                
                if inputs.input_ids.shape[1] == 0:
                    continue

                with torch.no_grad():
                    spec_out = profiler_pipeline.speculator_model(
                        input_ids=inputs.input_ids, output_attentions=True
                    )
                    scores = profiler_pipeline._get_adaptive_importance_scores(spec_out.attentions)
                
                for score_val in scores.cpu().numpy():
                    all_scores_data.append({"dataset": dataset_name, "score": score_val})
                
                # Explicitly clear cache after each sample
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"Could not process dataset {dataset_name}. Error: {e}")
            continue

    # --- Create plots ---
    if not all_scores_data:
        print("No data was generated. Exiting.")
        return

    df = pd.DataFrame(all_scores_data)
    
    # Plot 1: Violin plot to see the distribution shape
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.violinplot(x='dataset', y='score', data=df, ax=ax, cut=0)
    ax.set_title('Distribution of Adaptive Importance Scores', fontsize=16, pad=20)
    ax.set_xlabel('Dataset (Task Type)', fontsize=12)
    ax.set_ylabel('Token Importance Score', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "importance_score_distribution.png")
    plt.savefig(plot_path)
    print(f"\nScore distribution plot saved to: {plot_path}")
    plt.close()

    # Plot 2: Log-scale histogram for one dataset to see the magnitude span
    fig, ax = plt.subplots(figsize=(10, 6))
    if df.empty or datasets_to_profile[0] not in df['dataset'].unique():
        print("Not enough data to generate histogram for the first dataset.")
    else:
        example_dataset = datasets_to_profile[0]
        log_scores = np.log10(df[df['dataset'] == example_dataset]['score'].values + 1e-12) # add epsilon to avoid log(0)
        ax.hist(log_scores, bins=50)
        ax.set_title(f'Log-Scale Score Histogram for: {example_dataset}', fontsize=16)
        ax.set_xlabel('log10(Token Importance Score)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        plt.tight_layout()
        hist_path = os.path.join(output_dir, "log_score_histogram_example.png")
        plt.savefig(hist_path)
        print(f"Log-scale histogram saved to: {hist_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Profile prompt compressibility with Adaptive Context Pruning.")
    parser.add_argument("--speculator_model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--datasets", nargs='+', default=["qasper", "gov_report", "lcc", "hotpotqa"], help="List of LongBench datasets to profile.")
    parser.add_argument("--max_prompt_len", type=int, default=4096)
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples per dataset.")
    parser.add_argument("--output_dir", type=str, default="analysis_results", help="Directory for plots.")
    parser.add_argument("--score_sharpening_temp", type=float, default=0.1, help="Temperature for sharpening scores. Lower is sharper. 0 to disable.")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.speculator_model_name)
    
    pipeline = AdaptiveContextPruningPipeline(
        base_model_name=args.speculator_model_name,
        speculator_model_name=args.speculator_model_name,
        tokenizer=tokenizer,
        score_sharpening_temp=args.score_sharpening_temp
    )
    
    profile_and_visualize(
        profiler_pipeline=pipeline,
        datasets_to_profile=args.datasets,
        num_samples=args.num_samples,
        max_len=args.max_prompt_len,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main()
