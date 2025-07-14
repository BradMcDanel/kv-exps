# analysis/peak_counter_analyzer.py

import os
import sys
import argparse
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import find_peaks
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add project root to sys.path to allow for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Import necessary components ---
from analysis.compressibility_profiler import CompressibilityProfiler # We reuse the profiler to get the base importance scores
from eval.longbench.main import build_chat
from eval.longbench.evaluate import dataset2metric

class PeakCounterAnalyzer:
    """
    Analyzes the correlation between the number of attention peaks in a prompt
    and the final task performance of a full model.
    """
    def __init__(self, args):
        self.args = args
        print("--- Initializing Model, Tokenizer, and Profiler ---")
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
        self.profiler = CompressibilityProfiler(args.model_name, self.tokenizer)
        self.model = self.profiler.model # Use the same model instance for both
        print("-" * 50)

    def calculate_attention_peaks(self, importance_scores: np.ndarray) -> int:
        """
        Calculates the number of significant peaks in an importance score vector.
        """
        if importance_scores.size < 3: # find_peaks requires at least 3 points
            return importance_scores.size

        # A "peak" is defined as a point that is higher than a certain threshold.
        # A good dynamic threshold is a multiple of the standard deviation above the mean.
        # This helps ignore minor "noise" and focus on significant spikes.
        mean_score = importance_scores.mean()
        std_dev = importance_scores.std()
        
        # Hyperparameter: How many standard deviations above the mean to count as a peak.
        # A higher value makes the criterion stricter. Let's start with 1.5.
        height_threshold = mean_score + (self.args.peak_std_threshold * std_dev)

        # scipy.signal.find_peaks is perfect for this.
        # 'distance' ensures peaks are at least N tokens apart, preventing counting jittery noise as multiple peaks.
        peaks, _ = find_peaks(importance_scores, height=height_threshold, distance=self.args.peak_min_distance)
        
        return len(peaks)

    def run_analysis(self):
        """Main function to run the correlation analysis."""
        # --- Load Dataset and Prompt Formats ---
        try:
            with open("eval/longbench/config/dataset2prompt.json", "r") as f:
                dataset2prompt = json.load(f)
            model2maxlen_path = "eval/longbench/config/model2maxlen.json"
            model2maxlen = json.load(open(model2maxlen_path, "r")) if os.path.exists(model2maxlen_path) else {}
        except FileNotFoundError as e:
            print(f"Error: Config files not found. {e}. Please run from the project root.")
            return

        print(f"Loading dataset: {self.args.dataset}")
        dataset = load_dataset('THUDM/LongBench', self.args.dataset, split='test', trust_remote_code=True)
        
        num_samples = min(self.args.num_samples, len(dataset))
        samples_to_process = dataset.select(range(num_samples))
        
        prompt_format = dataset2prompt[self.args.dataset]
        max_len = model2maxlen.get(self.args.model_name, self.args.max_prompt_len)
        score_fn = dataset2metric.get(self.args.dataset)

        results = []
        for i, sample in enumerate(tqdm(samples_to_process, desc=f"Analyzing {self.args.dataset}")):
            prompt_text = prompt_format.format(**sample)
            
            # Truncate if necessary
            tokenized_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
            if len(tokenized_ids) > max_len:
                half = int(max_len / 2)
                prompt_text = self.tokenizer.decode(tokenized_ids[:half]) + self.tokenizer.decode(tokenized_ids[-half:])
            
            # --- 1. Get Attention Peaks (Predicted Difficulty) ---
            try:
                importance_scores = self.profiler.profile_prompt(prompt_text, max_prompt_len=max_len)
                peak_count = self.calculate_attention_peaks(importance_scores) if importance_scores.size > 0 else 0
            except Exception as e:
                print(f"\nWarning: Failed to calculate peaks for sample {i}. Error: {e}")
                continue

            # --- 2. Get Ground Truth Score ---
            final_prompt = build_chat(self.tokenizer, prompt_text, self.args.model_name)
            inputs = self.tokenizer(final_prompt, return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs, max_new_tokens=self.args.max_gen_len,
                    num_beams=1, do_sample=False)[0]
            
            prediction = self.tokenizer.decode(output_ids[inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            max_score = 0.0
            for gt in sample['answers']:
                max_score = max(max_score, score_fn(prediction, gt, all_classes=sample.get('all_classes')))
                
            results.append({
                "sample_id": i,
                "dataset": self.args.dataset,
                "peak_count": peak_count,
                "score": max_score,
            })

        # --- 3. Save Results ---
        if results:
            df = pd.DataFrame(results)
            output_path = os.path.join(self.args.output_dir, f"peak_correlation_data_{self.args.dataset}.csv")
            df.to_csv(output_path, index=False)
            print(f"\nAnalysis complete. Results saved to {output_path}")
            if 'peak_count' in df.columns and 'score' in df.columns and len(df) > 1:
                print("\n--- Spearman Correlation Summary ---")
                print(df[['peak_count', 'score']].corr(method='spearman'))
                print("-" * 34)

def main():
    parser = argparse.ArgumentParser(description="Correlate attention peak count with model performance.")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--dataset", type=str, default="qasper")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--max_prompt_len", type=int, default=4096)
    parser.add_argument("--max_gen_len", type=int, default=128)
    parser.add_argument("--output_dir", type=str, default="analysis_results")
    # Peak detection hyperparameters
    parser.add_argument("--peak_std_threshold", type=float, default=1.5, help="How many std devs above mean to be a peak.")
    parser.add_argument("--peak_min_distance", type=int, default=10, help="Minimum distance (in tokens) between peaks.")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    analyzer = PeakCounterAnalyzer(args)
    analyzer.run_analysis()

if __name__ == "__main__":
    main()
