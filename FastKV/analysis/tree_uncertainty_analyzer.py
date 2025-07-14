# analysis/tree_uncertainty_analyzer.py

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Tuple
import heapq
import json

# Add project root to sys.path if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from eval.longbench.main import build_chat
from eval.longbench.evaluate import dataset2metric

class LookaheadNode:
    """A node in the lookahead generation tree."""
    def __init__(self, token_id, log_prob, parent, past_key_values):
        self.token_id = token_id
        self.log_prob = log_prob # Cumulative log probability of the path to this node
        self.parent = parent
        self.past_key_values = past_key_values
        self.entropy = 0.0 # Entropy of the softmax distribution at this node
        self.children = []

    def __lt__(self, other):
        # heapq is a min-heap, so we use greater-than to keep the highest log_prob
        return self.log_prob > other.log_prob

class TreeUncertaintyAnalyzer:
    """
    Analyzes prompt difficulty by building and evaluating a lookahead generation tree.
    """
    def __init__(self, args):
        self.args = args
        print(f"Loading model: {args.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        ).eval()

    def calculate_softmax_entropy(self, logits: torch.Tensor) -> float:
        """Calculates the entropy of a single logit distribution."""
        if logits.numel() == 0: return 0.0
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        return entropy.item()

    @torch.no_grad()
    def analyze_prompt(self, prompt_text: str) -> Dict:
        """
        Builds a lookahead tree for a prompt and calculates uncertainty metrics.
        """
        inputs = self.tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=self.args.max_prompt_len).to(self.model.device)
        
        # --- Prefill Phase ---
        prefill_outputs = self.model(inputs.input_ids, use_cache=True)
        
        # --- Tree Building Phase ---
        root_logits = prefill_outputs.logits[:, -1, :]
        root_log_probs = F.log_softmax(root_logits, dim=-1)
        
        # Initialize the beam (a min-heap) for beam search
        # We store (cumulative_log_prob, node)
        beam = []
        
        # All entropies calculated in the tree
        all_node_entropies = [self.calculate_softmax_entropy(root_logits)]

        # Initialize the first level of the tree
        top_k_probs, top_k_indices = torch.topk(root_log_probs, self.args.branching_factor, dim=-1)
        
        for i in range(self.args.branching_factor):
            token_id = top_k_indices[0, i].item()
            log_prob = top_k_probs[0, i].item()
            # The root node has no token_id or parent
            root_node = LookaheadNode(None, 0.0, None, prefill_outputs.past_key_values)
            node = LookaheadNode(token_id, log_prob, root_node, prefill_outputs.past_key_values)
            heapq.heappush(beam, (node.log_prob, id(node), node)) # Use id(node) to break ties

        # Build the tree level by level
        for depth in range(1, self.args.tree_depth):
            next_beam = []
            
            # Limit the beam size to avoid explosion
            while beam and len(next_beam) < self.args.branching_factor:
                # Get the most likely sequence so far
                _, _, current_node = heapq.heappop(beam)
                
                # Get next token predictions
                current_token_tensor = torch.tensor([[current_node.token_id]], device=self.model.device)
                
                outputs = self.model(
                    input_ids=current_token_tensor,
                    past_key_values=current_node.past_key_values,
                    use_cache=True
                )
                
                next_logits = outputs.logits[:, -1, :]
                all_node_entropies.append(self.calculate_softmax_entropy(next_logits))
                
                next_log_probs = F.log_softmax(next_logits, dim=-1)
                top_k_probs, top_k_indices = torch.topk(next_log_probs, self.args.branching_factor, dim=-1)
                
                for i in range(self.args.branching_factor):
                    token_id = top_k_indices[0, i].item()
                    log_prob = top_k_probs[0, i].item()
                    
                    # Create a new node with updated cumulative probability
                    new_node = LookaheadNode(
                        token_id=token_id,
                        log_prob=current_node.log_prob + log_prob,
                        parent=current_node,
                        past_key_values=outputs.past_key_values
                    )
                    heapq.heappush(next_beam, (new_node.log_prob, id(new_node), new_node))
            
            beam = next_beam
            if not beam: break

        # --- Metric Calculation ---
        # The simplest, most robust metric is the average entropy of all nodes explored.
        avg_tree_entropy = np.mean(all_node_entropies) if all_node_entropies else 0.0
        
        return {"lookahead_tree_entropy": avg_tree_entropy}

    def run_full_analysis(self):
        """Orchestrates the entire analysis for a dataset."""
        # --- Load Dataset and Prompt Formats ---
        try:
            with open("eval/longbench/config/dataset2prompt.json", "r") as f:
                dataset2prompt = json.load(f)
        except FileNotFoundError:
            print("Error: Config files not found. Please run from the project root.")
            return

        print(f"Loading dataset: {self.args.dataset}")
        dataset = load_dataset('THUDM/LongBench', self.args.dataset, split='test', trust_remote_code=True)
        
        num_samples = min(self.args.num_samples, len(dataset))
        samples_to_process = dataset.select(range(num_samples))
        
        prompt_format = dataset2prompt[self.args.dataset]
        score_fn = dataset2metric.get(self.args.dataset)
        
        results = []
        for i, sample in enumerate(tqdm(samples_to_process, desc=f"Analyzing {self.args.dataset}")):
            prompt_text = prompt_format.format(**sample)
            
            # --- 1. Get Tree Uncertainty Metric ---
            uncertainty_metrics = self.analyze_prompt(prompt_text)
            
            # --- 2. Get Ground Truth Score ---
            final_prompt = build_chat(self.tokenizer, prompt_text, self.args.model_name)
            inputs = self.tokenizer(final_prompt, return_tensors="pt", truncation=True, max_length=self.args.max_prompt_len).to(self.model.device)
            
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
                "tree_entropy": uncertainty_metrics["lookahead_tree_entropy"],
                "score": max_score,
            })

        # --- 3. Save Results ---
        if results:
            df = pd.DataFrame(results)
            output_path = os.path.join(self.args.output_dir, f"tree_correlation_data_{self.args.dataset}.csv")
            df.to_csv(output_path, index=False)
            print(f"\nAnalysis complete. Results saved to {output_path}")
            if 'tree_entropy' in df.columns and 'score' in df.columns and len(df) > 1:
                print("\n--- Spearman Correlation Summary ---")
                print(df[['tree_entropy', 'score']].corr(method='spearman'))
                print("-" * 34)

def main():
    parser = argparse.ArgumentParser(description="Analyze prompt difficulty using lookahead tree uncertainty.")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--dataset", type=str, default="qasper")
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--max_prompt_len", type=int, default=4096)
    parser.add_argument("--max_gen_len", type=int, default=128)
    parser.add_argument("--output_dir", type=str, default="analysis_results")
    # Tree-specific arguments
    parser.add_argument("--tree_depth", type=int, default=3, help="Depth of the lookahead tree.")
    parser.add_argument("--branching_factor", type=int, default=3, help="Branching factor (beam width) at each node.")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    analyzer = TreeUncertaintyAnalyzer(args)
    analyzer.run_full_analysis()

if __name__ == "__main__":
    main()
