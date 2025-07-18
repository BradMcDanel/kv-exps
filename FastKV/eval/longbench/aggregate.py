# eval/longbench/aggregate.py

"""
Script to aggregate LongBench results across different KV cache methods.
Reads results.json files from each method folder and outputs to CSV.
"""

import os
import json
import csv
from pathlib import Path
import sys

# Define the datasets in the specified order with their display names
DATASETS = [
    ("narrativeqa", "NrtvQA"),
    ("qasper", "Qasper"), 
    ("multifieldqa_en", "MF-en"),
    ("hotpotqa", "HotpotQA"),
    ("2wikimqa", "2WikiMQA"),
    ("musique", "MuSiQue"),
    ("gov_report", "GovReport"),
    ("qmsum", "QMSum"),
    ("multi_news", "MultiNews"),
    ("trec", "TREC"),
    ("triviaqa", "TriviaQA"),
    ("samsum", "SAMSum"),
    ("lcc", "LCC"),
    ("repobench-p", "RB-P"),
    ("passage_count", "PCount"),
    ("passage_retrieval_en", "PRe"),
]

# Extract just the keys for internal use
DATASET_KEYS = [key for key, _ in DATASETS]
DATASET_NAMES = [name for _, name in DATASETS]

def load_results(results_file):
    """Load results from a JSON file, return empty dict if file doesn't exist or is invalid."""
    try:
        with open(results_file, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, IOError) as e:
        print(f"Warning: Could not load {results_file}: {e}")
        return {}

def parse_method_and_budget(folder_name):
    """
    Parse method name and KV budget from folder name.
    
    Args:
        folder_name: Name of the folder (e.g., 'fastkv_256', 'fullkv')
    
    Returns:
        tuple: (method_name, kv_budget)
    """
    # Special case for fullkv - always uses "Full" budget
    if folder_name == "fullkv":
        return "fullkv", "Full"
    
    # Check if folder name ends with _number
    parts = folder_name.split('_')
    if len(parts) > 1 and parts[-1].isdigit():
        method_name = '_'.join(parts[:-1])
        kv_budget = int(parts[-1])
    else:
        method_name = folder_name
        kv_budget = 512  # Default budget
    
    return method_name, kv_budget

def aggregate_longbench_results(base_path="outputs/meta-llama/Llama-3.1-8B-Instruct/longbench"):
    """
    Aggregate LongBench results from all method folders.
    
    Args:
        base_path: Path to the longbench results directory
    
    Returns:
        List of dictionaries containing aggregated results
    """
    results = []
    
    # Check if base path exists
    if not os.path.exists(base_path):
        print(f"Error: Base path {base_path} does not exist!")
        return results
    
    # Get all method folders
    method_folders = [d for d in os.listdir(base_path) 
                     if os.path.isdir(os.path.join(base_path, d))]
    method_folders.sort()  # Sort for consistent ordering
    
    print(f"Found method folders: {method_folders}")
    
    # Check for multiple fullkv entries
    fullkv_folders = [f for f in method_folders if f == "fullkv"]
    if len(fullkv_folders) > 1:
        raise ValueError(f"Found multiple fullkv folders: {fullkv_folders}. There should only be one.")
    
    for folder_name in method_folders:
        method_path = os.path.join(base_path, folder_name)
        results_file = os.path.join(method_path, "results.json")
        
        # Load results for this method
        method_results = load_results(results_file)
        
        # Parse method name and KV budget
        method_name, kv_budget = parse_method_and_budget(folder_name)
        
        # Create row for this method
        row = {
            "Method": method_name,
            "KVBudget": kv_budget
        }
        
        # Add results for each dataset (in specified order)
        valid_scores = []
        for dataset_key, dataset_name in DATASETS:
            if dataset_key in method_results:
                score = method_results[dataset_key]
                row[dataset_name] = score
                # Collect valid scores for average calculation
                if isinstance(score, (int, float)):
                    valid_scores.append(score)
            else:
                row[dataset_name] = ""  # Empty placeholder for missing datasets
        
        # Calculate average of valid scores
        if valid_scores:
            row["Avg"] = sum(valid_scores) / len(valid_scores)
        else:
            row["Avg"] = ""
        
        results.append(row)
    
    # Sort results: fullkv first, then by KVBudget (ascending), then by Method
    def sort_key(result):
        method = result["Method"]
        budget = result["KVBudget"]
        
        if method == "fullkv":
            return (0, 0, method)  # fullkv always first
        else:
            # For numeric budgets, sort by budget then method
            # For non-numeric budgets, sort by string value
            if isinstance(budget, int):
                return (1, budget, method)
            else:
                return (1, float('inf'), method)  # Non-numeric budgets go last
    
    results.sort(key=sort_key)
    
    return results

def save_to_csv(results, base_path, output_filename="longbench_results.csv"):
    """Save aggregated results to CSV file in the base path directory."""
    if not results:
        print("No results to save!")
        return
    
    # Create output file path in the base_path directory
    output_file = os.path.join(base_path, output_filename)
    
    # CSV headers: Method + KVBudget + all dataset display names + average
    headers = ["Method", "KVBudget"] + DATASET_NAMES + ["Avg"]
    
    try:
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            writer.writerows(results)
        print(f"Results saved to {output_file}")
    except IOError as e:
        print(f"Error saving CSV file: {e}")

def print_summary_table(results):
    """Print a formatted summary table to screen."""
    if not results:
        print("No results to display!")
        return
    
    print("\n" + "="*120)
    print("LONGBENCH RESULTS SUMMARY")
    print("="*120)
    
    # Calculate column widths
    method_width = max(8, max(len(str(r["Method"])) for r in results))  # Min 8 for "Method"
    budget_width = 8  # Fixed width for KVBudget
    score_width = 8   # Fixed width for scores
    
    # Build header row
    header_parts = []
    header_parts.append(f"{'Method':<{method_width}}")
    header_parts.append(f"{'KVBudget':<{budget_width}}")
    
    for dataset_name in DATASET_NAMES:
        header_parts.append(f"{dataset_name:<{score_width}}")
    header_parts.append(f"{'Avg':<{score_width}}")
    
    header = " ".join(header_parts)
    print(header)
    print("-" * len(header))
    
    # Print results for each method
    for result in results:
        row_parts = []
        
        # Method name
        row_parts.append(f"{result['Method']:<{method_width}}")
        
        # KV Budget
        row_parts.append(f"{result['KVBudget']:<{budget_width}}")
        
        # Dataset scores
        for dataset_name in DATASET_NAMES:
            value = result[dataset_name]
            if value == "":
                display_value = "N/A"
            else:
                display_value = f"{value:.2f}" if isinstance(value, (int, float)) else str(value)
            row_parts.append(f"{display_value:<{score_width}}")
        
        # Average
        avg_value = result["Avg"]
        if avg_value == "":
            avg_display = "N/A"
        else:
            avg_display = f"{avg_value:.2f}" if isinstance(avg_value, (int, float)) else str(avg_value)
        row_parts.append(f"{avg_display:<{score_width}}")
        
        row = " ".join(row_parts)
        print(row)
    
    print("="*120)

def main():
    """Main function to run the aggregation."""
    # Require base path as command line argument
    if len(sys.argv) < 2:
        print("Usage: python3 aggregate_longbench.py <base_path>")
        print("Example: python3 aggregate_longbench.py outputs/meta-llama/Llama-3.1-8B-Instruct/longbench")
        sys.exit(1)
    
    base_path = sys.argv[1]
    
    print(f"Aggregating LongBench results from: {base_path}")
    print()
    
    # Aggregate results
    results = aggregate_longbench_results(base_path)
    
    if results:
        # Print summary table
        print_summary_table(results)
        
        # Save to CSV
        save_to_csv(results, base_path)
        
        print(f"\nProcessed {len(results)} methods across {len(DATASET_KEYS)} datasets.")
    else:
        print("No results found!")

if __name__ == "__main__":
    main()
