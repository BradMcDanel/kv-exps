#!/usr/bin/env python3

"""
Script to aggregate specific LongBench results and generate a LaTeX table.

This script reads results.json files for a predefined set of methods and
keep-rate percentages, groups them, and outputs a formatted LaTeX table to stdout.
All informational messages are printed to stderr.
"""

import os
import json
import sys
from collections import defaultdict

# --- Configuration Constants ---

MODEL_NAME = "LLaMA-3.1-8B-Instruct"

# Hardcoded TSP layer for methods that require it ***
TSP_LAYER = 0

METHOD_ROOTS = ["fullkv", "oracle", "fastkv", "specprefill"]

METHOD_DISPLAY_NAMES = {
    "fullkv": "Full~KV",
    "fastkv": "FastKV~\\cite{jo2025fastkv}",
    "specprefill": "SpecPrefill~\\cite{liu2025speculative}",
    "oracle": "Oracle"
}

KEEP_RATES_PERCENT = [1, 2, 3, 4]

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
    ("passage_count", "PCount"),
    ("passage_retrieval_en", "PRe"),
    ("lcc", "LCC"),
    ("repobench-p", "RB-P"),
]

DATASET_KEYS = [key for key, _ in DATASETS]
DATASET_NAMES = [name for _, name in DATASETS]

CATEGORIES = {
    "Single-Document QA": 3,
    "Multi-Document QA": 3,
    "Summarization": 3,
    "Few-shot Learning": 3,
    "Synthetic": 2,
    "Code": 2,
}

# --- Helper Functions ---

def log_message(msg):
    """Prints a message to stderr to avoid polluting stdout."""
    print(msg, file=sys.stderr)

def load_results(results_file):
    """Load results from a JSON file, return empty dict if file doesn't exist or is invalid."""
    try:
        with open(results_file, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, IOError) as e:
        log_message(f"Warning: Could not load {results_file}: {e}")
        return {}

# --- Core Logic ---

def aggregate_results(base_path):
    if not os.path.exists(base_path):
        log_message(f"Error: Base path {base_path} does not exist!")
        return {}

    grouped_results = defaultdict(list)
    
    def process_folder(folder_name, method_root, keep_rate):
        method_path = os.path.join(base_path, folder_name)
        if not os.path.isdir(method_path):
            return

        log_message(f"  Processing folder: {folder_name}")
        results_file = os.path.join(method_path, "results.json")
        method_scores = load_results(results_file)
        
        if not method_scores:
            log_message(f"  Skipping {folder_name}, no results found.")
            return

        row = {
            "method_root": method_root,
            "method_name": METHOD_DISPLAY_NAMES.get(method_root, method_root),
            "keep_rate": keep_rate,
            "scores": {}
        }
        
        valid_scores = []
        for dataset_key, _ in DATASETS:
            score = method_scores.get(dataset_key)
            row["scores"][dataset_key] = score
            if isinstance(score, (int, float)):
                valid_scores.append(score)
        
        row["avg"] = (sum(valid_scores) / len(valid_scores)) if valid_scores else None
        
        grouped_results[keep_rate].append(row)

    if "fullkv" in METHOD_ROOTS:
        process_folder("fullkv", "fullkv", 100)
        
    for rate in KEEP_RATES_PERCENT:
        for method_root in METHOD_ROOTS:
            if method_root == "fullkv":
                continue
            
            if method_root in ["fastkv", "oracle"]:
                folder_name = f"{method_root}-l{TSP_LAYER}-{rate}p"
            else:
                folder_name = f"{method_root}-l{TSP_LAYER}-{rate}p"
            
            process_folder(folder_name, method_root, rate)

    for rate in grouped_results:
        grouped_results[rate].sort(key=lambda x: METHOD_ROOTS.index(x['method_root']))
        
    return grouped_results

def generate_latex_table(grouped_results):
    if not grouped_results:
        log_message("No results to generate table from.")
        return

    print(r"\begin{table*}[ht]")
    print(r"  \centering")
    print(r"  \setlength{\tabcolsep}{1pt}")
    print(rf"  \caption{{{MODEL_NAME} LongBench results. We vary the percentage of KV cache tokens to keep (Keep Rate).}}")
    print(r"  \label{tab:longbench_results_generated}")
    print(r"  \renewcommand{\arraystretch}{1.1}")
    print(r"  \scalebox{0.70}{")
    
    tabular_cols = "l|" + "|".join(f"*{{{count}}}{{W}}" for count in CATEGORIES.values()) + "|W"
    print(rf"    \begin{{tabular}}{{{tabular_cols}}}")
    print(r"      \toprule")

    category_headers = " & ".join(rf"\multicolumn{{{count}}}{{c|}}{{{name}}}" for name, count in CATEGORIES.items())
    print(rf"        & {category_headers} &     \\")

    start_col = 2
    cmidrules = []
    for count in CATEGORIES.values():
        end_col = start_col + count - 1
        cmidrules.append(rf"\cmidrule(lr){{{start_col}-{end_col}}}")
        start_col = end_col + 1
    print("      " + " ".join(cmidrules))
    
    rotated_headers = " & ".join(rf"\rotatebox{{60}}{{{name}}}" for name in DATASET_NAMES)
    print(rf"      Method & {rotated_headers} & \rotatebox{{60}}{{Avg.}} \\")
    
    print(r"      \midrule")
    print(r"      \midrule")

    sorted_keep_rates = sorted(grouped_results.keys(), key=lambda x: (x != 100, x))
    
    for i, rate in enumerate(sorted_keep_rates):
        results_for_rate = grouped_results[rate]
        
        valid_avgs = [res['avg'] for res in results_for_rate if res['avg'] is not None]
        max_avg = max(valid_avgs) if valid_avgs else None
        
        rate_display = "100\\%" if rate == 100 else f"{rate * 10}\\%"
        print(rf"\multicolumn{{18}}{{c}}{{\textbf{{Keep Token Rate = {rate_display}}}}} \\")
        print(r"\midrule")
        
        for res in results_for_rate:
            method_cell = f"{res['method_name']:<9}"
            score_cells = []
            for key in DATASET_KEYS:
                score = res['scores'].get(key)
                score_str = f"{score:.2f}" if isinstance(score, (int, float)) else "   -"
                score_cells.append(f"{score_str:>6}")
            
            avg_score = res['avg']
            if avg_score is not None:
                avg_str = f"{avg_score:.2f}"
                if max_avg is not None and abs(avg_score - max_avg) < 1e-5:
                    avg_str = rf"\textbf{{{avg_str}}}"
            else:
                avg_str = "   -"
            

            line1 = f"{method_cell} & " + " & ".join(score_cells[0:6])
            line2 = "          & " + " & ".join(score_cells[6:12])
            line3 = "          & " + " & ".join(score_cells[12:16]) + f" & {avg_str} \\\\"
            
            print(line1)
            print(line2)
            print(line3)

        if i < len(sorted_keep_rates) - 1:
            print(r"\midrule")

    print(r"      \bottomrule")
    print(r"    \end{tabular}")
    print(r"  }")
    print(r"\end{table*}")

def main():
    if len(sys.argv) < 2:
        log_message("Usage: python3 generate_latex_table.py <base_path>")
        log_message("Example: python3 generate_latex_table.py outputs/meta-llama/Llama-3.1-8B-Instruct/longbench")
        sys.exit(1)
        
    base_path = sys.argv[1]
    
    log_message(f"Aggregating LongBench results from: {base_path}")
    log_message(f"Model: {MODEL_NAME}")
    log_message(f"Target methods: {METHOD_ROOTS}")
    log_message(f"Target keep rates (%): {KEEP_RATES_PERCENT}")
    log_message("-" * 30)

    grouped_results = aggregate_results(base_path)
    
    if grouped_results:
        generate_latex_table(grouped_results)
        log_message("-" * 30)
        log_message("LaTeX table generated successfully to standard output.")
    else:
        log_message("No results found for the specified methods and keep rates.")

if __name__ == "__main__":
    main()
