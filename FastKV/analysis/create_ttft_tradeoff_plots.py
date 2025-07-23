#!/usr/bin/env python3
# analysis/create_ttft_tradeoff_plots.py

import os
import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict

try:
    from .viz_utils import set_publication_style, METHOD_COLORS
except ImportError:
    import sys
    sys.path.append('.')
    from analysis.viz_utils import set_publication_style, METHOD_COLORS

# Configuration
METHOD_DISPLAY_NAMES = {
    "fullkv": "FullKV", "fastkv": "FastKV", "gemfilter": "GemFilter",
    "speculative_prefill": "SpecPrefill", "oracle": "Oracle", "claa": "CLAA"
}

METHOD_MARKERS = {
    "FullKV": "s", "Oracle": "^", "FastKV": "o", 
    "CLAA": "D", "GemFilter": "v", "SpecPrefill": "P"
}

KEEP_RATES_DECIMAL = [0.1, 0.2, 0.4]
TSP_LAYER = 15

def load_longbench_results(longbench_path):
    """Load LongBench accuracy results."""
    results = defaultdict(dict)
    
    def process_folder(folder_name, method_root, keep_rate):
        results_file = os.path.join(longbench_path, folder_name, "results.json")
        try:
            with open(results_file, 'r') as f:
                method_scores = json.load(f)
                valid_scores = [v for v in method_scores.values() if isinstance(v, (int, float))]
                avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else None
                results[method_root][keep_rate] = {"avg_score": avg_score, "scores": method_scores}
                print(f"  Loaded {method_root} {keep_rate*100:.0f}%: {avg_score:.2f}")
        except Exception as e:
            print(f"  Could not load {folder_name}: {e}")
    
    print("Loading LongBench results...")
    
    # Process fullkv
    process_folder("fullkv", "fullkv", 1.0)
    
    # Process other methods
    for rate in KEEP_RATES_DECIMAL:
        for method_root in ["fastkv", "gemfilter", "speculative_prefill", "claa", "oracle"]:
            if method_root == "speculative_prefill":
                folder_name = f"specprefill_{rate}p"
            else:
                folder_name = f"{method_root}_l{TSP_LAYER}_{rate}p"
            process_folder(folder_name, method_root, rate)
    
    return results

def load_ttft_results(ttft_path):
    """Load TTFT results from JSON file or directory."""
    ttft_path = Path(ttft_path)
    
    if ttft_path.is_file() and ttft_path.suffix == '.json':
        # Single JSON file
        with open(ttft_path, 'r') as f:
            return json.load(f)
    elif ttft_path.is_dir():
        # Directory with individual files
        results = {}
        for json_file in ttft_path.glob("*_ttft.json"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    key = f"{data['method']}_{data['keep_rate']}"
                    results[key] = data
            except Exception as e:
                print(f"Warning: Could not load {json_file}: {e}")
        return results
    else:
        raise ValueError(f"TTFT path must be a JSON file or directory: {ttft_path}")

def combine_data(longbench_results, ttft_results):
    """Combine accuracy and TTFT data into DataFrame."""
    combined_data = []
    
    print("\nCombining data...")
    
    for method_key, ttft_data in ttft_results.items():
        method = ttft_data.get("method")
        keep_rate = ttft_data.get("keep_rate")
        
        if method not in longbench_results or keep_rate not in longbench_results[method]:
            print(f"  No LongBench data for {method} {keep_rate*100:.0f}%")
            continue
        
        lb_data = longbench_results[method][keep_rate]
        if lb_data["avg_score"] is None:
            print(f"  Invalid accuracy for {method} {keep_rate*100:.0f}%")
            continue
        
        method_display = METHOD_DISPLAY_NAMES.get(method, method)
        
        row = {
            "method_display": method_display,
            "keep_rate": keep_rate,
            "keep_rate_percent": keep_rate * 100,
            "avg_accuracy": lb_data["avg_score"],
            "ttft_ms": ttft_data.get("ttft_ms"),
            "memory_gb": ttft_data.get("memory_gb")
        }
        
        combined_data.append(row)
        print(f"  Combined {method_display} {keep_rate*100:.0f}%: {row['avg_accuracy']:.2f}%, {row['ttft_ms']:.1f}ms")
    
    return pd.DataFrame(combined_data)

def create_tradeoff_plots(df, output_dir):
    """Create accuracy vs TTFT and memory vs TTFT plots."""
    set_publication_style()
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nGenerating plots in {output_dir}...")
    
    # Accuracy vs TTFT plot
    fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
    
    for method in df['method_display'].unique():
        method_data = df[df['method_display'] == method]
        color = METHOD_COLORS.get(method, "#333333")
        marker = METHOD_MARKERS.get(method, "o")
        
        ax.scatter(method_data['ttft_ms'], method_data['avg_accuracy'], 
                  c=color, marker=marker, s=100, alpha=0.8, label=method, 
                  edgecolors='white', linewidth=1)
        
        # Connect points for each method (except FullKV)
        if method != "FullKV" and len(method_data) > 1:
            sorted_data = method_data.sort_values('keep_rate')
            ax.plot(sorted_data['ttft_ms'], sorted_data['avg_accuracy'], 
                   color=color, alpha=0.3, linewidth=1, linestyle='--')
        
        # Add keep rate annotations
        for _, row in method_data.iterrows():
            if method != "FullKV":
                ax.annotate(f"{row['keep_rate_percent']:.0f}%", 
                           (row['ttft_ms'], row['avg_accuracy']), 
                           xytext=(5, 5), textcoords='offset points', 
                           fontsize=8, alpha=0.7)
    
    ax.set_xlabel('Time to First Token (ms)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average LongBench Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Accuracy vs TTFT Tradeoff', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/accuracy_vs_ttft.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/accuracy_vs_ttft.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Memory vs TTFT plot
    fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
    
    for method in df['method_display'].unique():
        method_data = df[df['method_display'] == method]
        color = METHOD_COLORS.get(method, "#333333")
        marker = METHOD_MARKERS.get(method, "o")
        
        ax.scatter(method_data['ttft_ms'], method_data['memory_gb'], 
                  c=color, marker=marker, s=100, alpha=0.8, label=method,
                  edgecolors='white', linewidth=1)
        
        if method != "FullKV" and len(method_data) > 1:
            sorted_data = method_data.sort_values('keep_rate')
            ax.plot(sorted_data['ttft_ms'], sorted_data['memory_gb'], 
                   color=color, alpha=0.3, linewidth=1, linestyle='--')
    
    ax.set_xlabel('Time to First Token (ms)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Peak Memory Usage (GB)', fontsize=12, fontweight='bold') 
    ax.set_title('Memory vs TTFT Tradeoff', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/memory_vs_ttft.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/memory_vs_ttft.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("  Generated: accuracy_vs_ttft.pdf/png")
    print("  Generated: memory_vs_ttft.pdf/png")

def main():
    parser = argparse.ArgumentParser(description="Create TTFT vs accuracy tradeoff plots")
    parser.add_argument("--longbench_path", type=str, required=True,
                       help="Path to LongBench results directory")
    parser.add_argument("--ttft_data", type=str, required=True,
                       help="Path to TTFT results (JSON file or directory)")
    parser.add_argument("--output_dir", type=str, default="figures",
                       help="Output directory for plots")
    
    args = parser.parse_args()
    
    print("TTFT Tradeoff Analysis")
    print("=" * 30)
    
    # Load data
    longbench_results = load_longbench_results(args.longbench_path)
    ttft_results = load_ttft_results(args.ttft_data)
    
    print(f"\nFound {len(longbench_results)} methods in LongBench results")
    print(f"Found {len(ttft_results)} configurations in TTFT results")
    
    # Combine data
    df = combine_data(longbench_results, ttft_results)
    
    if df.empty:
        print("\nError: No matching data found!")
        return
    
    print(f"\nSuccessfully combined {len(df)} data points")
    
    # Create plots
    create_tradeoff_plots(df, args.output_dir)
    
    # Save combined data
    csv_file = f"{args.output_dir}/ttft_accuracy_combined.csv"
    df.to_csv(csv_file, index=False)
    print(f"  Generated: {csv_file}")
    
    print(f"\nAnalysis complete! Check {args.output_dir}/ for results")

if __name__ == "__main__":
    main()