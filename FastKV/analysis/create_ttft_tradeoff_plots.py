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

def load_longbench_results():
    """Load hardcoded LongBench accuracy results."""
    # Hardcoded accuracy results
    accuracy_data = {
        "fullkv": {1.0: 49.32},
        "oracle": {0.1: 47.83, 0.2: 48.39, 0.4: 48.88},
        "gemfilter": {0.1: 37.59, 0.2: 42.29, 0.4: 45.11},
        "fastkv": {0.1: 46.81, 0.2: 47.33, 0.4: 47.68},
        "speculative_prefill": {0.1: 41.99, 0.2: 44.57, 0.4: 46.55},
        "claa": {0.1: 47.13, 0.2: 48.12, 0.4: 48.72}
    }
    
    # Convert to expected format
    results = defaultdict(dict)
    print("Loading hardcoded LongBench results...")
    
    for method, keep_rates in accuracy_data.items():
        for keep_rate, accuracy in keep_rates.items():
            results[method][keep_rate] = {"avg_score": accuracy, "scores": {}}
            print(f"  Loaded {method} {keep_rate*100:.0f}%: {accuracy:.2f}")
    
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
    """Create accuracy vs TTFT plot."""
    set_publication_style()
    plt.rcParams.update({
        'axes.labelsize': 26,
        'lines.linewidth': 3,
    })
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nGenerating accuracy vs TTFT plot in {output_dir}...")
    print(f"Data summary:")
    print(df[['method_display', 'keep_rate_percent', 'ttft_ms']])
    
    # Create single plot
    fig, ax = plt.subplots(1, 1, figsize=(7, 6), dpi=300)
    
    for method in df['method_display'].unique():
        method_data = df[df['method_display'] == method]
        color = METHOD_COLORS[method]
        marker = METHOD_MARKERS[method]
        
        # Plot accuracy vs TTFT
        ax.scatter(method_data['ttft_ms'], method_data['avg_accuracy'], 
                   c=color, marker=marker, s=100, alpha=1.0, label=method, 
                   edgecolors='white', linewidth=1)
        
        # Connect points for each method (except FullKV)
        if method != "FullKV" and len(method_data) > 1:
            sorted_data = method_data.sort_values('keep_rate')
            
            ax.plot(sorted_data['ttft_ms'], sorted_data['avg_accuracy'], 
                    color=color, alpha=1.0, linewidth=2)
        
        # Add keep rate annotations only for Oracle points (upper left)
        if method == "Oracle":
            for _, row in method_data.iterrows():
                ax.annotate(f"{row['keep_rate_percent']:.0f}%", 
                           (row['ttft_ms'], row['avg_accuracy']), 
                           xytext=(-15, 8), textcoords='offset points', 
                           fontsize=12, color='black', fontweight='bold')
        
        # Add 100% annotation directly under FullKV point
        if method == "FullKV":
            for _, row in method_data.iterrows():
                ax.annotate("100%", 
                           (row['ttft_ms'], row['avg_accuracy']), 
                           xytext=(-5, -15), textcoords='offset points', 
                           fontsize=12, color='black', fontweight='bold', ha='center')
        
    
    # Format the plot
    ax.set_xlabel('Time to First Token (ms)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('LongBench Accuracy vs TTFT', fontsize=24)
    ax.grid(True, which='major', linestyle=':', linewidth=0.6)
    
    # Create custom legend with lines through markers in specified order
    legend_order = ["FullKV", "Oracle", "GemFilter", "FastKV", "SpecPrefill", "CLAA"]
    legend_elements = []
    
    for method in legend_order:
        if method not in df['method_display'].unique():
            continue
            
        color = METHOD_COLORS[method]
        marker = METHOD_MARKERS[method]
        
        # Create line with marker for legend
        if method == "FullKV":
            # FullKV only has one point, so just marker
            legend_elements.append(plt.Line2D([0], [0], marker=marker, color=color, 
                                            markerfacecolor=color, markersize=8,
                                            linestyle='None', label=method,
                                            markeredgecolor='white', markeredgewidth=1))
        else:
            # Other methods have connected points, so line + marker
            legend_elements.append(plt.Line2D([0], [0], marker=marker, color=color,
                                            markerfacecolor=color, markersize=8,
                                            linestyle='-', linewidth=2, alpha=1.0,
                                            label=method, markeredgecolor='white', 
                                            markeredgewidth=1))
    
    # Place legend in lower right
    ax.legend(handles=legend_elements, loc='lower right', fontsize=18, 
              frameon=True, facecolor='white', framealpha=0.9)
    
    plt.tight_layout()
    
    # Save accuracy vs TTFT plot
    plt.savefig(f"{output_dir}/ttft_accuracy_analysis.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/ttft_accuracy_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("  Generated: ttft_accuracy_analysis.pdf/png")

def generate_debug_data():
    """Generate dummy data for testing visualization."""
    import pandas as pd
    
    data = []
    # FullKV baseline (100%) - realistic values
    data.append({
        "method_display": "FullKV", "keep_rate": 1.0, "keep_rate_percent": 100,
        "avg_accuracy": 49.32, "ttft_ms": 900.0, "memory_gb": 18.5
    })
    
    # Other methods with multiple keep rates (realistic memory values)
    methods_data = {
        "FastKV": [(0.1, 46.81, 500, 16.2), (0.2, 47.33, 580, 16.35), (0.4, 47.68, 670, 16.48)],
        "GemFilter": [(0.1, 37.59, 520, 16.1), (0.2, 42.29, 600, 16.25), (0.4, 45.11, 690, 16.4)], 
        "CLAA": [(0.1, 47.13, 510, 16.15), (0.2, 48.12, 590, 16.3), (0.4, 48.72, 680, 16.45)],
        "Oracle": [(0.1, 47.83, 505, 16.0), (0.2, 48.39, 585, 16.2), (0.4, 48.88, 675, 16.35)],
        "SpecPrefill": [(0.1, 41.99, 530, 24.8), (0.2, 44.57, 610, 25.2), (0.4, 46.55, 700, 25.7)]  # Outlier memory usage
    }
    
    for method, configs in methods_data.items():
        for keep_rate, acc, ttft, mem in configs:
            data.append({
                "method_display": method, "keep_rate": keep_rate, 
                "keep_rate_percent": keep_rate * 100,
                "avg_accuracy": acc, "ttft_ms": ttft, "memory_gb": mem
            })
    
    return pd.DataFrame(data)

def main():
    parser = argparse.ArgumentParser(description="Create TTFT vs accuracy tradeoff plots")
    parser.add_argument("--longbench_path", type=str, 
                       help="Path to LongBench results directory")
    parser.add_argument("--ttft_data", type=str,
                       help="Path to TTFT results (JSON file or directory)")
    parser.add_argument("--output_dir", type=str, default="figures",
                       help="Output directory for plots")
    parser.add_argument("--debug", action="store_true",
                       help="Use dummy data for testing")
    
    args = parser.parse_args()
    
    print("TTFT Tradeoff Analysis")
    print("=" * 30)
    
    if args.debug:
        print("Using debug mode with dummy data...")
        df = generate_debug_data()
        print(f"Generated {len(df)} dummy data points")
    else:
        if not args.ttft_data:
            print("Error: --ttft_data required (or use --debug)")
            return
            
        # Load data
        longbench_results = load_longbench_results()
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
    
    print(f"\nAnalysis complete! Check {args.output_dir}/ for results")

if __name__ == "__main__":
    main()
