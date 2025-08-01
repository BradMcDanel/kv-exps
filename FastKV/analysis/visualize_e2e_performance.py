#!/usr/bin/env python3
"""
Visualization script for E2E (prefill + decode) performance analysis.
Shows throughput vs memory usage and KV cache size comparisons.
"""

import os
import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from .viz_utils import set_publication_style, METHOD_COLORS

# Method display names and markers
METHOD_DISPLAY_NAMES = {
    "fullkv": "FullKV", "fastkv": "FastKV", "gemfilter": "GemFilter",
    "speculative_prefill": "SpecPrefill", "oracle": "Oracle", "claa": "CLAA"
}

METHOD_MARKERS = {
    "FullKV": "s", "Oracle": "^", "FastKV": "o", 
    "CLAA": "D", "GemFilter": "v", "SpecPrefill": "P"
}

def generate_dummy_data():
    """Generate dummy data for testing visualization."""
    data = []
    
    # FullKV baseline
    data.append({
        "method": "fullkv", "keep_rate": 1.0, "min_layer_idx": 0,
        "total_throughput_tps": 15.5, "max_memory_gb": 18.5, "kv_cache_size_gb": 4.2,
        "seqlen": 4000, "num_decode_steps": 32
    })
    
    # Other methods with multiple configurations
    methods_configs = {
        "fastkv": [
            (0.1, 0, 45.2, 16.2, 0.42), (0.2, 0, 42.1, 16.35, 0.84), (0.4, 0, 38.7, 16.48, 1.68),
            (0.1, 5, 43.8, 16.1, 0.35), (0.2, 5, 40.5, 16.25, 0.70), (0.4, 5, 37.2, 16.38, 1.40)
        ],
        "claa": [
            (0.1, 0, 44.1, 16.15, 0.41), (0.2, 0, 41.2, 16.3, 0.82), (0.4, 0, 38.1, 16.45, 1.64),
            (0.1, 5, 42.9, 16.05, 0.34), (0.2, 5, 39.8, 16.2, 0.68), (0.4, 5, 36.5, 16.35, 1.36)
        ],
        "oracle": [
            (0.1, 0, 46.8, 16.0, 0.42), (0.2, 0, 43.9, 16.2, 0.84), (0.4, 0, 40.1, 16.35, 1.68)
        ],
        "gemfilter": [
            (0.1, 0, 35.1, 16.1, 0.41), (0.2, 0, 32.8, 16.25, 0.82), (0.4, 0, 30.2, 16.4, 1.64)
        ]
    }
    
    for method, configs in methods_configs.items():
        for keep_rate, min_layer, throughput, memory, kv_cache in configs:
            data.append({
                "method": method, "keep_rate": keep_rate, "min_layer_idx": min_layer,
                "total_throughput_tps": throughput, "max_memory_gb": memory, "kv_cache_size_gb": kv_cache,
                "seqlen": 4000, "num_decode_steps": 32
            })
    
    return data

def load_e2e_results(results_path):
    """Load E2E benchmark results from JSON files."""
    results_path = Path(results_path)
    results = []
    
    if results_path.is_file() and results_path.suffix == '.json':
        with open(results_path, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                results.extend(data)
            else:
                results.append(data)
    elif results_path.is_dir():
        for json_file in results_path.glob("*_e2e*.json"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        results.extend(data)
                    else:
                        results.append(data)
            except Exception as e:
                print(f"Warning: Could not load {json_file}: {e}")
    else:
        raise ValueError(f"E2E results path must be a JSON file or directory: {results_path}")
    
    return results

def process_results_to_dataframe(results):
    """Convert results list to DataFrame with proper formatting."""
    processed_data = []
    
    for result in results:
        method = result.get("method")
        method_display = METHOD_DISPLAY_NAMES.get(method, method)
        keep_rate = result.get("keep_rate", 1.0)
        
        # Handle None keep_rate for fullkv
        if keep_rate is None or method == "fullkv":
            keep_rate = 1.0
        
        row = {
            "method": method,
            "method_display": method_display,
            "keep_rate": keep_rate,
            "keep_rate_percent": keep_rate * 100,
            "min_layer_idx": result.get("min_layer_idx", 0),
            "total_throughput_tps": result.get("total_throughput_tps", 0),
            "decode_throughput_tps": result.get("decode_throughput_tps", 0),
            "max_memory_gb": result.get("max_memory_gb", 0),
            "kv_cache_size_gb": result.get("kv_cache_size_gb", 0),
            "ttft_ms": result.get("ttft_ms", 0),
            "e2e_time_ms": result.get("e2e_time_ms", 0),
            "seqlen": result.get("seqlen", 0),
            "num_decode_steps": result.get("num_decode_steps", 0),
        }
        processed_data.append(row)
    
    return pd.DataFrame(processed_data)

def plot_throughput_vs_memory(df: pd.DataFrame, output_prefix: str):
    """Create throughput vs memory usage plot."""
    set_publication_style()
    plt.rcParams.update({'axes.labelsize': 26})
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=300)
    
    # Plot order for legend
    legend_order = ["FullKV", "Oracle", "GemFilter", "FastKV", "SpecPrefill", "CLAA"]
    legend_elements = []
    
    for method in legend_order:
        if method not in df['method_display'].unique():
            continue
            
        method_data = df[df['method_display'] == method]
        color = METHOD_COLORS[method]
        marker = METHOD_MARKERS[method]
        
        # Plot throughput vs memory
        ax.scatter(method_data['max_memory_gb'], method_data['total_throughput_tps'], 
                   c=color, marker=marker, s=100, alpha=0.8, label=method, 
                   edgecolors='white', linewidth=1)
        
        # Connect points for each method (except FullKV)
        if method != "FullKV" and len(method_data) > 1:
            sorted_data = method_data.sort_values('keep_rate')
            ax.plot(sorted_data['max_memory_gb'], sorted_data['total_throughput_tps'], 
                    color=color, alpha=0.3, linewidth=2, linestyle='--')
        
        # Add keep rate annotations (only for non-FullKV)
        for _, row in method_data.iterrows():
            if method != "FullKV":
                ax.annotate(f"{row['keep_rate_percent']:.0f}%", 
                           (row['max_memory_gb'], row['total_throughput_tps']), 
                           xytext=(5, 5), textcoords='offset points', 
                           fontsize=9, alpha=0.7)
        
        # Create legend elements
        if method == "FullKV":
            legend_elements.append(plt.Line2D([0], [0], marker=marker, color=color, 
                                            markerfacecolor=color, markersize=8,
                                            linestyle='None', label=method,
                                            markeredgecolor='white', markeredgewidth=1))
        else:
            legend_elements.append(plt.Line2D([0], [0], marker=marker, color=color,
                                            markerfacecolor=color, markersize=8,
                                            linestyle='--', linewidth=2, alpha=1.0,
                                            label=method, markeredgecolor='white', 
                                            markeredgewidth=1))
    
    ax.set_xlabel('Peak Memory Usage (GB)')
    ax.set_ylabel('Total Throughput (tokens/s)')
    ax.set_title('E2E Throughput vs Memory Usage')
    ax.grid(True, which='major', linestyle=':', linewidth=0.6)
    
    ax.legend(handles=legend_elements, loc='best', fontsize=18, 
              frameon=True, facecolor='white', framealpha=0.9)
    
    plt.tight_layout()
    
    plt.savefig(f"{output_prefix}.pdf", format='pdf', dpi=300, bbox_inches='tight')
    print(f"Saved throughput vs memory plot to: {output_prefix}.pdf")
    plt.savefig(f"{output_prefix}.png", format='png', dpi=300, bbox_inches='tight')
    print(f"Saved throughput vs memory plot to: {output_prefix}.png")
    plt.close(fig)

def plot_throughput_vs_kv_cache(df: pd.DataFrame, output_prefix: str):
    """Create throughput vs KV cache size plot."""
    set_publication_style()
    plt.rcParams.update({'axes.labelsize': 26})
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=300)
    
    # Plot order for legend
    legend_order = ["FullKV", "Oracle", "GemFilter", "FastKV", "SpecPrefill", "CLAA"]
    legend_elements = []
    
    for method in legend_order:
        if method not in df['method_display'].unique():
            continue
            
        method_data = df[df['method_display'] == method]
        color = METHOD_COLORS[method]
        marker = METHOD_MARKERS[method]
        
        # Plot KV cache size vs throughput
        ax.scatter(method_data['kv_cache_size_gb'], method_data['total_throughput_tps'], 
                   c=color, marker=marker, s=100, alpha=0.8, label=method, 
                   edgecolors='white', linewidth=1)
        
        # Connect points for each method (except FullKV)
        if method != "FullKV" and len(method_data) > 1:
            sorted_data = method_data.sort_values('keep_rate')
            ax.plot(sorted_data['kv_cache_size_gb'], sorted_data['total_throughput_tps'], 
                    color=color, alpha=0.3, linewidth=2, linestyle='--')
        
        # Add keep rate annotations (only for non-FullKV)
        for _, row in method_data.iterrows():
            if method != "FullKV":
                ax.annotate(f"{row['keep_rate_percent']:.0f}%", 
                           (row['kv_cache_size_gb'], row['total_throughput_tps']), 
                           xytext=(5, 5), textcoords='offset points', 
                           fontsize=9, alpha=0.7)
        
        # Create legend elements
        if method == "FullKV":
            legend_elements.append(plt.Line2D([0], [0], marker=marker, color=color, 
                                            markerfacecolor=color, markersize=8,
                                            linestyle='None', label=method,
                                            markeredgecolor='white', markeredgewidth=1))
        else:
            legend_elements.append(plt.Line2D([0], [0], marker=marker, color=color,
                                            markerfacecolor=color, markersize=8,
                                            linestyle='--', linewidth=2, alpha=1.0,
                                            label=method, markeredgecolor='white', 
                                            markeredgewidth=1))
    
    ax.set_xlabel('KV Cache Size (GB)')
    ax.set_ylabel('Total Throughput (tokens/s)')
    ax.set_title('Throughput vs KV Cache Size')
    ax.grid(True, which='major', linestyle=':', linewidth=0.6)
    
    ax.legend(handles=legend_elements, loc='best', fontsize=18, 
              frameon=True, facecolor='white', framealpha=0.9)
    
    plt.tight_layout()
    
    plt.savefig(f"{output_prefix}.pdf", format='pdf', dpi=300, bbox_inches='tight')
    print(f"Saved throughput vs KV cache plot to: {output_prefix}.pdf")
    plt.savefig(f"{output_prefix}.png", format='png', dpi=300, bbox_inches='tight')
    print(f"Saved throughput vs KV cache plot to: {output_prefix}.png")
    plt.close(fig)

def plot_min_layer_analysis(df: pd.DataFrame, output_prefix: str):
    """Create min_layer_idx impact analysis plot."""
    if df['min_layer_idx'].nunique() <= 1:
        print("Skipping min_layer analysis plot - not enough variation in min_layer_idx")
        return
    
    set_publication_style()
    plt.rcParams.update({'axes.labelsize': 26})
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=300)
    
    # Filter to methods that have min_layer variation
    methods_with_variation = []
    for method in df['method_display'].unique():
        method_data = df[df['method_display'] == method]
        if method_data['min_layer_idx'].nunique() > 1:
            methods_with_variation.append(method)
    
    for method in methods_with_variation:
        method_data = df[df['method_display'] == method]
        color = METHOD_COLORS.get(method, '#000000')
        marker = METHOD_MARKERS.get(method, 'o')
        
        # Group by keep_rate to show min_layer impact
        for keep_rate in method_data['keep_rate'].unique():
            rate_data = method_data[method_data['keep_rate'] == keep_rate]
            if len(rate_data) > 1:
                rate_data = rate_data.sort_values('min_layer_idx')
                
                label = f"{method} ({keep_rate:.0%})" if keep_rate < 1.0 else f"{method}"
                
                # Plot 1: Throughput vs min_layer_idx
                ax1.plot(rate_data['min_layer_idx'], rate_data['total_throughput_tps'],
                        color=color, marker=marker, markersize=8, alpha=0.8,
                        label=label, markeredgecolor='white', markeredgewidth=1)
                
                # Plot 2: Memory vs min_layer_idx
                ax2.plot(rate_data['min_layer_idx'], rate_data['max_memory_gb'],
                        color=color, marker=marker, markersize=8, alpha=0.8,
                        label=label, markeredgecolor='white', markeredgewidth=1)
    
    # Format left subplot
    ax1.set_xlabel('Min Layer Index')
    ax1.set_ylabel('Total Throughput (tokens/s)')
    ax1.set_title('Throughput vs Min Layer Index')
    ax1.grid(True, which='major', linestyle=':', linewidth=0.6)
    ax1.legend(fontsize=14)
    
    # Format right subplot
    ax2.set_xlabel('Min Layer Index')
    ax2.set_ylabel('Peak Memory Usage (GB)')
    ax2.set_title('Memory vs Min Layer Index')
    ax2.grid(True, which='major', linestyle=':', linewidth=0.6)
    ax2.legend(fontsize=14)
    
    plt.tight_layout()
    
    plt.savefig(f"{output_prefix}.pdf", format='pdf', dpi=300, bbox_inches='tight')
    print(f"Saved min layer analysis to: {output_prefix}.pdf")
    plt.savefig(f"{output_prefix}.png", format='png', dpi=300, bbox_inches='tight')
    print(f"Saved min layer analysis to: {output_prefix}.png")
    plt.close(fig)

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Visualize E2E benchmark performance")
    parser.add_argument("--results_path", type=str, 
                       help="Path to E2E results (JSON file or directory)")
    parser.add_argument("--debug", action="store_true",
                       help="Use dummy data for testing")
    
    args = parser.parse_args()
    
    print("Creating E2E performance visualizations...")
    
    # Create output directory
    output_dir = "figures"
    os.makedirs(output_dir, exist_ok=True)
    
    if args.debug:
        print("--- Using --debug mode: Generating random dummy data ---")
        results = generate_dummy_data()
        df = process_results_to_dataframe(results)
        print(f"Generated {len(df)} dummy data points")
    else:
        if not args.results_path:
            print("Error: --results_path required (or use --debug)")
            return
            
        # Load data
        results = load_e2e_results(args.results_path)
        print(f"Loaded {len(results)} E2E benchmark results")
        
        if not results:
            print("Error: No results found!")
            return
        
        df = process_results_to_dataframe(results)
        print(f"Processed {len(df)} data points")
    
    print(f"Methods found: {list(df['method_display'].unique())}")
    print(f"Keep rates: {sorted(df['keep_rate'].unique())}")
    print(f"Min layer indices: {sorted(df['min_layer_idx'].unique())}")
    
    # Generate plots
    plot_throughput_vs_memory(df, os.path.join(output_dir, "e2e_throughput_vs_memory"))
    plot_throughput_vs_kv_cache(df, os.path.join(output_dir, "e2e_throughput_vs_kv_cache"))
    plot_min_layer_analysis(df, os.path.join(output_dir, "e2e_min_layer_analysis"))
    
    # Save combined data
    csv_file = os.path.join(output_dir, "e2e_combined.csv")
    df.to_csv(csv_file, index=False)
    print(f"Saved combined data to: {csv_file}")
    
    print("E2E performance visualization completed!")

if __name__ == "__main__":
    main()