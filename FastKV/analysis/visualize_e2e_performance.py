#!/usr/bin/env python3
"""
E2E Performance Visualization with Stacked Bar Charts.
Shows prefill vs decode time breakdown with memory annotations and throughput metrics.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
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

KEEP_RATES = [0.1, 0.2, 0.4]  # 10%, 20%, 40%
METHODS_ORDER = ["FullKV", "Oracle", "FastKV", "CLAA", "GemFilter", "SpecPrefill"]

def load_e2e_results(results_dir):
    """Load E2E benchmark results from JSON files."""
    results_dir = Path(results_dir)
    results = defaultdict(dict)
    
    print(f"Loading E2E results from {results_dir}...")
    
    # Load all result files
    for json_file in results_dir.glob("*_e2e.json"):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                
            method = data.get("method")
            keep_rate = data.get("keep_rate")
            
            if method and keep_rate is not None:
                method_display = METHOD_DISPLAY_NAMES.get(method, method)
                results[method_display][keep_rate] = data
                print(f"  Loaded {method_display} {keep_rate*100:.0f}%: "
                      f"TTFT={data.get('ttft_ms', 0):.1f}ms, "
                      f"Decode={data.get('decode_time_ms', 0):.1f}ms, "
                      f"Throughput={data.get('total_throughput_tps', 0):.1f}tps, "
                      f"Memory={data.get('max_memory_gb', 0):.1f}GB")
                
        except Exception as e:
            print(f"  Warning: Could not load {json_file}: {e}")
    
    return results

def create_stacked_bar_chart(results, output_dir):
    """Create stacked bar chart showing prefill vs decode time breakdown."""
    set_publication_style()
    plt.rcParams.update({
        'axes.labelsize': 26,
        'lines.linewidth': 3,
    })
    
    # Create figure with subplots for each keep rate
    fig, axes = plt.subplots(1, 3, figsize=(18, 8), sharey=True, dpi=300)
    fig.suptitle('E2E Performance: Prefill vs Decode Time Breakdown', fontsize=32, fontweight='bold')
    
    bar_width = 0.6
    
    for idx, keep_rate in enumerate(KEEP_RATES):
        ax = axes[idx]
        
        # Collect data for this keep rate
        methods_data = []
        method_names = []
        
        for method in METHODS_ORDER:
            if method in results and keep_rate in results[method]:
                data = results[method][keep_rate]
                methods_data.append({
                    'method': method,
                    'ttft_ms': data.get('ttft_ms', 0),
                    'decode_time_ms': data.get('decode_time_ms', 0),
                    'total_throughput_tps': data.get('total_throughput_tps', 0),
                    'max_memory_gb': data.get('max_memory_gb', 0),
                    'kv_cache_size_gb': data.get('kv_cache_size_gb', 0)
                })
                method_names.append(method)
        
        if not methods_data:
            ax.text(0.5, 0.5, f'No data for {keep_rate*100:.0f}%', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=20)
            ax.set_title(f'{keep_rate*100:.0f}% Keep Rate', fontsize=28)
            continue
        
        # Extract data arrays
        ttft_times = [d['ttft_ms'] for d in methods_data]
        decode_times = [d['decode_time_ms'] for d in methods_data]
        throughputs = [d['total_throughput_tps'] for d in methods_data]
        memories = [d['max_memory_gb'] for d in methods_data]
        kv_caches = [d['kv_cache_size_gb'] for d in methods_data]
        
        # Create stacked bars
        x_pos = np.arange(len(method_names))
        
        # Bottom bars (prefill time) - solid colors
        bars1 = ax.bar(x_pos, ttft_times, bar_width, 
                      color=[METHOD_COLORS[method] for method in method_names], 
                      alpha=0.9, label='Prefill (TTFT)', linewidth=1, edgecolor='white')
        
        # Top bars (decode time) - lighter colors  
        bars2 = ax.bar(x_pos, decode_times, bar_width, bottom=ttft_times,
                      color=[METHOD_COLORS[method] for method in method_names], 
                      alpha=0.5, label='Decode', linewidth=1, edgecolor='white')
        
        # Add memory annotations inside bars
        for i, (method, data) in enumerate(zip(method_names, methods_data)):
            total_time = data['ttft_ms'] + data['decode_time_ms']
            memory_gb = data['max_memory_gb']
            kv_cache_gb = data['kv_cache_size_gb']
            throughput = data['total_throughput_tps']
            
            # Memory annotation in the middle of the bar
            if total_time > 0:
                ax.text(i, total_time/2, f'{memory_gb:.1f}GB\n{kv_cache_gb:.1f}GB KV', 
                       ha='center', va='center', fontweight='bold', fontsize=11,
                       color='white', bbox=dict(boxstyle='round,pad=0.3', 
                                              facecolor='black', alpha=0.8))
            
            # Throughput annotation above the bar
            if total_time > 0:
                ax.text(i, total_time + max([ttft_times[j] + decode_times[j] for j in range(len(method_names))])*0.05, 
                       f'{throughput:.0f}\ntps', 
                       ha='center', va='bottom', fontweight='bold', fontsize=12,
                       color=METHOD_COLORS[method])
        
        # Customize subplot to match other viz scripts
        ax.set_xlabel('Methods', fontsize=26)
        if idx == 0:
            ax.set_ylabel('Time (ms)', fontsize=26)
        ax.set_title(f'{keep_rate*100:.0f}% Keep Rate', fontsize=28, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(method_names, rotation=45, ha='right', fontsize=18)
        ax.grid(True, which='major', linestyle=':', linewidth=0.6)
        
        # Add legend only to first subplot, matching other scripts
        if idx == 0:
            ax.legend(loc='upper left', fontsize=18, frameon=True, 
                     facecolor='white', framealpha=0.9)
    
    plt.tight_layout()
    
    # Save the plot with same parameters as other viz scripts
    os.makedirs(output_dir, exist_ok=True)
    output_path_pdf = os.path.join(output_dir, "e2e_performance.pdf")
    output_path_png = os.path.join(output_dir, "e2e_performance.png")
    
    plt.savefig(output_path_pdf, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved stacked bar chart: {output_path_pdf}")
    print(f"Saved stacked bar chart: {output_path_png}")

def create_summary_table(results, output_dir):
    """Create a summary table of all metrics."""
    print("\nE2E Performance Summary:")
    print("=" * 100)
    print(f"{'Method':<12} {'Keep%':<6} {'TTFT(ms)':<10} {'Decode(ms)':<12} {'Total(ms)':<11} {'Throughput(tps)':<15} {'Memory(GB)':<12} {'KV Cache(GB)':<12}")
    print("=" * 100)
    
    summary_data = []
    
    for keep_rate in KEEP_RATES:
        for method in METHODS_ORDER:
            if method in results and keep_rate in results[method]:
                data = results[method][keep_rate]
                
                ttft = data.get('ttft_ms', 0)
                decode = data.get('decode_time_ms', 0)
                total = ttft + decode
                throughput = data.get('total_throughput_tps', 0)
                memory = data.get('max_memory_gb', 0)
                kv_cache = data.get('kv_cache_size_gb', 0)
                
                print(f"{method:<12} {keep_rate*100:<6.0f} {ttft:<10.1f} {decode:<12.1f} {total:<11.1f} {throughput:<15.1f} {memory:<12.1f} {kv_cache:<12.2f}")
                
                summary_data.append({
                    'method': method,
                    'keep_rate': keep_rate,
                    'ttft_ms': ttft,
                    'decode_time_ms': decode,
                    'total_time_ms': total,
                    'throughput_tps': throughput,
                    'memory_gb': memory,
                    'kv_cache_gb': kv_cache
                })
    
    # Save summary as JSON
    os.makedirs(output_dir, exist_ok=True)
    summary_file = os.path.join(output_dir, "e2e_performance_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"\nSaved summary table: {summary_file}")

def generate_debug_data():
    """Generate dummy data for testing visualization."""
    results = defaultdict(dict)
    
    # FullKV baseline (100%) - realistic values for 4K+32 tokens
    results["FullKV"][1.0] = {
        "ttft_ms": 850.0, "decode_time_ms": 450.0, "total_throughput_tps": 65.0, 
        "max_memory_gb": 18.5, "kv_cache_size_gb": 2.8
    }
    
    # Other methods with different keep rates
    methods_data = {
        "FastKV": [
            (0.1, 120.0, 420.0, 70.5, 16.2, 0.28),
            (0.2, 135.0, 430.0, 68.2, 16.35, 0.56), 
            (0.4, 150.0, 440.0, 66.8, 16.48, 1.12)
        ],
        "GemFilter": [
            (0.1, 125.0, 435.0, 67.8, 16.1, 0.28),
            (0.2, 140.0, 445.0, 65.5, 16.25, 0.56),
            (0.4, 155.0, 455.0, 63.2, 16.4, 1.12)
        ],
        "CLAA": [
            (0.1, 118.0, 415.0, 71.2, 16.15, 0.28),
            (0.2, 132.0, 425.0, 69.1, 16.3, 0.56),
            (0.4, 148.0, 435.0, 67.5, 16.45, 1.12)
        ],
        "Oracle": [
            (0.1, 115.0, 410.0, 72.1, 16.0, 0.28),
            (0.2, 128.0, 420.0, 70.3, 16.2, 0.56),
            (0.4, 145.0, 430.0, 68.8, 16.35, 1.12)
        ],
        "SpecPrefill": [
            (0.1, 130.0, 485.0, 58.2, 24.8, 0.28),
            (0.2, 145.0, 495.0, 56.8, 25.2, 0.56),
            (0.4, 165.0, 510.0, 54.9, 25.7, 1.12)
        ]
    }
    
    for method, configs in methods_data.items():
        for keep_rate, ttft, decode, throughput, memory, kv_cache in configs:
            results[method][keep_rate] = {
                "ttft_ms": ttft,
                "decode_time_ms": decode, 
                "total_throughput_tps": throughput,
                "max_memory_gb": memory,
                "kv_cache_size_gb": kv_cache
            }
    
    return results

def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create E2E performance bar charts")
    parser.add_argument("--results_dir", type=str, default="e2e_results",
                       help="Directory containing E2E benchmark results")
    parser.add_argument("--output_dir", type=str, default="figures",
                       help="Output directory for plots")
    parser.add_argument("--debug", action="store_true",
                       help="Use dummy data for testing")
    
    args = parser.parse_args()
    
    print("E2E Performance Bar Chart Visualization")
    print("=" * 50)
    
    if args.debug:
        print("--- Using --debug mode: Generating random dummy data ---")
        results = generate_debug_data()
        print(f"Generated data for {len(results)} methods")
    else:
        if not os.path.exists(args.results_dir):
            print(f"Error: Results directory {args.results_dir} not found")
            print("Use --debug for testing with dummy data")
            return
        
        results = load_e2e_results(args.results_dir)
        
        if not results:
            print("No E2E results found! Use --debug for testing")
            return
    
    # Generate visualizations
    create_stacked_bar_chart(results, args.output_dir)
    create_summary_table(results, args.output_dir)
    
    print(f"\nVisualization complete! Check {args.output_dir}/ for results")

if __name__ == "__main__":
    main()