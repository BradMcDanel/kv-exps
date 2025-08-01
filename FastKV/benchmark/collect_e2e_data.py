#!/usr/bin/env python3
"""
Systematic E2E data collection script for all method/keep_rate combinations.
Matches the structure used in collect_ttft_data.py
"""

import os
import json
import subprocess
import sys
import argparse
from pathlib import Path
import time

# Configuration matching collect_ttft_data.py
METHOD_ROOTS = ["fullkv", "oracle", "gemfilter", "fastkv", "speculative_prefill", "claa"]
KEEP_RATES_DECIMAL = [0.1, 0.2, 0.4]
TSP_LAYER = 15  # Hardcoded TSP layer for methods that require it
DEFAULT_SEQLEN = 4000  # 4K tokens for E2E
DEFAULT_DECODE_STEPS = 32  # Standard decode length

def run_e2e_benchmark(mode, model_name, seqlen, num_decode_steps, keep_rate=None, output_dir=None):
    """Run E2E benchmark for a specific configuration."""
    print(f"\nStarting E2E benchmark:")
    print(f"  Mode: {mode}")
    print(f"  Keep Rate: {keep_rate*100 if keep_rate else 'N/A'}%")  
    print(f"  Sequence Length: {seqlen}")
    print(f"  Decode Steps: {num_decode_steps}")
    
    cmd = [
        "python", "benchmark/benchmark_e2e.py",
        "--model", model_name,
        "--mode", mode,
        "--seqlen", str(seqlen),
        "--num_decode_steps", str(num_decode_steps),
        "--num_runs", "5",
        "--num_warmups", "3"
    ]
    
    # Add method-specific arguments with correct parameters
    if mode == "fastkv" and keep_rate is not None:
        # FastKV: tsp_idx=15, vary tsp_len_percentage with max_capacity_prompt_percentage
        cmd.extend([
            "--tsp_idx", str(TSP_LAYER),
            "--tsp_len_percentage", str(keep_rate),
            "--max_capacity_prompt_percentage", str(keep_rate)
        ])
    elif mode == "gemfilter" and keep_rate is not None:
        # GemFilter: filter_idx=15, topk_percentage
        cmd.extend([
            "--filter_idx", str(TSP_LAYER),
            "--topk_percentage", str(keep_rate)
        ])
    elif mode == "speculative_prefill" and keep_rate is not None:
        # SpecPrefill: look_ahead_k=8, max_capacity_prompt_percentage
        cmd.extend([
            "--look_ahead_k", "8",
            "--max_capacity_prompt_percentage", str(keep_rate)
        ])
    elif mode == "claa" and keep_rate is not None:
        # CLAA: Same as FastKV but with last_n_layers=4 and min_layer_idx=4
        cmd.extend([
            "--tsp_idx", str(TSP_LAYER),
            "--tsp_len_percentage", str(keep_rate),
            "--max_capacity_prompt_percentage", str(keep_rate),
            "--last_n_layers", "4",
            "--min_layer_idx", "4"
        ])
    elif mode == "oracle" and keep_rate is not None:
        # Oracle: Same as FastKV (uses dummy indices)
        cmd.extend([
            "--tsp_idx", str(TSP_LAYER),
            "--tsp_len_percentage", str(keep_rate),
            "--max_capacity_prompt_percentage", str(keep_rate)
        ])
    
    print(f"  Command: {' '.join(cmd)}")
    print(f"  Expected duration: ~60-120 seconds")
    print("  Waiting for GPU to start...")
    
    try:
        # Capture both stdout and stderr
        print("  Running benchmark (this may take a few minutes)...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)  # 10 min timeout
        
        print("  Benchmark completed, parsing results...")
        
        if result.returncode != 0:
            print(f"  ERROR: Benchmark failed for {mode} with keep_rate {keep_rate}")
            print(f"  STDOUT: {result.stdout}")
            print(f"  STDERR: {result.stderr}")
            return None
            
        # Parse results from output
        lines = result.stdout.strip().split('\n')
        ttft_ms = None
        decode_time_ms = None
        e2e_time_ms = None
        decode_throughput_tps = None
        total_throughput_tps = None
        max_memory_gb = None
        kv_cache_size_gb = None
        
        for line in lines:
            if "Time to First Token (TTFT):" in line:
                ttft_ms = float(line.split(":")[1].strip().split()[0])
            elif "Decode Time" in line and "tokens):" in line:
                decode_time_ms = float(line.split(":")[1].strip().split()[0])
            elif "Total E2E Time:" in line:
                e2e_time_ms = float(line.split(":")[1].strip().split()[0])
            elif "Decode Throughput:" in line:
                decode_throughput_tps = float(line.split(":")[1].strip().split()[0])
            elif "Total Throughput:" in line:
                total_throughput_tps = float(line.split(":")[1].strip().split()[0])
            elif "Max Memory Allocated:" in line:
                max_memory_gb = float(line.split(":")[1].strip().split()[0])
            elif "Actual KV Cache Size:" in line:
                kv_cache_size_gb = float(line.split(":")[1].strip().split()[0])
        
        if ttft_ms is not None and max_memory_gb is not None:
            print(f"  SUCCESS: TTFT = {ttft_ms:.2f}ms, Throughput = {total_throughput_tps:.1f} tps, Memory = {max_memory_gb:.2f}GB")
        else:
            print(f"  WARNING: Could not parse all metrics from output")
            print(f"  Last few lines of output: {lines[-5:]}")
        
        return {
            "ttft_ms": ttft_ms,
            "decode_time_ms": decode_time_ms,
            "e2e_time_ms": e2e_time_ms,
            "decode_throughput_tps": decode_throughput_tps,
            "total_throughput_tps": total_throughput_tps,
            "max_memory_gb": max_memory_gb,
            "kv_cache_size_gb": kv_cache_size_gb,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
        
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT: Benchmark exceeded 10 minutes for {mode} with keep_rate {keep_rate}")
        return None
    except Exception as e:
        print(f"  EXCEPTION: Error running {mode} with keep_rate {keep_rate}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--seqlen", type=int, default=DEFAULT_SEQLEN, help="Sequence length for benchmarking")
    parser.add_argument("--num_decode_steps", type=int, default=DEFAULT_DECODE_STEPS, help="Number of decode steps")
    parser.add_argument("--output_dir", type=str, default="e2e_results", help="Output directory for results")
    parser.add_argument("--methods", nargs="+", default=METHOD_ROOTS, help="Methods to benchmark")
    parser.add_argument("--keep_rates", nargs="+", type=float, default=KEEP_RATES_DECIMAL, help="Keep rates to test")
    parser.add_argument("--skip_existing", action="store_true", help="Skip if result file already exists")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Model name for file naming
    model_safe_name = args.model.replace("/", "_")
    
    all_results = {}
    
    # Process fullkv (100% keep rate) - special case
    if "fullkv" in args.methods:
        result_file = output_dir / f"{model_safe_name}_fullkv_e2e.json"
        
        if not (args.skip_existing and result_file.exists()):
            print("\n" + "="*50)
            print(f"Running FullKV (100% keep rate)")
            print("="*50)
            
            result = run_e2e_benchmark("fullkv", args.model, args.seqlen, args.num_decode_steps)
            
            if result:
                config_data = {
                    "method": "fullkv",
                    "keep_rate": 1.0,
                    "seqlen": args.seqlen,
                    "num_decode_steps": args.num_decode_steps,
                    "model": args.model,
                    **result
                }
                
                with open(result_file, 'w') as f:
                    json.dump(config_data, f, indent=2)
                
                all_results["fullkv_1.0"] = config_data
                print(f"✓ FullKV: {result['total_throughput_tps']:.1f} tps, {result['max_memory_gb']:.2f}GB")
            else:
                print("✗ FullKV failed")
        else:
            print(f"Skipping FullKV - result exists at {result_file}")
    
    # Process other methods with different keep rates
    for method in args.methods:
        if method == "fullkv":
            continue
            
        for keep_rate in args.keep_rates:
            # Generate folder name matching collect_ttft_data.py logic
            if method == "specprefill":
                folder_name = f"{method}_{keep_rate}p"
            else:
                folder_name = f"{method}_l{TSP_LAYER}_{keep_rate}p"
            
            result_file = output_dir / f"{model_safe_name}_{folder_name}_e2e.json"
            
            if args.skip_existing and result_file.exists():
                print(f"Skipping {method} {keep_rate*100:.0f}% - result exists")
                continue
                
            print("\n" + "="*50)
            print(f"Running {method.upper()} with {keep_rate*100:.0f}% keep rate")
            print("="*50)
            
            result = run_e2e_benchmark(method, args.model, args.seqlen, args.num_decode_steps, keep_rate)
            
            if result:
                config_data = {
                    "method": method,
                    "keep_rate": keep_rate,
                    "folder_name": folder_name,
                    "seqlen": args.seqlen,
                    "num_decode_steps": args.num_decode_steps,
                    "model": args.model,
                    **result
                }
                
                with open(result_file, 'w') as f:
                    json.dump(config_data, f, indent=2)
                
                all_results[f"{method}_{keep_rate}"] = config_data
                print(f"COMPLETED {method}: {result['total_throughput_tps']:.1f} tps, {result['max_memory_gb']:.2f}GB")
            else:
                print(f"FAILED {method} {keep_rate*100:.0f}%")
            
            # Small delay between runs
            time.sleep(2)
    
    # Save summary
    summary_file = output_dir / f"{model_safe_name}_e2e_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*50}")
    print(f"E2E data collection complete!")
    print(f"Results saved to: {output_dir}")
    print(f"Summary: {summary_file}")
    print(f"Individual files: {len(all_results)} configurations")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()