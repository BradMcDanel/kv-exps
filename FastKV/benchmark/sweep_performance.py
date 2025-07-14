# benchmark/sweep_performance.py

import csv
import argparse
from pathlib import Path
import gc
import multiprocessing as mp

# Import the refactored benchmark function
from benchmark.benchmark_e2e import run_e2e_benchmark

# --- Configuration Section ---
MODELS_TO_TEST = {
    "Llama-3.1-8B": {
        "model_name": "meta-llama/Llama-3.1-8B-Instruct",
        "speculator_name": "meta-llama/Llama-3.2-1B-Instruct"
    },
}
METHODS_TO_SWEEP = [
    "fullkv",
    "fastkv",
    "speculative_prefill",
    "hgp",
]
SEQ_LEN = 4000
NUM_DECODE_STEPS = 16
LOOK_AHEAD_K = 4
NUM_RUNS = 5
NUM_WARMUPS = 3
PRUNING_LEVELS = [0.1, 0.2, 0.3, 0.4]
# --- End Configuration Section ---

def create_run_args(config):
    """Creates an argparse.Namespace object from a config dictionary."""
    defaults = {
        'model': None, 'speculator_model_name': None, 'mode': None,
        'seqlen': SEQ_LEN, 'num_decode_steps': NUM_DECODE_STEPS,
        'look_ahead_k': LOOK_AHEAD_K, 'num_runs': NUM_RUNS,
        'num_warmups': NUM_WARMUPS, 'seed': 42,
        'max_capacity_prompt': 512, 'max_capacity_prompt_percentage': None,
        'tsp_schedule': "", 'tsp_idx': 15, 'tsp_len': 2048,
        'kernel_size': 7,
        'pooling': 'avgpool',
        'window_size': 8,
        'method': 'ReasonKV'
    }

    defaults['model'] = config['model_name']
    defaults['speculator_model_name'] = config.get('speculator_name')
    method = config["method"]
    level = config["pruning_level"]
    
    if method == "fullkv":
        defaults['mode'] = "fullkv"
    elif method == "fastkv":
        defaults['mode'] = "fastkv"
        defaults['tsp_idx'] = 15
        defaults['tsp_len'] = int(config["seq_len"] * level)
        defaults['max_capacity_prompt'] = int(config["seq_len"] * level)
    elif method == "speculative_prefill":
        defaults['mode'] = "speculative_prefill"
        defaults['max_capacity_prompt_percentage'] = level
    elif method == "hgp":
        defaults['mode'] = "draft_tsp"
        early_level = min(1.0, level + 0.15)
        late_level = max(0.05, level - 0.15)
        defaults['tsp_schedule'] = f"0:{early_level:.2f},11:{level:.2f},23:{late_level:.2f}"
        print(f"Using HGP schedule: {defaults['tsp_schedule']}")
    else:
        raise ValueError(f"Unknown method strategy: {method}")
    return argparse.Namespace(**defaults)

def benchmark_process_wrapper(args_namespace, result_queue):
    """Wrapper to run the benchmark in a separate, isolated process."""
    try:
        results = run_e2e_benchmark(args_namespace)
        result_queue.put(results)
    except Exception as e:
        print(f"--- ERROR in child process for mode {args_namespace.mode} ---")
        import traceback
        traceback.print_exc()
        result_queue.put(e)

def main():
    parser = argparse.ArgumentParser(description="Sweep performance benchmarks for different KV caching methods.")
    parser.add_argument("--output_dir", type=str, default="analysis_results", help="Directory to save the output CSV.")
    args = parser.parse_args()
    
    mp.set_start_method("spawn", force=True)
    Path(args.output_dir).mkdir(exist_ok=True)
    
    for model_key, model_info in MODELS_TO_TEST.items():
        output_filename = Path(args.output_dir) / f"performance_sweep_{model_key}.csv"
        print(f"Starting sweep for model: {model_key}. Results will be saved to {output_filename}")
        
        run_configs = []
        for method in METHODS_TO_SWEEP:
            if method == "fullkv":
                run_configs.append({"method": method, "pruning_level": 1.0})
            else:
                for level in PRUNING_LEVELS:
                    run_configs.append({"method": method, "pruning_level": level})
        
        for config in run_configs:
            config.update({
                "model_key": model_key, "model_name": model_info["model_name"],
                "speculator_name": model_info.get("speculator_name"), "seq_len": SEQ_LEN,
            })
        
        with open(output_filename, 'w', newline='') as f:
            writer = None
            for i, config in enumerate(run_configs):
                print("\n" + "="*80)
                print(f"RUNNING ({i+1}/{len(run_configs)}): Method={config['method']}, Pruning Level={config['pruning_level']*100:.0f}%")
                print("="*80)
                run_args = create_run_args(config)
                result_queue = mp.Queue()
                process = mp.Process(target=benchmark_process_wrapper, args=(run_args, result_queue))
                process.start()
                process.join()
                result = result_queue.get()
                
                if isinstance(result, dict):
                    metrics = result
                    config['schedule_used'] = run_args.tsp_schedule if config['method'] == 'hgp' else 'N/A'
                    row_data = {**config, **metrics}
                    if writer is None:
                        writer = csv.DictWriter(f, fieldnames=row_data.keys())
                        writer.writeheader()
                    writer.writerow(row_data)
                    f.flush()
                else:
                    print(f"--- FATAL ERROR during benchmark run. Skipping. ---")
                    print(f"Error from child process: {result}")
                del process
                gc.collect()

        print(f"\nSweep for {model_key} complete. Results saved to {output_filename}")

if __name__ == "__main__":
    main()
