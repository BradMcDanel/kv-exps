# benchmark/benchmark_e2e.py

import sys
sys.path.append(".")

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from utils import utils
import gc
import json
import os

def average_excluding_min_max(numbers):
    """Calculates the average of a list of numbers, excluding the min and max values."""
    if not numbers:
        return 0
    if len(numbers) <= 2:
        return sum(numbers) / len(numbers)
    
    numbers_excluding_min_max = sorted(numbers)[1:-1]
    return sum(numbers_excluding_min_max) / len(numbers_excluding_min_max)

def calculate_kv_cache_size(model, sequence_length, batch_size=1):
    """Calculate the theoretical KV cache size in GB."""
    if hasattr(model, 'config'):
        config = model.config
        num_layers = config.num_hidden_layers
        num_heads = config.num_attention_heads
        head_dim = config.hidden_size // num_heads
        
        # Each token stores key and value vectors for all layers
        # 2 for K and V, * 2 for float16 (2 bytes per element)
        kv_cache_size_bytes = sequence_length * batch_size * num_layers * num_heads * head_dim * 2 * 2
        return kv_cache_size_bytes / (1024**3)  # Convert to GB
    return 0

def get_actual_kv_cache_size(past_key_values):
    """Calculate actual KV cache size from past_key_values."""
    if past_key_values is None:
        return 0
    
    total_size = 0
    for layer_kv in past_key_values:
        if layer_kv is not None:
            key_cache, value_cache = layer_kv
            total_size += key_cache.numel() * key_cache.element_size()
            total_size += value_cache.numel() * value_cache.element_size()
    
    return total_size / (1024**3)  # Convert to GB


def run_e2e_benchmark(args):
    """
    Core logic for the E2E benchmark. Can be called as a library function.
    
    Args:
        args: An object (like argparse.Namespace) with all necessary benchmark parameters.
    
    Returns:
        A dictionary containing the benchmark results.
    """
    set_seed(args.seed)

    # Apply monkey-patching for different modes
    if args.mode == 'fullkv':
        from baseline.fullkv.monkeypatch import replace_llama, replace_mistral
        replace_llama(); replace_mistral()
    elif args.mode == 'fastkv':
        from baseline.fastkv.monkeypatch import replace_llama, replace_mistral
        replace_llama(); replace_mistral()
    elif args.mode == 'hfastkv':
        from baseline.hfastkv.monkeypatch import replace_llama, replace_mistral
        replace_llama(); replace_mistral()
    elif args.mode == 'snapkv':
        from baseline.snapkv.monkeypatch import replace_llama, replace_mistral, replace_phi3
        replace_llama(); replace_mistral(); replace_phi3()
    elif args.mode == 'gemfilter':
        from baseline.gemfilter.monkeypatch import replace_llama, replace_mistral
        from baseline.gemfilter.gemfilter_utils import set_topk
        replace_llama(); replace_mistral()
    elif args.mode == 'adakv':
        from baseline.adakv.adaptive_snapkv.monkeypatch import replace_llama_adaptive, replace_mistral_adaptive
        replace_llama_adaptive(); replace_mistral_adaptive()
    elif args.mode == 'headkv':
        from baseline.headkv.headkv.monkeypatch import replace_llama, replace_mistral
        replace_llama(args.method); replace_mistral(args.method)
    elif args.mode == 'claa':
        from baseline.claa.monkeypatch import replace_llama, replace_mistral
        replace_llama(); replace_mistral()
    elif args.mode == 'oracle':
        from baseline.oracle.monkeypatch import replace_llama, replace_mistral
        replace_llama(); replace_mistral()
    elif args.mode in ['draft_tsp', 'speculative_prefill']:
        pass
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

    # Load Model, Tokenizer, or Pipeline
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model, pipeline = None, None
    torch.cuda.reset_peak_memory_stats()
    
    if args.mode in ['draft_tsp', 'speculative_prefill']:
        if args.mode == 'draft_tsp':
            from baseline.draft_tsp.main import DraftTSPPipeline as Pipeline
            pipeline = Pipeline(
                base_model_name=args.model, speculator_model_name=args.speculator_model_name,
                tokenizer=tokenizer, args=args, detailed_timing=True
            )
        elif args.mode == 'speculative_prefill':
            from baseline.speculative_prefill.main import SpeculativePrefillPipeline as Pipeline
            max_cap_prompt = args.max_capacity_prompt
            if args.max_capacity_prompt_percentage is not None: max_cap_prompt = None
            pipeline = Pipeline(
                base_model_name=args.model, speculator_model_name=args.speculator_model_name,
                tokenizer=tokenizer, max_capacity_prompt=max_cap_prompt,
                max_capacity_prompt_percentage=args.max_capacity_prompt_percentage,
                pool_kernel_size=args.kernel_size if args.pooling != 'none' else None,
                pool_type=args.pooling, detailed_timing=True
            )
        model_device = pipeline.device
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, device_map='auto', attn_implementation='flash_attention_2',
            torch_dtype=torch.float16
        )
        model.eval()
        model_device = model.device

        # Apply compression with min_layer_idx support
        if args.mode == 'fastkv':
            from baseline.fastkv.fastkv_utils import compress
            compress(model, args)
        elif args.mode == 'claa':
            from baseline.claa.claa_utils import compress
            compress(model, args)
        elif args.mode == 'oracle':
            from baseline.oracle.oracle_utils import compress, set_oracle_rankings
            # For benchmarking, use random rankings since we don't have precomputed ones
            import numpy as np
            random_rankings = np.random.rand(args.seqlen).astype(np.float32)
            set_oracle_rankings(random_rankings)
            compress(model, args)
        elif args.mode == 'gemfilter':
            from baseline.gemfilter.gemfilter_utils import compress
            compress(model, args)

    # Prepare input
    input_ids = torch.ones((1, args.seqlen), dtype=torch.int64).to(model_device)
    
    # Calculate theoretical KV cache size
    if model:
        theoretical_kv_cache_gb = calculate_kv_cache_size(model, args.seqlen + args.num_decode_steps)
    else:
        theoretical_kv_cache_gb = 0
    
    # Warmup
    for _ in range(args.num_warmups):
        with torch.no_grad():
            if pipeline:
                _ = pipeline.run(input_ids=input_ids, look_ahead_k=args.look_ahead_k, max_generation_length=2)
            else:
                _ = model.generate(input_ids, max_new_tokens=2, use_cache=True)
        utils.cleanup_memory(verbos=False)

    # Benchmark runs
    ttft_list, decode_time_list, e2e_time_list = [], [], []
    kv_cache_sizes = []
    
    for run_idx in range(args.num_runs):
        utils.cleanup_memory(verbos=False)
        prefill_time, decoding_time = 0, 0
        run_kv_cache_size = 0
        
        prefill_start = torch.cuda.Event(enable_timing=True)
        prefill_end = torch.cuda.Event(enable_timing=True)
        decode_end = torch.cuda.Event(enable_timing=True)

        with torch.no_grad():
            if pipeline:
                total_start = torch.cuda.Event(enable_timing=True)
                total_end = torch.cuda.Event(enable_timing=True)
                total_start.record()
                _, run_metadata = pipeline.run(
                    input_ids=input_ids, look_ahead_k=args.look_ahead_k,
                    max_generation_length=args.num_decode_steps
                )
                total_end.record()
                torch.cuda.synchronize()

                ttft = run_metadata.get("base_ttft", 0) * 1000
                total_e2e_time = total_start.elapsed_time(total_end)
                decoding_time = total_e2e_time - ttft
                prefill_time = ttft
            else:
                prefill_start.record()
                outputs = model(input_ids, use_cache=True)
                prefill_end.record()
                
                past_key_values = outputs.past_key_values
                run_kv_cache_size = get_actual_kv_cache_size(past_key_values)
                
                next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(-1)
                
                for step in range(args.num_decode_steps - 1):
                    outputs = model(next_token, past_key_values=past_key_values, use_cache=True)
                    past_key_values = outputs.past_key_values
                    next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(-1)
                    
                    # Update KV cache size (it should be similar for all decode steps)
                    if step == 0:  # Just measure once during decode for efficiency
                        run_kv_cache_size = max(run_kv_cache_size, get_actual_kv_cache_size(past_key_values))
                decode_end.record()
                torch.cuda.synchronize()
                prefill_time = prefill_start.elapsed_time(prefill_end)
                decoding_time = prefill_end.elapsed_time(decode_end)
        
        ttft_list.append(prefill_time)
        decode_time_list.append(decoding_time)
        e2e_time_list.append(prefill_time + decoding_time)
        kv_cache_sizes.append(run_kv_cache_size)

    # Calculate final metrics
    mean_ttft = average_excluding_min_max(ttft_list)
    mean_decode_time = average_excluding_min_max(decode_time_list)
    mean_e2e_time = average_excluding_min_max(e2e_time_list)
    mean_kv_cache_size = average_excluding_min_max(kv_cache_sizes) if kv_cache_sizes else 0
    
    num_generated_tokens = args.num_decode_steps - 1
    decode_throughput = (num_generated_tokens / (mean_decode_time / 1000.0)) if num_generated_tokens > 0 and mean_decode_time > 0 else 0
    # E2E throughput should be decode tokens over total time (prefill + decode)
    total_throughput = (num_generated_tokens / (mean_e2e_time / 1000.0)) if num_generated_tokens > 0 and mean_e2e_time > 0 else 0
    max_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)

    # Clean up to release memory before returning
    if model:
        del model
    if pipeline:
        del pipeline
    del tokenizer
    del input_ids
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "method": args.mode,
        "keep_rate": getattr(args, 'max_capacity_prompt_percentage', None) or (args.max_capacity_prompt / args.seqlen if hasattr(args, 'max_capacity_prompt') else None),
        "ttft_ms": mean_ttft,
        "decode_time_ms": mean_decode_time,
        "e2e_time_ms": mean_e2e_time,
        "decode_throughput_tps": decode_throughput,
        "total_throughput_tps": total_throughput,
        "max_memory_gb": max_memory_gb,
        "kv_cache_size_gb": mean_kv_cache_size,
        "theoretical_kv_cache_gb": theoretical_kv_cache_gb,
        "min_layer_idx": getattr(args, 'min_layer_idx', 0),
        "tsp_idx": getattr(args, 'tsp_idx', None),
        "seqlen": args.seqlen,
        "num_decode_steps": args.num_decode_steps,
    }

def main(cli_args):
    """Handles command-line execution."""
    print(f"Initializing E2E benchmark for mode: {cli_args.mode}...")
    results = run_e2e_benchmark(cli_args)

    # Print results in the original format for standalone runs
    print("\n" + "="*50)
    print(f"E2E Benchmark Results for Mode: {cli_args.mode}")
    print("="*50)
    print(f"Prompt Length: {cli_args.seqlen} tokens")
    print(f"Generated Tokens: {cli_args.num_decode_steps} tokens")
    print(f"Min Layer Index: {getattr(cli_args, 'min_layer_idx', 0)}")

    if cli_args.mode == "speculative_prefill":
        if cli_args.max_capacity_prompt_percentage:
            print(f"Max Prompt Capacity: {cli_args.max_capacity_prompt_percentage:.1%}")
        else:
            print(f"Max Prompt Capacity: {cli_args.max_capacity_prompt} tokens")
    elif cli_args.mode == "draft_tsp":
        print(f"TSP Schedule: '{cli_args.tsp_schedule}'")
        print(f"Lookahead K: {cli_args.look_ahead_k}")
    elif cli_args.mode != "fullkv":
        print(f"Context Capacity: {cli_args.max_capacity_prompt}")
    
    print("-" * 50)
    print(f"Time to First Token (TTFT):     {results['ttft_ms']:.3f} ms")
    print(f"Decode Time ({cli_args.num_decode_steps - 1} tokens):       {results['decode_time_ms']:.3f} ms")
    print(f"Total E2E Time:                 {results['e2e_time_ms']:.3f} ms")
    print(f"Decode Throughput:              {results['decode_throughput_tps']:.2f} tokens/s")
    print(f"Total Throughput:               {results['total_throughput_tps']:.2f} tokens/s")
    print("-" * 50)
    print(f"Max Memory Allocated:           {results['max_memory_gb']:.2f} GB")
    print(f"Actual KV Cache Size:           {results['kv_cache_size_gb']:.2f} GB")
    print(f"Theoretical KV Cache Size:      {results['theoretical_kv_cache_gb']:.2f} GB")
    print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-end benchmark for prefill and decode performance.")
    # Add all arguments...
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--seqlen", type=int, default=4000)
    parser.add_argument("--num_decode_steps", type=int, default=32)
    parser.add_argument("--num_warmups", type=int, default=2)
    parser.add_argument("--num_runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", type=str, default="fastkv", choices=["fullkv", "fastkv", "snapkv", "gemfilter", "adakv", "headkv", "hfastkv", "draft_tsp", "speculative_prefill", "claa", "oracle"])
    parser.add_argument("--speculator_model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--look_ahead_k", type=int, default=4)
    parser.add_argument("--max_capacity_prompt", type=int, default=512)
    parser.add_argument("--max_capacity_prompt_percentage", type=float, default=None)
    parser.add_argument("--min_layer_idx", type=int, default=0, help="Minimum layer index for KV compression. Layers below this index skip compression.")
    parser.add_argument("--kernel_size", type=int, default=7)
    parser.add_argument("--pooling", type=str, default="avgpool", choices=['avgpool', 'maxpool', 'none'])
    parser.add_argument("--tsp_schedule", type=str, default="", help="Hierarchical TSP schedule for DraftTSP/HFastKV")
    parser.add_argument("--tsp_idx", type=int, default=15)
    parser.add_argument("--tsp_len", type=int, default=2048)
    parser.add_argument("--tsp_len_percentage", type=float, default=None)
    parser.add_argument("--window_size", type=int, default=8)
    parser.add_argument("--method", type=str, default='ReasonKV', choices=['ReasonKV'])
    parser.add_argument("--last_n_layers", type=int, default=None, help="Number of layers for CLAA aggregation")
    
    # GemFilter specific
    parser.add_argument("--filter_idx", type=int, default=13)
    parser.add_argument("--topk", type=int, default=1024)
    parser.add_argument("--topk_percentage", type=float, default=None, help="Use a percentage of the prompt length for GemFilter topk")
    parser.add_argument("--select_layer_idx", type=int, default=13, help="Layer index for GemFilter selection")
    
    # Oracle specific
    parser.add_argument("--oracle_model", type=str, default=None, help="Oracle model path")
    
    args = parser.parse_args()
    if args.num_runs <= 2:
        print("Warning: num_runs is <= 2. The average will be calculated without excluding min/max.")

    main(args)
