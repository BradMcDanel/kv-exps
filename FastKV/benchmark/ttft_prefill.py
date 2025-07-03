import sys
sys.path.append(".")

import argparse
import os
import time
import json
import logging
import pprint
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn

from accelerate import dispatch_model, load_checkpoint_in_model, infer_auto_device_map
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from datasets import load_dataset

from utils import utils

def average_excluding_min_max(numbers):
    if len(numbers) <= 2:
        if len(numbers) > 0:
            return sum(numbers) / len(numbers)
        return 0
    
    numbers_excluding_min_max = numbers.copy()
    numbers_excluding_min_max.remove(min(numbers))
    numbers_excluding_min_max.remove(max(numbers))

    return sum(numbers_excluding_min_max) / len(numbers_excluding_min_max)


def main(args):
    set_seed(args.seed)

    # This initial setup is for single-model modes.
    # draft_tsp will handle its own setup later.
    if args.mode == 'fullkv':
        from baseline.fullkv.monkeypatch import replace_llama, replace_mistral
        replace_llama()
        replace_mistral()
    elif args.mode == 'fastkv':
        from baseline.fastkv.monkeypatch import replace_llama, replace_mistral
        replace_llama()
        replace_mistral()
    elif args.mode == 'hfastkv':
        from baseline.hfastkv.monkeypatch import replace_llama, replace_mistral
        replace_llama()
        replace_mistral()
    elif args.mode == 'snapkv':
        from baseline.snapkv.monkeypatch import replace_llama, replace_mistral, replace_phi3
        replace_llama()
        replace_mistral()
        replace_phi3()
    elif args.mode == 'gemfilter':
        from baseline.gemfilter.monkeypatch import replace_llama, replace_mistral
        from baseline.gemfilter.gemfilter_utils import set_topk
        replace_llama()
        replace_mistral()
    elif args.mode == 'adakv':
        from baseline.adakv.adaptive_snapkv.monkeypatch import replace_llama_adaptive, replace_mistral_adaptive
        replace_llama_adaptive()
        replace_mistral_adaptive()
    elif args.mode == 'headkv':
        from baseline.headkv.headkv.monkeypatch import replace_llama, replace_mistral
        replace_llama(args.method)
        replace_mistral(args.method)
    elif args.mode == 'draft_tsp':
        # Setup for draft_tsp is handled by the pipeline itself. No patching here.
        pass
    elif args.mode == 'speculative_prefill':
        pass
    else:
        raise ValueError(f"We does not support {args.mode} mode")

    # Load Model, Tokenizer, or Pipeline based on mode
    tokenizer = AutoTokenizer.from_pretrained(args.model, device_map='auto', trust_remote_code=True)
    tokenizer.padding_side  = 'left'
    tokenizer.pad_token = tokenizer.eos_token

    model = None
    pipeline = None

    if args.mode == 'draft_tsp':
        from baseline.draft_tsp.main import DraftTSPPipeline
        print("Initializing DraftTSPPipeline for benchmarking...")
        pipeline = DraftTSPPipeline(
            base_model_name=args.model,
            speculator_model_name=args.speculator_model_name,
            tokenizer=tokenizer,
            args=args,
            detailed_timing=args.detailed_timing,
        )
        model_device = pipeline.device
    elif args.mode == 'speculative_prefill':
        from baseline.speculative_prefill.main import SpeculativePrefillPipeline
        print("Initializing SpeculativePrefillPipeline for benchmarking...")
        pipeline = SpeculativePrefillPipeline(
            base_model_name=args.model,
            speculator_model_name=args.speculator_model_name,
            tokenizer=tokenizer,
            max_capacity_prompt=args.max_capacity_prompt,
            max_capacity_prompt_percentage=args.max_capacity_prompt_percentage,
            pool_kernel_size=args.kernel_size if args.pooling != 'none' else None,
            pool_type=args.pooling,
            use_chunk_selection=True, # Assuming default behavior
            chunk_size=64, # Assuming default behavior
            detailed_timing=args.detailed_timing,
        )
        model_device = pipeline.device
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, device_map='auto', attn_implementation='flash_attention_2', torch_dtype=torch.float16)
        model.eval()
        model_device = model.device

        # Configure single-model modes
        if args.mode == 'fastkv':
            from baseline.fastkv.fastkv_utils import compress
            compress(model, args)
        elif args.mode == 'hfastkv':
            from baseline.hfastkv.hfastkv_utils import compress
            compress(model, args)
        elif args.mode == 'snapkv':
            from baseline.snapkv.snapkv_utils import compress
            compress(model, args)
        elif args.mode == 'adakv':
            from baseline.adakv.adaptive_snapkv.snapkv_utils import compress
            compress(model, args)
        elif args.mode == 'headkv':
            from baseline.headkv.headkv.snapkv_utils import compress
            compress(model, args)
        
    # Input Sequence      
    input_id = torch.ones((1,args.seqlen), dtype=torch.int64).to(model_device)
    attn_mask = torch.ones((1,args.seqlen), dtype=torch.int64).to(model_device)
    context_length = input_id.shape[-1]

    # Warmup
    if args.num_warmups > 0:
        print(f"Running {args.num_warmups} warmup(s)...")
        for i in range(args.num_warmups):
            if pipeline:
                if args.mode == 'draft_tsp':
                    with torch.no_grad():
                        _ = pipeline.run(input_ids=input_id, look_ahead_k=args.look_ahead_k, max_generation_length=1)
                elif args.mode == 'speculative_prefill':
                    with torch.no_grad():
                        _ = pipeline.run(input_ids=input_id, look_ahead_k=args.look_ahead_k, max_generation_length=1)
            elif args.mode == 'gemfilter':
                from baseline.gemfilter.gemfilter_utils import gemfilter_generate_selection_prefill
                set_topk(model, args.max_capacity_prompt, mode='gemfilter')
                with torch.no_grad():
                    _ = gemfilter_generate_selection_prefill(input_id, attn_mask, model, tokenizer, select_layer_idx=args.filter_idx)
            else: # Standard single-model modes
                with torch.no_grad():
                    _ = model(input_id, attention_mask=attn_mask)
            utils.cleanup_memory(verbos=False)

    # Benchmark runs
    result_list = []
    metadata_list = []
    print(f"Running {args.num_runs} benchmark run(s)...")
    for i in range(args.num_runs):        
        total_time = 0
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        run_metadata = {}

        # Prefill timing logic
        with torch.no_grad():
            start.record()
            if pipeline:
                if args.mode == 'draft_tsp':
                    _, run_metadata = pipeline.run(input_ids=input_id, look_ahead_k=args.look_ahead_k, max_generation_length=1)
                elif args.mode == 'speculative_prefill':
                    _, run_metadata = pipeline.run(input_ids=input_id, look_ahead_k=args.look_ahead_k, max_generation_length=1)
            elif args.mode == 'gemfilter':
                from baseline.gemfilter.gemfilter_utils import gemfilter_generate_selection_prefill
                set_topk(model, args.max_capacity_prompt, mode='gemfilter')
                _ = gemfilter_generate_selection_prefill(input_id, attn_mask, model, tokenizer, select_layer_idx=args.filter_idx)
            else: # Standard single-model modes
                _ = model(input_id, attention_mask=attn_mask)
            
            end.record()
            torch.cuda.synchronize()
            total_time += start.elapsed_time(end)
        
        result_list.append(total_time)
        if run_metadata:
            metadata_list.append(run_metadata)
        utils.cleanup_memory(verbos=False)

    mean_ttft = average_excluding_min_max(result_list)

    print(f"\nMode: {args.mode}")
    print(f"Context Length = {context_length}")
    if args.mode == "draft_tsp":
        print(f"TSP Schedule = '{args.tsp_schedule}'")
    elif args.mode == "speculative_prefill":
        if args.max_capacity_prompt_percentage:
            print(f"Max Prompt Capacity: {args.max_capacity_prompt_percentage:.1%}")
        else:
            print(f"Max Prompt Capacity: {args.max_capacity_prompt} tokens")
    elif args.mode != "fullkv":
        print(f"Context Capacity = {args.max_capacity_prompt}")
    
    print(f"TTFT: {(mean_ttft):.5f} msec")

    if args.detailed_timing and metadata_list:
        avg_spec_prefill = average_excluding_min_max([m.get('speculation_prefill', 0) * 1000 for m in metadata_list])
        avg_spec_decode = average_excluding_min_max([m.get('speculation_decode', 0) * 1000 for m in metadata_list])
        avg_base_prefill = average_excluding_min_max([m.get('base_prefill', 0) * 1000 for m in metadata_list])
        print(f"  - Speculator Prefill: {avg_spec_prefill:.5f} msec")
        print(f"  - Speculator Decode/Scoring: {avg_spec_decode:.5f} msec")
        print(f"  - Base Model Prefill: {avg_base_prefill:.5f} msec")

    print(f"Max memory allocated: {torch.cuda.max_memory_allocated() / (1024**3):.2f} GB\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model Arguments
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", help="model name of model path")
    parser.add_argument("--seed", type=int, default=42, help="Seed")

    # ==================== ADD DRAFT_TSP TO CHOICES AND ITS ARGS ====================
    parser.add_argument("--mode", type=str, default="fastkv", choices=["fullkv", "fastkv", "snapkv", "gemfilter", "adakv", "headkv", "hfastkv", "draft_tsp", "speculative_prefill"])
    parser.add_argument("--speculator_model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="Speculator model for draft_tsp.")
    
    parser.add_argument("--look_ahead_k", type=int, default=8, help="Number of lookahead steps for Draft TSP.")
    # =============================================================================
    
    # Common KV Compression Arguments
    parser.add_argument("--window_size", type=int, default=8)
    parser.add_argument("--max_capacity_prompt", type=int, default=512)
    parser.add_argument("--max_capacity_prompt_percentage", type=float, default=None, help="Use a percentage of the prompt length for max capacity.")
    parser.add_argument("--kernel_size", type=int, default=7)
    parser.add_argument("--pooling", type=str, default="avgpool")
    
    # FastKV
    parser.add_argument("--tsp_idx", type=int, default=15)
    parser.add_argument("--tsp_len", type=int, default=2048)
    # Hierarchical FastKV / Draft TSP
    parser.add_argument("--tsp_schedule", type=str, default="", help="Hierarchical TSP schedule for HFastKV/Draft_TSP, e.g., '15:2048'")

    # GemFilter
    parser.add_argument("--filter_idx", type=int, default=13)
    # AdaKV
    parser.add_argument("--skip", type=int, default=-1)
    parser.add_argument('--floor_alpha', type=float, default=0.2)
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--pyram', action='store_true')
    parser.add_argument('--pyram_beta', default=20,type=int)
    # HeadKV
    parser.add_argument("--method", type=str, default='ReasonKV', choices=['ReasonKV'])
    parser.add_argument("--head_choice", type=str, default='reason', choices=['copy', 'reason'])
    parser.add_argument('--beta', type=float, default=1.2)
    parser.add_argument('--temp', type=float, default=1.0)

    parser.add_argument("--detailed_timing", action="store_true", help="Enable detailed, per-part timing with CUDA synchronization.")

    # Benchmark Option
    parser.add_argument("--seqlen", type=int, default=131072, help="")
    parser.add_argument("--num_warmups", type=int, default=2, help="")
    parser.add_argument("--num_runs", type=int, default=10, help="num_runs must be larger than 2")


    args = parser.parse_args()
    if args.num_runs <= 2:
        print("Warning: num_runs is <= 2. The average will be calculated without excluding min/max.")

    main(args)
    utils.cleanup_memory()
