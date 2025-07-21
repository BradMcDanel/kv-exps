# main.py (longbench evaluation script)
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
import os
import numpy as np

import torch
import numpy as np 

from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from datasets import load_dataset

from utils import utils

# =====================================================================================
# 1. STARTUP PHASE: Functions for one-time setup
# =====================================================================================

def setup_model_and_tokenizer(args):
    """
    Handles all one-time setup for the chosen mode.
    This includes monkey-patching, model/tokenizer/pipeline loading, and any
    post-load configuration like compression.
    
    Returns:
        - A fully configured model object (or None if using a pipeline).
        - A fully configured pipeline object (or None if not using a pipeline).
        - A tokenizer object.
    """
    logging.info(f"Setting up for mode: {args.mode}")
    
    # Load tokenizer once, as it's needed by all modes for prompt prep.
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = None
    pipeline = None

    # --- MODIFICATION START: Re-structured mode handling ---

    # Group true pipeline-based modes together
    if args.mode in ["speculative_prefill", "uniform", "echo_cache", "draft_tsp"]:
        if args.mode == "speculative_prefill":
            from baseline.speculative_prefill.main import SpeculativePrefillPipeline as Pipeline
            logging.info("Initializing SpeculativePrefillPipeline with custom parameters...")
            max_cap_prompt = args.max_capacity_prompt
            max_cap_pct = args.max_capacity_prompt_percentage
            if max_cap_pct is not None:
                if max_cap_prompt is not None:
                    logging.info(f"Using `max_capacity_prompt_percentage={max_cap_pct}`. Ignoring `max_capacity_prompt={max_cap_prompt}`.")
                max_cap_prompt = None
            pipeline = Pipeline(
                base_model_name=args.model,
                speculator_model_name=args.speculator_model_name,
                tokenizer=tokenizer,
                max_capacity_prompt=max_cap_prompt,
                max_capacity_prompt_percentage=max_cap_pct,
                pool_kernel_size=args.kernel_size if args.pooling != 'none' else None,
                pool_type=args.pooling,
                use_chunk_selection=args.use_chunk_selection, 
                chunk_size=args.chunk_size,
            )
        elif args.mode == "uniform":
            from baseline.uniform.main import UniformRandomPipeline as Pipeline
            logging.info("Initializing UniformRandomPipeline...")
            pipeline = Pipeline(
                base_model_name=args.model,
                tokenizer=tokenizer,
                keep_percentage=args.keep_percentage,
                first_k=args.uniform_first_k,
                last_k=args.uniform_last_k,
            )
        elif args.mode == "echo_cache":
            from baseline.echo_cache.main import EchoCachePipeline as Pipeline
            logging.info(f"Initializing pipeline for {args.mode}...")
            pipeline = Pipeline(
                base_model_name=args.model,
                speculator_model_name=args.speculator_model_name,
                tokenizer=tokenizer,
                max_capacity_prompt=args.max_capacity_prompt,
                tsp_len=args.tsp_len,
                cache_granularity=args.cache_granularity,
                pool_kernel_size=args.kernel_size if args.pooling != 'none' else None,
                pool_type=args.pooling,
                use_chunk_selection=args.use_chunk_selection, 
                chunk_size=args.chunk_size,
            )
        elif args.mode == "draft_tsp":
            from baseline.draft_tsp.main import DraftTSPPipeline as Pipeline
            logging.info(f"Initializing DraftTSPPipeline...")
            pipeline = Pipeline(
                base_model_name=args.model,
                speculator_model_name=args.speculator_model_name,
                tokenizer=tokenizer,
                args=args, # Pass all args for simplicity
            )
    else:
        if args.mode == 'fullkv':
            from baseline.fullkv.monkeypatch import replace_llama, replace_mistral
            replace_llama(); replace_mistral()
        elif args.mode == 'fastkv':
            from baseline.fastkv.monkeypatch import replace_llama, replace_mistral
            replace_llama(); replace_mistral()
        elif args.mode == 'hfastkv':
            from baseline.hfastkv.monkeypatch import replace_llama, replace_mistral
            replace_llama(); replace_mistral()
        elif args.mode == 'taper':
            from baseline.taper.monkeypatch import replace_llama, replace_mistral
            replace_llama(); replace_mistral()
        elif args.mode == 'claa':
            from baseline.claa.monkeypatch import replace_llama, replace_mistral
            replace_llama(); replace_mistral()
        elif args.mode == 'snapkv':
            from baseline.snapkv.monkeypatch import replace_llama, replace_mistral, replace_phi3
            replace_llama(); replace_mistral(); replace_phi3()
        elif args.mode == 'gemfilter':
            from baseline.gemfilter.monkeypatch import replace_llama, replace_mistral
            replace_llama(); replace_mistral()
        elif args.mode == 'adakv':
            from baseline.adakv.adaptive_snapkv.monkeypatch import replace_llama_adaptive, replace_mistral_adaptive
            replace_llama_adaptive(); replace_mistral_adaptive()
        elif args.mode == 'oracle':
            from baseline.oracle.monkeypatch import replace_llama, replace_mistral
            replace_llama(); replace_mistral()
        elif args.mode == 'headkv':
            from baseline.headkv.headkv.monkeypatch import replace_llama, replace_mistral
            replace_llama(args.method); replace_mistral(args.method)

        logging.info(f'Loading Model for {args.mode}...')
        model = AutoModelForCausalLM.from_pretrained(
            args.model, 
            device_map='auto', 
            attn_implementation='flash_attention_2', 
            torch_dtype=torch.float16
        )
        model.eval()

        if args.mode == 'fastkv':
            from baseline.fastkv.fastkv_utils import compress
            compress(model, args)
        elif args.mode == 'taper':
            from baseline.taper.taper_utils import compress
            compress(model, args)
        elif args.mode == 'claa':
            from baseline.claa.claa_utils import compress
            compress(model, args)
        elif args.mode == 'hfastkv':
            from baseline.hfastkv.hfastkv_utils import compress
            compress(model, args)
        elif args.mode == 'snapkv':
            from baseline.snapkv.snapkv_utils import compress
            compress(model, args)
        elif args.mode == 'gemfilter':
            from baseline.gemfilter.gemfilter_utils import compress
            compress(model, args)
        elif args.mode == 'adakv':
            from baseline.adakv.adaptive_snapkv.snapkv_utils import compress
            compress(model, args)
        elif args.mode == 'oracle':
            from baseline.oracle.oracle_utils import compress
            compress(model, args)
        elif args.mode == 'headkv':
            from baseline.headkv.headkv.snapkv_utils import compress
            compress(model, args)

    # --- MODIFICATION END ---
    return model, pipeline, tokenizer

def build_chat(tokenizer, prompt, model_name):
    """Applies the chat template to a prompt string."""
    if "Llama-2" in model_name:
        return f"[INST]{prompt}[/INST]"
    else:
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# =====================================================================================
# 2. RUNTIME PHASE: Functions for repetitive processing
# =====================================================================================

def generate_longbench(data, max_length, max_gen, prompt_format, 
                       dataset, model, pipeline, tokenizer,
                       out_path, args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model is not None:
        device = model.device
    elif pipeline is not None:
        device = pipeline.device

    if os.path.exists(out_path):
        os.remove(out_path)
        logging.info(f"Deleted existing output file: {out_path}")

    for i, json_obj in tqdm(enumerate(data), desc=f"Generating Responses for {args.mode}...", total=len(data)):
        try:
            if args.mode == 'oracle':
                from baseline.oracle.oracle_utils import set_oracle_rankings
                
                # Construct path to oracle rankings file
                oracle_file = os.path.join(args.oracle_rankings_path, args.oracle_model_name.replace("/", "_"), f"{dataset}.npz")
                sample_key = f"sample_{i}"
                
                if not os.path.exists(oracle_file):
                    print(f"Oracle rankings file not found: {oracle_file}. Skipping sample {i}.")
                    continue
                
                # Load rankings for this sample
                try:
                    oracle_data = np.load(oracle_file, allow_pickle=True)
                    if sample_key not in oracle_data:
                        print(f"Sample {sample_key} not found in {oracle_file}. Skipping sample {i}.")
                        continue
                    
                    sample_data = oracle_data[sample_key].item()
                    rankings = sample_data['ranking']
                    set_oracle_rankings(rankings)
                    
                except Exception as e:
                    print(f"Error loading oracle rankings for sample {i}: {e}. Skipping.")
                    continue
            prompt = prompt_format.format(**json_obj)
            # Tokenize once to check length, then decode for truncation
            tokenized_prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            if len(tokenized_prompt_ids) > max_length:
                half = int(max_length / 2)
                prompt = tokenizer.decode(tokenized_prompt_ids[:half], skip_special_tokens=True) + tokenizer.decode(tokenized_prompt_ids[-half:], skip_special_tokens=True)
            
            if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]: 
                final_prompt = build_chat(tokenizer, prompt, args.model)
            else:
                final_prompt = prompt

            inputs = tokenizer(final_prompt, truncation=False, return_tensors="pt").to(device)
            
            pred = ""
            
            if pipeline is not None:
                if args.mode == "uniform":
                     pred, _ = pipeline.run(
                        input_ids=inputs.input_ids,
                        max_generation_length=max_gen,
                    )
                else:
                    pred, _ = pipeline.run(
                        input_ids=inputs.input_ids,
                        look_ahead_k=args.look_ahead_k, 
                        max_generation_length=max_gen,
                    )
            else:
                with torch.inference_mode():
                    # Standard generation call for all patched models
                    context_length = inputs.input_ids.shape[-1]
                    if args.mode == 'gemfilter':
                        from baseline.gemfilter.gemfilter_utils import gemfilter_generate_selection
                        pred = gemfilter_generate_selection(
                            inputs['input_ids'], inputs['attention_mask'], 
                            model, tokenizer, max_gen_len=max_gen, select_layer_idx=args.filter_idx
                        )
                    else:
                        output = model.generate(
                                **inputs, max_new_tokens=max_gen, num_beams=1,
                                do_sample=False, temperature=1.0, top_p=1.0,
                                )[0]
                        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)


            print(pred)
            
            with open(out_path, "a", encoding="utf-8") as f:
                json.dump({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]}, f, ensure_ascii=False)
                f.write('\n')
        except RuntimeError as e:
            if "out of memory" in str(e):
                logging.warning(f"[[CUDA OOM - Skipping]]")
                torch.cuda.empty_cache()
                continue
            else:
                raise e


# =====================================================================================
# 3. MAIN ORCHESTRATION
# =====================================================================================

def main(args):
    set_seed(args.seed)
 
    if args.save_path:
        save_path = os.path.join(f"outputs/{args.model}/longbench", args.save_path)
    else:
        tm = time.localtime(time.time())
        f_name = f"{tm.tm_year}_{tm.tm_mon}_{tm.tm_mday}_{tm.tm_hour}_{tm.tm_min}_{tm.tm_sec}"
        save_path = os.path.join(f"outputs/{args.model}/longbench", f_name)
    Path(save_path).mkdir(parents=True, exist_ok=True)  

    utils.config_logging(os.path.join(save_path, 'process.log'))
    logging.info('Arguments: \n' + pprint.pformat(vars(args)))
    logging.info('--' * 30)

    model2maxlen = json.load(open("eval/longbench/config/model2maxlen.json", "r"))
    dataset2prompt = json.load(open("eval/longbench/config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("eval/longbench/config/dataset2maxlen.json", "r"))
    
    model, pipeline, tokenizer = setup_model_and_tokenizer(args)
    
    dataset_name = args.dataset
    if args.longbench_type == "longbench-e":
        data = load_dataset('THUDM/LongBench', f"{dataset_name}_e", split='test', trust_remote_code=True)
    else:
        data = load_dataset('THUDM/LongBench', dataset_name, split='test', trust_remote_code=True)
    
    out_path = os.path.join(save_path, f"{dataset_name}.jsonl")
    prompt_format = dataset2prompt[dataset_name]
    max_gen = dataset2maxlen[dataset_name]
    max_length = model2maxlen[args.model]
    data_all = [data_sample for data_sample in data]
    
    # Apply limit if specified
    if args.limit is not None:
        data_all = data_all[:args.limit]
        logging.info(f"Limited dataset to {len(data_all)} samples")

    generate_longbench(
        data=data_all, max_length=max_length, max_gen=max_gen, 
        prompt_format=prompt_format, dataset=dataset_name, 
        model=model, pipeline=pipeline, tokenizer=tokenizer,
        out_path=out_path, args=args
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Model Arguments
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", help="model name of model path")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--save_path", default="", type=str, help="Path to save the output")

    # KV Compression & Prefill Modes
    parser.add_argument("--mode", type=str, default="fastkv", 
                        choices=["fullkv", "fastkv", "snapkv", "gemfilter", "adakv", "headkv", 
                                 "speculative_prefill", "echo_cache", "hfastkv", "draft_tsp", 
                                 "uniform", "taper", "claa", "oracle"])

    parser.add_argument("--window_size", type=int, default=8)
    parser.add_argument("--max_capacity_prompt", type=int, default=512)
    parser.add_argument("--max_capacity_prompt_percentage", type=float, default=None, help="Use a percentage of the prompt length for max capacity.")

    # Uniform
    parser.add_argument("--uniform_first_k", type=int, default=64, help="Number of initial tokens to always keep for uniform mode.")
    parser.add_argument("--uniform_last_k", type=int, default=256, help="Number of final tokens to always keep for uniform mode.")

    # KV Compression & Prefill Modes (+ Speculative Prefill and Echo Cache)
    parser.add_argument("--kernel_size", type=int, default=7, help="Pooling kernel size. Must be odd.")
    parser.add_argument("--pooling", type=str, default="avgpool", choices=['avgpool', 'maxpool', 'none'])
    
    # Speculative Prefill / Echo Cache
    parser.add_argument("--cache_granularity", type=str, default="head", choices=["global", "layer", "head"],
                    help="Pruning granularity for EchoCache: 'global' (single stage), "
                         "'layer' (global then per-layer), or 'head' (global then per-head).")
    parser.add_argument("--speculator_model_name", type=str, default="meta-llama/Llama-3-8B-Instruct", help="Speculator model for prefill/echocache. Ensure compatibility.")
    parser.add_argument("--look_ahead_k", type=int, default=1, help="Number of lookahead steps for Speculative Prefill.")
    parser.add_argument('--use_chunk_selection', action='store_true', help="Use chunk-based token selection (it's enabled by default).")
    parser.add_argument("--chunk_size", type=int, default=64, help="Chunk size for Speculative Prefill.")
    
    # FastKV
    parser.add_argument("--tsp_idx", type=int, default=15)
    parser.add_argument("--tsp_len", type=int, default=2048)
    parser.add_argument("--tsp_len_percentage", type=float, default=None, help="Use a percentage of the prompt length for TSP length.")
    # GemFilter
    parser.add_argument("--filter_idx", type=int, default=13)
    parser.add_argument("--topk", type=int, default=1024, help="Fixed number of tokens to keep for GemFilter")
    parser.add_argument("--topk_percentage", type=float, default=None, help="Use a percentage of the prompt length for GemFilter topk")
    parser.add_argument("--select_layer_idx", type=int, default=13, help="Layer index for GemFilter selection")
    # AdaKV
    parser.add_argument("--skip", type=int, default=-1)
    parser.add_argument('--floor_alpha', type=float, default=0.2)
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--pyram', action='store_true')
    parser.add_argument('--pyram_beta', default=20,type=int)
    parser.add_argument('--gqa_support', action='store_true')
    # HeadKV
    parser.add_argument("--method", type=str, default='ReasonKV', choices=['ReasonKV'])
    parser.add_argument("--head_choice", type=str, default='reason', choices=['copy', 'reason'])
    parser.add_argument('--beta', type=float, default=1.2)
    parser.add_argument('--temp', type=float, default=1.0)
    # TAPER
    parser.add_argument("--tsp_schedule", type=str, default="", 
                        help="Progressive TSP schedule. Format depends on mode: 'LAYER_IDX:KEEP_RATIO,...' e.g., '0:0.8,7:0.5'. ")
    # Oracle
    parser.add_argument("--oracle_rankings_path", type=str, default="", 
                        help="Path to directory containing oracle rankings (.npz files)")
    parser.add_argument("--oracle_model_name", type=str, default="", 
                        help="Model name for oracle rankings (e.g., meta-llama_Llama-3.1-8B-Instruct)")

    # Evaluation
    parser.add_argument('--dataset', type=str, default='qasper', help="Dataset to evaluate on")
    parser.add_argument('--longbench_type', type=str, default='longbench', choices=['longbench', 'longbench-e'])
    parser.add_argument('--limit', type=int, default=None, help="Limit number of samples to process (default: process all)")

    args = parser.parse_args()
    
    dataset_list = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", "gov_report", "qmsum", "multi_news", "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
    if args.longbench_type == "longbench-e":
        dataset_list = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
    if args.dataset not in dataset_list:
        raise ValueError(f"Dataset '{args.dataset}' not found in the supported list for {args.longbench_type}")

    main(args)
