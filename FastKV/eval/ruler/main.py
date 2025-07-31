# eval/ruler/main.py
import argparse
import json
import logging
import os
import pprint
import time
from pathlib import Path
from tqdm import tqdm
import importlib
import numpy as np

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from eval.ruler.utils import config_logging

def setup_model_and_tokenizer(args):
    """
    Handles all one-time setup for the chosen model and mode.
    This includes monkey-patching and loading the model/tokenizer.
    """
    logging.info(f"Setting up for mode: {args.mode}")
    
    # Load tokenizer once, as it's needed by all modes for prompt prep.
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
        logging.info("Set tokenizer pad_token to eos_token")

    model = None
    pipeline = None

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

    return model, pipeline, tokenizer

def prepare_and_save_data(args, task_config, tokenizer, seq_len, save_dir):
    """
    Generates and saves the data for a given task and sequence length.
    Returns the path to the generated data file.
    """
    base_task_module = task_config['base_task']
    output_file = save_dir / f"{args.task}.jsonl"

    try:
        # Dynamically import the data generator module (e.g., eval.ruler.data.qa)
        data_generator_module = importlib.import_module(f"eval.ruler.data.{base_task_module}")
    except ImportError:
        logging.error(f"Could not find data generator module: eval/ruler/data/{base_task_module}.py")
        return None

    logging.info(f"Generating {args.num_samples} samples for task '{args.task}'...")
    
    with open(output_file, "w") as f:
        for i in tqdm(range(args.num_samples), desc=f"Generating data for {seq_len}"):
            # Each data generator's create_instance function is responsible for creating
            # the raw data (context, query, etc.) needed for the prompt.
            instance = data_generator_module.create_instance(
                tokenizer=tokenizer,
                max_length=seq_len,
                task_args=task_config['args'],
                prompt_template=task_config['prompt_template'],
                instance_index=i
            )
            instance['index'] = i
            f.write(json.dumps(instance) + "\n")

    logging.info(f"Data saved to {output_file}")
    return output_file

def build_chat(tokenizer, user_content, assistant_prefix):
    """
    Applies the chat template with a completion-style format, correctly handling
    a pre-filled assistant response.
    """
    messages = [
        {"role": "user", "content": user_content},
    ]
    if assistant_prefix:
        # If there is a prefix, add it as an incomplete assistant message.
        messages.append({"role": "assistant", "content": assistant_prefix})
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            continue_final_message=True
        )
    else:
        # If there is no prefix, we want the model to generate a response from scratch.
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def generate_and_save_predictions(model, pipeline, tokenizer, data_file, task_config, save_dir, args):
    """
    Reads a data file, formats the prompt correctly, runs generation, and saves predictions.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model is not None:
        device = model.device
    elif pipeline is not None:
        device = pipeline.device
        
    output_file = save_dir / f"{data_file.stem}.jsonl"
    
    # Load templates from the task configuration
    prompt_template = task_config['prompt_template']
    answer_prefix_template = task_config.get('answer_prefix', '') # Safely get prefix
    max_gen = task_config['max_gen']
    
    with open(data_file, "r", encoding="utf-8") as f:
        all_data = [json.loads(line) for line in f]

    logging.info(f"Running generation for {len(all_data)} samples...")
    with open(output_file, "w", encoding="utf-8") as f_out:
        for i, data_point in enumerate(tqdm(all_data, desc=f"Generating predictions")):
            try:
                if args.mode == 'oracle':
                    from baseline.oracle.oracle_utils import set_oracle_rankings
                    
                    # Construct path to oracle rankings file
                    oracle_file = os.path.join(args.oracle_rankings_path, args.oracle_model_name.replace("/", "_"), f"{args.task}.npz")
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
                        
                # 1. Create the dictionary of arguments for formatting the templates.
                # By copying the whole data_point, we flexibly handle all keys required
                # by any task (niah, qa, vt, etc.).
                format_kwargs = data_point.copy()

                # 2. Assemble the user message and the assistant's prefix.
                user_content = prompt_template.format(**format_kwargs)
                assistant_prefix = answer_prefix_template.format(**format_kwargs)
                
                # 3. Apply the chat template to the assembled content.
                final_prompt_for_model = build_chat(tokenizer, user_content, assistant_prefix)
                
                if i == 0:
                    logging.info(f"--- DEBUG: First full prompt being sent to model ---\n{final_prompt_for_model}\n----------------------------------------------------")

                inputs = tokenizer(final_prompt_for_model, return_tensors="pt", truncation=False).to(device)
                
                pred_text = ""
                
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
                    pred_text = pred.strip()
                else:
                    with torch.inference_mode():
                        # Standard generation call for all patched models
                        context_length = inputs.input_ids.shape[-1]
                        if args.mode == 'gemfilter':
                            from baseline.gemfilter.gemfilter_utils import gemfilter_generate_selection
                            pred_text = gemfilter_generate_selection(
                                inputs['input_ids'], inputs['attention_mask'], 
                                model, tokenizer, max_gen_len=max_gen, select_layer_idx=args.filter_idx
                            )
                        else:
                            output_ids = model.generate(
                                **inputs,
                                max_new_tokens=max_gen,
                                num_beams=1,
                                do_sample=False,
                                eos_token_id=tokenizer.eos_token_id,
                                pad_token_id=tokenizer.pad_token_id
                            )[0]
                            pred_text = tokenizer.decode(output_ids[context_length:], skip_special_tokens=True)

                result = {
                    "index": data_point["index"],
                    "pred": pred_text.strip(),
                    "outputs": data_point["outputs"],
                    "metadata": data_point.get("metadata", {})
                }
                f_out.write(json.dumps(result) + "\n")
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logging.warning(f"[[CUDA OOM - Skipping sample {i}]]")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
    
    logging.info(f"Predictions saved to {output_file}")
    return output_file

def run_benchmark(args, model, pipeline, tokenizer):
    """
    Orchestrates the data preparation, generation, and evaluation.
    """
    logging.info(f"--- Starting RULER Benchmark for Task: {args.task} ---")

    with open("eval/ruler/config/task_details.json", "r") as f:
        task_details = json.load(f)
    with open("eval/ruler/config/model2maxlen.json", "r") as f:
        model2maxlen = json.load(f)

    if args.task not in task_details:
        raise ValueError(f"Task '{args.task}' not found in task_details.json. Available tasks: {list(task_details.keys())}")

    task_config = task_details[args.task]
    max_context_len = model2maxlen.get(args.model, 4096)
    
    logging.info(f"Model: {args.model} | Max Context Length: {max_context_len}")
    logging.info(f"Task Config: {pprint.pformat(task_config)}")

    seq_lengths = [args.seq_length]
    
    for seq_len in seq_lengths:
        if seq_len > max_context_len:
            logging.info(f"Skipping sequence length {seq_len} as it exceeds model's max length {max_context_len}.")
            continue

        logging.info(f"\n===== Running for Sequence Length: {seq_len} =====\n")
        
        seq_len_save_path = Path(f"outputs/{args.model}/ruler/{args.save_path}/{seq_len}")
        data_save_dir = seq_len_save_path / "data"
        pred_save_dir = seq_len_save_path / "pred"
        data_save_dir.mkdir(parents=True, exist_ok=True)
        pred_save_dir.mkdir(parents=True, exist_ok=True)
        
        data_file = prepare_and_save_data(args, task_config, tokenizer, seq_len, data_save_dir)
        
        if not data_file:
            logging.error("Data preparation failed. Aborting.")
            continue
        
        generate_and_save_predictions(
            model=model,
            pipeline=pipeline,
            tokenizer=tokenizer,
            data_file=data_file,
            task_config=task_config,
            save_dir=pred_save_dir,
            args=args,
        )

    logging.info("--- RULER Benchmark Run Finished ---")

def main():
    parser = argparse.ArgumentParser()
    # Model Arguments
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", help="model name of model path")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the output")

    # KV Compression & Prefill Modes
    parser.add_argument("--mode", type=str, default="fullkv", 
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
    # CLAA
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Alpha parameter for CLAA value magnitude scaling.")
    # Oracle
    parser.add_argument("--oracle_rankings_path", type=str, default="", 
                        help="Path to directory containing oracle rankings (.npz files)")
    parser.add_argument("--oracle_model_name", type=str, default="", 
                        help="Model name for oracle rankings (e.g., meta-llama_Llama-3.1-8B-Instruct)")

    # Evaluation - RULER specific
    parser.add_argument("--task", type=str, default="niah_single_1", help="RULER task to evaluate.")
    parser.add_argument("--num_samples", type=int, default=500, help="Number of samples to generate per task length.")
    parser.add_argument("--seq_length", type=int, default=2048, help="Sequence length to evaluate.")
    
    # Additional arguments for compatibility
    parser.add_argument("--keep_percentage", type=float, default=0.1, help="Percentage of tokens to keep for uniform mode.")
    
    args = parser.parse_args()

    save_path = Path(f"outputs/{args.model}/ruler/{args.save_path}")
    save_path.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    config_logging(save_path / "process.log")
    logging.info("Arguments:\n" + pprint.pformat(vars(args)))
    logging.info("-" * 30)

    model, pipeline, tokenizer = setup_model_and_tokenizer(args)
    run_benchmark(args, model, pipeline, tokenizer)


if __name__ == "__main__":
    main()