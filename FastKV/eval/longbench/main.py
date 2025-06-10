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

from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from datasets import load_dataset

from utils import utils

def build_chat(tokenizer, prompt, model_name):
    if "Llama-2" in model_name:
        prompt = f"[INST]{prompt}[/INST]"
    else:
        messages = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    return prompt
            
@torch.inference_mode()
def generate_longbench(data, max_length, max_gen, prompt_format, 
                       dataset, model, tokenizer,
                       out_path, args):

    if args.mode in ["speculative_prefill", "echo_cache"]:
        if args.mode == "speculative_prefill":
            from baseline.speculative_prefill.main import SpeculativePrefillPipeline as Pipeline
        else: # echo_cache
            from baseline.echo_cache.main import EchoCachePipeline as Pipeline
        
        # The pipeline handles its own model and tokenizer loading.
        pipeline = Pipeline(
            base_model_name=args.model,
            speculator_model_name=args.speculator_model_name,
            max_capacity_prompt=args.max_capacity_prompt
        )
        for json_obj in tqdm(data, desc=f"Generating Responses for {args.mode}..."):
            prompt = prompt_format.format(**json_obj)
            pred, _ = pipeline.run(prompt, look_ahead_k=1, max_generation_length=max_gen)

            with open(out_path, "a", encoding="utf-8") as f:
                json.dump({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]}, f, ensure_ascii=False)
                f.write('\n')
    else:
        # Standard logic for all other modes
        device = model.device
        for json_obj in tqdm(data, desc=f"Generating Responses for {args.mode}..."):
            # load compress args
            if args.mode == 'fastkv':
                from baseline.fastkv.fastkv_utils import compress
                compress(model, args)
            elif args.mode == 'snapkv':
                from baseline.snapkv.snapkv_utils import compress
                compress(model, args)
            elif args.mode == 'gemfilter':
                from baseline.gemfilter.gemfilter_utils import gemfilter_generate_selection, set_topk
                set_topk(model, args.max_capacity_prompt, mode='gemfilter')
            elif args.mode == 'adakv':
                from baseline.adakv.adaptive_snapkv.snapkv_utils import compress
                compress(model, args)
            elif args.mode == 'headkv':
                from baseline.headkv.headkv.snapkv_utils import compress
                compress(model, args)

            prompt = prompt_format.format(**json_obj)

            # truncate to fit max_length
            tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
            if len(tokenized_prompt) > max_length:
                half = int(max_length/2)
                prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
            
            if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]: 
                prompt = build_chat(tokenizer, prompt, args.model)

            input_ids = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
            context_length = input_ids.input_ids.shape[-1]

            if args.mode == 'gemfilter':
                pred = gemfilter_generate_selection(input_ids['input_ids'], input_ids['attention_mask'], 
                    model, tokenizer, max_gen_len=max_gen, select_layer_idx=args.filter_idx)
            else:
                output = model.generate(
                        **input_ids,
                        max_new_tokens=max_gen,
                        num_beams=1,
                        do_sample=False,
                        temperature=1.0,
                        top_p=1.0,
                        )[0]
                pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
            
            with open(out_path, "a", encoding="utf-8") as f:
                json.dump({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]}, f, ensure_ascii=False)
                f.write('\n')

def main(args):
    set_seed(args.seed)
 
    if args.save_path:
        save_path = os.path.join(f"outputs/{args.model}/longbench", args.save_path)
    else:
        tm = time.localtime(time.time())
        f_name = f"{tm.tm_year}_{tm.tm_mon}_{tm.tm_mday}_{tm.tm_hour}_{tm.tm_min}_{tm.tm_sec}"
        save_path = os.path.join(f"outputs/{args.model}/longbench", f_name)
    Path(save_path).mkdir(parents=True, exist_ok=True)  

    utils.config_logging(os.path.join(save_path, f'process.log'))
    logging.info('Arguments: ')
    logging.info(pprint.pformat(vars(args)))
    logging.info('--' * 30)

    model2maxlen = json.load(open("eval/longbench/config/model2maxlen.json", "r"))
    max_length = model2maxlen[args.model]

    dataset_list = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", "gov_report", "qmsum", "multi_news", "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
    if args.longbench_type == "longbench-e":
        dataset_list = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]

    if args.dataset not in dataset_list:
        raise ValueError(f"Dataset {args.dataset} not found in the supported list for {args.longbench_type}")
    
    dataset2prompt = json.load(open("eval/longbench/config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("eval/longbench/config/dataset2maxlen.json", "r"))

    dataset = args.dataset
    model, tokenizer = None, None

    # For pipeline-based modes, model and tokenizer are loaded inside the pipeline.
    # For other modes, load them here.
    if args.mode not in ["speculative_prefill", "echo_cache"]:
        if args.mode == 'fullkv':
            from baseline.fullkv.monkeypatch import replace_llama, replace_mistral
            replace_llama()
            replace_mistral()
        elif args.mode == 'fastkv':
            from baseline.fastkv.monkeypatch import replace_llama, replace_mistral
            replace_llama()
            replace_mistral()
        elif args.mode == 'snapkv':
            from baseline.snapkv.monkeypatch import replace_llama, replace_mistral, replace_phi3
            replace_llama()
            replace_mistral()
            replace_phi3()
        elif args.mode == 'gemfilter':
            from baseline.gemfilter.monkeypatch import replace_llama, replace_mistral
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

        logging.info(f'Loading Model & Tokenizer for {args.mode}...')
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(args.model, device_map='auto', attn_implementation='flash_attention_2', torch_dtype=torch.float16)
        model.eval()

    if args.longbench_type == "longbench-e":
        data = load_dataset('THUDM/LongBench', f"{dataset}_e", split='test')
    else:
        data = load_dataset('THUDM/LongBench', dataset, split='test')

    out_path = os.path.join(save_path, f"{dataset}.jsonl")
    prompt_format = dataset2prompt[dataset]
    max_gen = dataset2maxlen[dataset]
    data_all = [data_sample for data_sample in data]

    generate_longbench(data=data_all, max_length=max_length, max_gen=max_gen, prompt_format=prompt_format, 
                       dataset=dataset, model=model, tokenizer=tokenizer,
                       out_path=out_path, args=args)
 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Model Arguments
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", help="model name of model path")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--save_path", default="", type=str, help="Path to save the output")

    # KV Compression & Prefill Modes
    parser.add_argument("--mode", type=str, default="fastkv", choices=["fullkv", "fastkv", "snapkv", "gemfilter", "adakv", "headkv", "speculative_prefill", "echo_cache"])
    parser.add_argument("--window_size", type=int, default=8)
    parser.add_argument("--max_capacity_prompt", type=int, default=512)
    parser.add_argument("--kernel_size", type=int, default=7)
    parser.add_argument("--pooling", type=str, default="avgpool")
    
    # Speculative Prefill / Echo Cache
    parser.add_argument("--speculator_model_name", type=str, default="meta-llama/Llama-3-8B-Instruct", help="Speculator model for prefill/echocache. Ensure compatibility.")
    
    # FastKV
    parser.add_argument("--tsp_idx", type=int, default=15)
    parser.add_argument("--tsp_len", type=int, default=2048)
    # GemFilter
    parser.add_argument("--filter_idx", type=int, default=13)
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
    
    # Evaluation
    parser.add_argument('--dataset', type=str, default='qasper', help="Dataset to evaluate on")
    parser.add_argument('--longbench_type', type=str, default='longbench', choices=['longbench', 'longbench-e'])

    args = parser.parse_args()
    main(args)
