import os
import argparse
import json
import time
from typing import List, Dict, Tuple, Optional, Any

import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.cache_utils import Cache, DynamicCache

class OraclePrefillPipeline:
    """
    Implements a pipeline that uses pre-computed oracle rankings to perform
    a selective prefill on a base model. This simulates a "perfect" KV cache
    compression scenario to measure its upper-bound performance.
    """
    def __init__(
        self,
        base_model_name: str,
        tokenizer: AutoTokenizer,
        oracle_rankings_path: str,
        keep_percentage: float = 0.1,
        detailed_timing: bool = True,
    ):
        self.base_model_name = base_model_name
        self.tokenizer = tokenizer
        self.base_model = self._load_model(self.base_model_name)
        self.base_config: AutoConfig = self.base_model.config
        self.device = self.base_model.device
        self.dtype = self.base_model.dtype
        
        self.oracle_rankings_path = oracle_rankings_path
        self.keep_percentage = keep_percentage
        self.detailed_timing = detailed_timing
        
        self._validate_config()
        self.eos_token_ids = self._extract_eos_token_ids()

    def _validate_config(self):
        if not (0.0 < self.keep_percentage <= 1.0):
            raise ValueError("`keep_percentage` must be between 0.0 and 1.0.")
        if not os.path.isdir(self.oracle_rankings_path):
            raise FileNotFoundError(f"Oracle rankings path not found: {self.oracle_rankings_path}")

    def _load_model(self, model_name: str) -> AutoModelForCausalLM:
        print(f"Loading base model: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto", attn_implementation="sdpa"
        )
        return model.eval()

    def _extract_eos_token_ids(self) -> List[int]:
        config_eos = self.base_config.eos_token_id
        if isinstance(config_eos, int): return [config_eos]
        if isinstance(config_eos, list): return list(config_eos)
        if self.tokenizer.eos_token_id: return [self.tokenizer.eos_token_id]
        raise ValueError("eos_token_id not defined in config or tokenizer.")

    def _load_oracle_ranking(self, model_name: str, dataset_name: str, sample_key: str) -> Optional[Dict[str, np.ndarray]]:
        """Loads the oracle data for a specific sample."""
        model_name_sanitized = model_name.replace('/', '_')
        file_path = os.path.join(self.oracle_rankings_path, model_name_sanitized, f"{dataset_name}.npz")
        
        if not os.path.exists(file_path):
            print(f"Warning: Oracle ranking file not found at: {file_path}")
            return None
        
        try:
            with np.load(file_path, allow_pickle=True) as npz_file:
                if sample_key in npz_file:
                    # .item() extracts the object array (which is a dict)
                    return npz_file[sample_key].item()
                else:
                    print(f"Warning: Sample key '{sample_key}' not found in {file_path}")
                    return None
        except Exception as e:
            print(f"Error loading oracle data from {file_path}: {e}")
            return None

    @torch.no_grad()
    def run(
        self,
        input_ids: torch.Tensor,
        oracle_model_for_path: str, # Model name used to generate the oracle (e.g., meta-llama/Llama-3.1-8B-Instruct)
        dataset_name: str,
        sample_key: str,
        max_generation_length: int,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Runs the oracle-guided prefill and generation.

        Args:
            input_ids: The full prompt token IDs.
            oracle_model_for_path: The model name used to create the oracle file path.
            dataset_name: The name of the dataset for the sample.
            sample_key: The key identifying the sample within the oracle .npz file.
            max_generation_length: The maximum number of tokens to generate.
        
        Returns:
            A tuple of the generated text and a dictionary of metadata.
        """
        run_metadata: Dict[str, Any] = {}
        
        if self.detailed_timing:
            start_time = time.time()
            prefill_start_event = torch.cuda.Event(enable_timing=True)
            prefill_end_event = torch.cuda.Event(enable_timing=True)
            decode_start_event = torch.cuda.Event(enable_timing=True)
            decode_end_event = torch.cuda.Event(enable_timing=True)

        # --- Stage 1: Load Oracle and Select Tokens ---
        prompt_length = input_ids.shape[1]
        oracle_data = self._load_oracle_ranking(oracle_model_for_path, dataset_name, sample_key)

        if oracle_data is None:
            raise RuntimeError(f"Failed to load oracle data for {dataset_name}/{sample_key}")

        # Sanity check
        oracle_input_ids = torch.from_numpy(oracle_data['input_ids']).to(self.device).unsqueeze(0)
        if not torch.equal(input_ids, oracle_input_ids):
            raise ValueError("Input IDs from prompt do not match Input IDs from the oracle file!")

        ranking_scores = torch.from_numpy(oracle_data['ranking']).to(self.device)
        num_to_keep = int(prompt_length * self.keep_percentage)

        if num_to_keep >= prompt_length:
            indices_to_keep = torch.arange(prompt_length, device=self.device, dtype=torch.long)
        else:
            _, top_k_indices = torch.topk(ranking_scores, k=num_to_keep, dim=-1)
            indices_to_keep = torch.sort(top_k_indices).values
        
        selected_prompt_ids = input_ids[:, indices_to_keep]
        
        # This is the crucial step: use the original positions for RoPE.
        selective_pos_ids = indices_to_keep.unsqueeze(0).to(torch.long)
        
        run_metadata["prompt_input_length"] = prompt_length
        run_metadata["selective_prefill_kept_token_count"] = selected_prompt_ids.shape[1]
        run_metadata["token_keep_rate"] = (selected_prompt_ids.shape[1] / prompt_length * 100.0) if prompt_length > 0 else 100.0

        # --- Stage 2: Base Model Selective Prefill ---
        if self.detailed_timing: prefill_start_event.record()

        selective_cache_pos = torch.arange(selected_prompt_ids.shape[1], device=self.device)
        base_out = self.base_model(
            selected_prompt_ids,
            position_ids=selective_pos_ids,
            use_cache=True,
            cache_position=selective_cache_pos
        )
        base_model_next_token_ids = torch.argmax(base_out.logits[:, -1, :], dim=-1, keepdim=True)
        base_model_cache_after_prefill = base_out.past_key_values

        if self.detailed_timing:
            prefill_end_event.record()
            torch.cuda.synchronize()
            run_metadata["base_prefill_time"] = prefill_start_event.elapsed_time(prefill_end_event) / 1000.0
            run_metadata["ttft"] = run_metadata["base_prefill_time"]
            decode_start_event.record()


        # --- Stage 3: Base Model Generation (Decoding) ---
        gen_token_ids_list: List[int] = []
        if base_model_next_token_ids is not None and base_model_cache_after_prefill is not None:
            first_gen_token_id = base_model_next_token_ids.item()
            gen_token_ids_list.append(first_gen_token_id)
            
            current_decode_tokens = base_model_next_token_ids
            current_decode_kv_cache = DynamicCache.from_legacy_cache(base_model_cache_after_prefill)

            if first_gen_token_id not in self.eos_token_ids:
                # The generation must continue from the position of the LAST token in the original sequence
                # that we kept, not from the length of the compressed cache.
                start_pos_for_generation = (selective_pos_ids[0, -1] + 1).item() if selective_pos_ids.numel() > 0 else 0

                for i in range(max_generation_length - 1):
                    current_cache_len = current_decode_kv_cache.get_seq_length(0)
                    
                    # Position IDs for generation continue from the original sequence
                    pos_ids = torch.tensor([[start_pos_for_generation + i]], device=self.device, dtype=torch.long)
                    decode_cache_pos = torch.tensor([current_cache_len], device=self.device)
                    
                    decode_out = self.base_model(
                        current_decode_tokens,
                        position_ids=pos_ids,
                        past_key_values=current_decode_kv_cache,
                        use_cache=True,
                        cache_position=decode_cache_pos
                    )

                    next_tokens = torch.argmax(decode_out.logits[:, -1, :], dim=-1, keepdim=True)
                    gen_token_ids_list.append(next_tokens.item())
                    current_decode_kv_cache = DynamicCache.from_legacy_cache(decode_out.past_key_values)
                    current_decode_tokens = next_tokens

                    if gen_token_ids_list[-1] in self.eos_token_ids:
                        break

        if self.detailed_timing:
            decode_end_event.record()
            torch.cuda.synchronize()
            run_metadata["decode_time"] = decode_start_event.elapsed_time(decode_end_event) / 1000.0
            run_metadata["total_time"] = time.time() - start_time

        final_gen_text = self.tokenizer.decode(gen_token_ids_list, skip_special_tokens=True)
        return final_gen_text, run_metadata


def main():
    parser = argparse.ArgumentParser(description="Run KV Cache compression using pre-computed Oracle rankings.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--base_model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--oracle_rankings_path", type=str, default="analysis_results/oracles", help="Path to the root directory of oracle .npz files.")
    parser.add_argument("--dataset_name", type=str, default="qasper", help="Name of the LongBench dataset to test.")
    parser.add_argument("--sample_idx_in_file", type=int, default=0, help="The index of the sample to use from the dataset's oracle file.")
    parser.add_argument("--keep_percentage", type=float, default=0.05, help="Percentage of prompt tokens to keep for prefill (e.g., 0.05 for 5%).")
    parser.add_argument("--max_generation_length", type=int, default=128, help="Maximum number of tokens to generate.")
    parser.add_argument("--max_prompt_len", type=int, default=8192, help="Maximum prompt length to consider.")
    
    args = parser.parse_args()
    
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
        
    pipeline = OraclePrefillPipeline(
        base_model_name=args.base_model_name,
        tokenizer=tokenizer,
        oracle_rankings_path=args.oracle_rankings_path,
        keep_percentage=args.keep_percentage,
    )

    # --- Load Prompt and Find Sample Key ---
    try:
        # Construct path relative to this script's location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
        prompt_config_path = os.path.join(project_root, 'eval', 'longbench', 'config', 'dataset2prompt.json')
        
        with open(prompt_config_path, "r") as f:
            dataset2prompt = json.load(f)
        
        data = load_dataset('THUDM/LongBench', args.dataset_name, split='test', trust_remote_code=True)
        
        # Find the sample key corresponding to the desired index
        oracle_file_path = os.path.join(args.oracle_rankings_path, args.base_model_name.replace('/', '_'), f"{args.dataset_name}.npz")
        if not os.path.exists(oracle_file_path):
            raise FileNotFoundError(f"Oracle file not found. Please generate it first: {oracle_file_path}")
            
        with np.load(oracle_file_path, allow_pickle=True) as f:
            available_keys = sorted(f.files)
        
        if args.sample_idx_in_file >= len(available_keys):
            raise IndexError(f"sample_idx_in_file {args.sample_idx_in_file} is out of bounds for {len(available_keys)} available samples.")
        
        target_sample_key = available_keys[args.sample_idx_in_file]
        # The key is like 'sample_XX', so we extract the index XX
        original_data_index = int(target_sample_key.split('_')[1])
        sample = data[original_data_index]

        raw_prompt = dataset2prompt[args.dataset_name].format(**sample)
        messages = [{"role": "user", "content": raw_prompt}]
        templated_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    except (FileNotFoundError, IndexError, KeyError) as e:
        print(f"Error loading data or prompt: {e}")
        return

    inputs = tokenizer(templated_prompt, return_tensors="pt", truncation=True, max_length=args.max_prompt_len).to(pipeline.device)

    print(f"\n--- Running Oracle Prefill Pipeline ---")
    print(f"Base Model: {args.base_model_name}")
    print(f"Dataset: {args.dataset_name} | Sample Key: {target_sample_key}")
    print(f"Keep Percentage: {args.keep_percentage:.1%}")
    print("-" * 43)

    generated_text, run_metadata = pipeline.run(
        input_ids=inputs.input_ids,
        oracle_model_for_path=args.base_model_name,
        dataset_name=args.dataset_name,
        sample_key=target_sample_key,
        max_generation_length=args.max_generation_length,
    )
    
    print(f"\n--- Generated Text ---")
    print(generated_text)
    print("-" * 22)

    print("\n--- Performance Metrics ---")
    print(f"Prompt Length (Original): {run_metadata.get('prompt_input_length', 'N/A')} tokens")
    print(f"Kept for Prefill (Compressed): {run_metadata.get('selective_prefill_kept_token_count', 'N/A')} tokens")
    print(f"Token Keep Rate: {run_metadata.get('token_keep_rate', 0):.2f}%")
    print(f"Time to First Token (TTFT): {run_metadata.get('ttft', 0):.4f} seconds")
    print(f"  - Base Model Prefill Time: {run_metadata.get('base_prefill_time', 0):.4f} s")
    print(f"Decoding Time: {run_metadata.get('decode_time', 0):.4f} seconds")
    print(f"Total Pipeline Time: {run_metadata.get('total_time', 0):.4f} seconds")
    print("-" * 27)

if __name__ == "__main__":
    main()
