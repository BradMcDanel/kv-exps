# review_prompts.py

import argparse
import json
import os
from typing import Dict, Any

from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

def build_chat(tokenizer: AutoTokenizer, prompt: str) -> str:
    """
    Applies the model's chat template to a raw prompt string.
    This logic is based on generate_oracles.py for robust formatting.
    """
    messages = [{"role": "user", "content": prompt}]
    # `add_generation_prompt=True` is crucial to get the final `assistant` turn marker
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

def main():
    """
    Main function to load data, format a specific prompt, and display it for review.
    """
    parser = argparse.ArgumentParser(
        description="Review formatted prompts and answers for LongBench datasets.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="The name of the LongBench dataset to review (e.g., 'qasper', 'trec')."
    )
    parser.add_argument(
        "--sample_idx",
        type=int,
        required=True,
        help="The index of the sample to inspect within the dataset."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="The model name, used to load the correct tokenizer for chat templating."
    )
    parser.add_argument(
        '--longbench_type', 
        type=str, 
        default='longbench', 
        choices=['longbench', 'longbench-e'],
        help="The type of LongBench split to use ('longbench' or 'longbench-e')."
    )

    args = parser.parse_args()

    # --- 1. Load Resources ---
    # Construct path to the prompt configuration file
    try:
        config_path = os.path.join('eval', 'longbench', 'config', 'dataset2prompt.json')
        with open(config_path, "r") as f:
            dataset2prompt = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find '{config_path}'. Make sure you are running this script from the project's root directory.")
        return

    # Load tokenizer
    print(f"Loading tokenizer for '{args.model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    # Load dataset from Hugging Face Hub
    print(f"Loading dataset '{args.dataset}' (type: {args.longbench_type})...")
    try:
        dataset_name_for_hub = f"{args.dataset}_e" if args.longbench_type == "longbench-e" else args.dataset
        data: Dataset = load_dataset('THUDM/LongBench', dataset_name_for_hub, split='test')
    except Exception as e:
        print(f"Error loading dataset '{args.dataset}' from THUDM/LongBench: {e}")
        return

    # --- 2. Select and Process Sample ---
    if not (0 <= args.sample_idx < len(data)):
        print(f"Error: --sample_idx {args.sample_idx} is out of bounds. The dataset '{args.dataset}' has {len(data)} samples (0 to {len(data)-1}).")
        return

    sample: Dict[str, Any] = data[args.sample_idx]

    # Get the specific prompt format for this dataset
    if args.dataset not in dataset2prompt:
        print(f"Error: No prompt format found for '{args.dataset}' in '{config_path}'.")
        return
    prompt_format = dataset2prompt[args.dataset]

    # --- 3. Format the Prompt ---
    try:
        raw_prompt = prompt_format.format(**sample)
    except KeyError as e:
        print(f"Error formatting prompt for dataset '{args.dataset}'. A required key is missing: {e}")
        print("Available keys in sample:", list(sample.keys()))
        return

    # Datasets that should NOT use a chat template (based on `generate_oracles.py` and `main.py`)
    datasets_without_chat_template = ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]

    if args.dataset not in datasets_without_chat_template:
        final_prompt_text = build_chat(tokenizer, raw_prompt)
    else:
        final_prompt_text = raw_prompt

    # --- 4. Display Results ---
    ground_truth_answers = sample.get("answers", "N/A")

    print("\n" + "="*70)
    print(f"{'REVIEWING LONG BENCH SAMPLE':^70}")
    print("="*70)
    print(f"- Dataset:        {args.dataset} ({args.longbench_type})")
    print(f"- Sample Index:   {args.sample_idx}")
    print(f"- Model Template: {args.model_name}")
    print("-"*70)
    print(f"{'FINAL PROMPT (as fed to the model)':^70}")
    print("-"*70)
    print(final_prompt_text)
    print("-"*70)
    print(f"{'GROUND TRUTH ANSWER(S)':^70}")
    print("-"*70)
    print(ground_truth_answers)
    print("="*70)


if __name__ == "__main__":
    main()
