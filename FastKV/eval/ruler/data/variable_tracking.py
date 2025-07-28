import json
import random
import string
import numpy as np
import heapq
from nltk.tokenize import sent_tokenize
import os
import re

def generate_chains(num_chains, num_hops, is_icl=False):
    """Exact copy from original RULER"""
    vars_all = []
    k = 5 if not is_icl else 3
    num_hops = num_hops if not is_icl else min(10, num_hops)
    vars_all = [''.join(random.choices(string.ascii_uppercase, k=k)).upper() for _ in range((num_hops+1) * num_chains)]
    while len(set(vars_all)) < num_chains * (num_hops+1):
        vars_all.append(''.join(random.choices(string.ascii_uppercase, k=k)).upper())

    vars_ret = []
    chains_ret = []
    for i in range(0, len(vars_all), num_hops+1):
        this_vars = vars_all[i:i+num_hops+1]
        vars_ret.append(this_vars)
        if is_icl:
            this_chain = [f"VAR {this_vars[0]} = 12345"]
        else:
            this_chain = [f"VAR {this_vars[0]} = {str(np.random.randint(10000, 99999))}"]
        for j in range(num_hops):
            this_chain.append(f"VAR {this_vars[j+1]} = VAR {this_vars[j]} ")
        chains_ret.append(this_chain)
    return vars_ret, chains_ret

def shuffle_sublists_heap(lst):
    heap = []
    for i in range(len(lst)):
        heapq.heappush(heap, (random.random(), i, 0))
    shuffled_result = []
    while heap:
        _, list_idx, elem_idx = heapq.heappop(heap)
        shuffled_result.append(lst[list_idx][elem_idx])
        if elem_idx + 1 < len(lst[list_idx]):
            heapq.heappush(heap, (random.random(), list_idx, elem_idx + 1))
    return shuffled_result

def create_instance(tokenizer, max_length, task_args, prompt_template, instance_index=None):
    """
    Create a single variable tracking instance - matches original RULER exactly.
    """
    type_haystack = task_args.get('type_haystack')
    if not type_haystack:
        raise ValueError("`type_haystack` must be specified in task_args")
    num_chains = task_args.get('num_chains', 1)
    num_hops = task_args.get('num_hops', 4)
    
    # Load essay data - NO FALLBACKS
    if type_haystack == 'essay':
        essay_path = os.path.join(os.path.dirname(__file__), "../RULER/scripts/data/synthetic/json/PaulGrahamEssays.json")
        with open(essay_path, 'r') as f:
            essay_data = json.load(f)
            haystack = re.sub(r'\s+', " ", essay_data['text']).split(" ")
    elif type_haystack == 'noise':
        haystack = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
    else:
        raise NotImplementedError(f'{type_haystack} is not implemented.')
    
    # Use original RULER's generate_input_output logic
    tokens_to_generate = 30  # From original RULER constants
    max_seq_length = max_length - tokens_to_generate
    
    # Generate ICL example
    _, icl_context, icl_vars, icl_value = generate_input_output(
        tokenizer, 256, num_chains, num_hops, type_haystack, haystack, prompt_template, is_icl=True
    )
    icl_prompt = prompt_template.format(
        context=icl_context,
        query=icl_value,
        num_v=len(icl_vars[0]) if icl_vars else 0
    ) + f" {', '.join(icl_vars[0])}"

    # Binary search for optimal noise count
    _, context, vars_list, value = generate_input_output(
        tokenizer, max_seq_length, num_chains, num_hops, type_haystack, haystack, prompt_template, icl_prompt=icl_prompt
    )
    
    # Find variables assigned the queried value
    query_value = value
    assigned_vars = []
    if vars_list:
        # The first chain (index 0) is the one with the value we are looking for.
        # The ground truth is all variables in that chain.
        assigned_vars = vars_list[0]
    
    return {
        "context": context,
        "query": query_value,
        "num_v": len(assigned_vars),
        "outputs": assigned_vars,
        "metadata": {
            "type_haystack": type_haystack,
            "num_chains": num_chains,
            "num_hops": num_hops,
            "value": query_value,
            "total_variables": len(assigned_vars)
        }
    }

def generate_input_output(tokenizer, max_seq_length, num_chains, num_hops, type_haystack, haystack, template, icl_prompt=None, is_icl=False):
    """Exact copy of original RULER's generate_input_output function"""
    DEPTHS = list(np.round(np.linspace(0, 100, num=40, endpoint=True)).astype(int))
    
    def generate_input_output_inner(num_noises, is_icl=False):
        vars_list, chains = generate_chains(num_chains, num_hops, is_icl=is_icl)
        value = chains[0][0].split("=")[-1].strip()

        if type_haystack == 'essay':
            text = " ".join(haystack[:num_noises])
            document_sents = sent_tokenize(text.strip())
            chains_flat = shuffle_sublists_heap(chains)
            insertion_positions = [0] + \
                                  sorted([int(len(document_sents) * (depth / 100)) for depth in random.sample(DEPTHS, len(chains_flat))]) + \
                                  [len(document_sents)]
            document_sents_list = []
            for i in range(1, len(insertion_positions)):
                last_pos = insertion_positions[i-1]
                next_pos = insertion_positions[i]
                document_sents_list.append(" ".join(document_sents[last_pos:next_pos]))
                if i-1 < len(chains_flat):
                    document_sents_list.append(chains_flat[i-1].strip() + ".")
            context = " ".join(document_sents_list)

        elif type_haystack == 'noise':
            sentences = [haystack] * num_noises
            for chain in chains:
                positions = list(sorted(random.sample(range(len(sentences)), len(chain))))
                for insert_pi, j in zip(positions, range(len(chain))):
                    sentences.insert(insert_pi+j, chain[j])
            context = "\n".join(sentences)

        context = context.replace(". \n", ".\n")
        return context, vars_list, value

    # Binary search for optimal noise count
    def test_length(num_noises):
        context, vars_list, value = generate_input_output_inner(num_noises)
        
        # Add ICL prompt if provided
        if icl_prompt:
            context = f"{icl_prompt}\n\n{context}"

        test_input = template.format(
            context=context,
            query=value,
            num_v=len(vars_list[0]) if vars_list else 0
        )
        return len(tokenizer.encode(test_input)), context, vars_list, value

    # Binary search implementation
    if type_haystack == 'noise':
        lower_bound = num_hops + 1
    else:
        lower_bound = max(num_chains, 1)
    upper_bound = max_seq_length // 10  # Conservative upper bound
    optimal_context = ""
    optimal_vars = []
    optimal_value = ""
    
    while lower_bound <= upper_bound:
        mid = (lower_bound + upper_bound) // 2
        tokens, context, vars_list, value = test_length(mid)
        
        if tokens <= max_seq_length:
            optimal_context = context
            optimal_vars = vars_list
            optimal_value = value
            lower_bound = mid + 1
        else:
            upper_bound = mid - 1
    
    # If no optimal found, use minimum
    if not optimal_context:
        _, optimal_context, optimal_vars, optimal_value = test_length(lower_bound)
    
    # Format final input
    final_input = template.format(
        context=optimal_context,
        query=optimal_value,
        num_v=len(optimal_vars[0]) if optimal_vars else 0
    )
    
    return final_input, optimal_context, optimal_vars, optimal_value