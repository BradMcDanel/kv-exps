# eval/ruler/data/niah.py
import random
import uuid
import os
import json
import re
import wonderwords
import numpy as np
from wonderwords import RandomWord

# Load words exactly like original RULER - NO FALLBACKS
nouns = wonderwords.random_word._get_words_from_text_file("nounlist.txt")
adjs = wonderwords.random_word._get_words_from_text_file("adjectivelist.txt")
words = [f"{adj}-{noun}" for adj in adjs for noun in nouns]
words = sorted(list(set(words)))

# Load Paul Graham essays for haystack - NO FALLBACKS
essay_path = os.path.join(os.path.dirname(__file__), "../RULER/scripts/data/synthetic/json/PaulGrahamEssays.json")
with open(essay_path, 'r') as f:
    essay_data = json.load(f)
    haystack_words = re.sub(r'\s+', " ", essay_data['text']).split(" ")

# Define needle format exactly like original RULER
needle_format = "One of the special magic {type_needle_v} for {key} is: {value}."

def generate_random_number(num_digits=7):
    """Exact copy from original RULER"""
    lower_bound = 10**(num_digits - 1)
    upper_bound = 10**num_digits - 1
    return str(random.randint(lower_bound, upper_bound))

def generate_random_word():
    """Exact copy from original RULER"""
    return random.choice(words)

def generate_random_uuid():
    """Exact copy from original RULER"""
    return str(uuid.UUID(int=random.getrandbits(128), version=4))

def generate_random(type_needle):
    """Exact copy from original RULER"""
    if type_needle == 'numbers':
        return generate_random_number()
    elif type_needle == 'words':
        return generate_random_word()
    elif type_needle == 'uuids':
        return generate_random_uuid()
    else:
        raise NotImplementedError(f'{type_needle} is not implemented.')

def create_instance(tokenizer, max_length, task_args, prompt_template, instance_index=None):
    """
    Generates a single instance for the NIAH task - matches original RULER exactly.
    """
    # Get task parameters exactly like original RULER
    num_needle_k = task_args.get("num_needle_k", 1)
    num_needle_v = task_args.get("num_needle_v", 1) 
    num_needle_q = task_args.get("num_needle_q", 1)
    type_needle_k = task_args.get("type_needle_k", "words")
    type_needle_v = task_args.get("type_needle_v", "numbers")
    type_haystack = task_args.get("type_haystack", "essay")
    
    # Ensure consistency like original RULER
    num_needle_k = max(num_needle_k, num_needle_q)
    
    # Use original RULER's generate_input_output logic
    tokens_to_generate = 128  # From original RULER constants
    max_seq_length = max_length - tokens_to_generate
    
    # Binary search for optimal haystack size like original RULER
    input_text, keys, values, context = generate_input_output(
        tokenizer, max_seq_length, num_needle_k, num_needle_v, num_needle_q,
        type_needle_k, type_needle_v, type_haystack, prompt_template
    )
    
    # Use first key for query (original RULER behavior for single needle)
    query_key = keys[0] if keys else ""
    query_values = values[0] if values else []
    
    return {
        "context": context,
        "query": query_key,
        "type_needle_v": type_needle_v,
        "outputs": query_values,
        "metadata": {
            "key": query_key,
            "values": query_values,
            "type_needle_v": type_needle_v,
            "num_needle_k": num_needle_k,
            "num_needle_v": num_needle_v
        }
    }

def generate_input_output(tokenizer, max_seq_length, num_needle_k, num_needle_v, num_needle_q, 
                         type_needle_k, type_needle_v, type_haystack, template):
    """Exact copy of original RULER's generate_input_output function"""
    from nltk.tokenize import sent_tokenize
    
    # DEPTHS from original RULER
    DEPTHS = list(np.round(np.linspace(0, 100, num=40, endpoint=True)).astype(int))
    
    def generate_input_output_inner(num_haystack):
        keys, values, needles = [], [], []
        for _ in range(num_needle_k):
            keys.append(generate_random(type_needle_k))
            value = []
            for _ in range(num_needle_v):
                value.append(generate_random(type_needle_v))
                needles.append(needle_format.format(
                    type_needle_v=type_needle_v,
                    key=keys[-1],
                    value=value[-1],
                ))
            values.append(value)

        random.shuffle(needles)

        # Context generation - exact copy from original RULER
        if type_haystack == 'essay':
            if num_haystack <= len(haystack_words):
                text = " ".join(haystack_words[:num_haystack])
            else:
                # Repeat haystack as many times as needed and slice to num_haystack
                repeats = (num_haystack + len(haystack_words) - 1) // len(haystack_words)  # Ceiling division
                text = " ".join((haystack_words * repeats)[:num_haystack])
            
            document_sents = sent_tokenize(text.strip())
            insertion_positions = [0] + \
                                  sorted([int(len(document_sents) * (depth / 100)) for depth in random.sample(DEPTHS, len(needles))]) + \
                                  [len(document_sents)]
            document_sents_list = []
            for i in range(1, len(insertion_positions)):
                last_pos = insertion_positions[i-1]
                next_pos = insertion_positions[i]
                document_sents_list.append(" ".join(document_sents[last_pos:next_pos]))
                if i-1 < len(needles):
                    document_sents_list.append(needles[i-1])
            context = " ".join(document_sents_list)

        elif type_haystack == 'noise':
            sentences = ["The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."] * num_haystack
            indexes = sorted(random.sample(range(num_haystack), len(needles)), reverse=True)
            for i, needle in zip(indexes, needles):
                sentences.insert(i, needle)
            context = " ".join(sentences)
            
        elif type_haystack == 'needle':
            sentences = [needle_format.format(
                type_needle_v=type_needle_v,
                key=generate_random(type_needle_k),
                value=generate_random(type_needle_v),
            ) for _ in range(num_haystack)]
            indexes = sorted(random.sample(range(num_haystack), len(needles)), reverse=True)
            for i, needle in zip(indexes, needles):
                sentences.insert(i, needle)
            context = " ".join(sentences)
        
        return context, keys, values

    # Binary search for optimal haystack size
    def test_length(num_haystack):
        context, keys, values = generate_input_output_inner(num_haystack)
        query_key = keys[0] if keys else ""
        test_input = template.format(
            context=context,
            query=query_key,
            type_needle_v=type_needle_v
        )
        return len(tokenizer.encode(test_input)), context, keys, values

    # Binary search implementation
    lower_bound = max(num_needle_k * num_needle_v, 1)
    upper_bound = max_seq_length // 10  # Conservative upper bound
    optimal_context = ""
    optimal_keys = []
    optimal_values = []
    
    while lower_bound <= upper_bound:
        mid = (lower_bound + upper_bound) // 2
        tokens, context, keys, values = test_length(mid)
        
        if tokens <= max_seq_length:
            optimal_context = context
            optimal_keys = keys
            optimal_values = values
            lower_bound = mid + 1
        else:
            upper_bound = mid - 1
    
    # If no optimal found, use minimum
    if not optimal_context:
        _, optimal_context, optimal_keys, optimal_values = test_length(lower_bound)
    
    # Format final input
    query_key = optimal_keys[0] if optimal_keys else ""
    final_input = template.format(
        context=optimal_context,
        query=query_key,
        type_needle_v=type_needle_v
    )
    
    return final_input, optimal_keys, optimal_values, optimal_context