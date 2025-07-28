import json
import random
import os
import wonderwords

def create_instance(tokenizer, max_length, task_args, prompt_template, instance_index=None):
    """
    Create a single common words extraction instance - matches original RULER exactly.
    """
    freq_cw = task_args.get('freq_cw', 30)
    freq_ucw = task_args.get('freq_ucw', 3)
    num_cw = task_args.get('num_cw', 10)
    
    # Load wonderwords exactly like original RULER - NO FALLBACKS
    nouns = wonderwords.random_word._get_words_from_text_file("nounlist.txt")
    adjs = wonderwords.random_word._get_words_from_text_file("adjectivelist.txt")
    verbs = wonderwords.random_word._get_words_from_text_file("verblist.txt")
    words = nouns + adjs + verbs
    words = sorted(list(set(words)))
    random.shuffle(words)
    
    # Load Randleword english words from original RULER - NO FALLBACKS
    english_words_path = os.path.join(os.path.dirname(__file__), "../RULER/scripts/data/synthetic/json/english_words.json")
    with open(english_words_path, 'r') as f:
        word_dict = json.load(f)
        randle_words = list(word_dict.values())
    
    # Binary search logic from original RULER to find optimal word count
    tokens_to_generate = 120  # From original RULER constants
    max_seq_length = max_length - tokens_to_generate
    
    # Estimate tokens per question to determine reasonable upper bound
    sample_context, _ = get_example(words, randle_words, 4096, freq_cw, freq_ucw, num_cw)
    sample_tokens = len(tokenizer.encode(sample_context))
    tokens_per_words = sample_tokens / 4096
    
    # Binary search for optimal haystack size (from original RULER)
    estimated_max_words = int(max_seq_length // tokens_per_words) * 2
    lower_bound = 10
    upper_bound = max(estimated_max_words, 20)
    
    optimal_num_words = None
    
    while lower_bound <= upper_bound:
        mid = (lower_bound + upper_bound) // 2
        test_context, test_answer = get_example(words, randle_words, mid, freq_cw, freq_ucw, num_cw)
        test_prompt = prompt_template.format(context=test_context, num_cw=num_cw)
        total_tokens = len(tokenizer.encode(test_prompt)) + tokens_to_generate
        
        if total_tokens <= max_length:
            optimal_num_words = mid
            lower_bound = mid + 1
        else:
            upper_bound = mid - 1
    
    num_words = optimal_num_words if optimal_num_words is not None else 10
    
    # Generate final instance
    context, outputs = get_example(words, randle_words, num_words, freq_cw, freq_ucw, num_cw)
    
    return {
        "context": context,
        "num_cw": num_cw,  # Add this for template compatibility
        "outputs": outputs,
        "metadata": {
            "freq_cw": freq_cw,
            "freq_ucw": freq_ucw,
            "num_cw": num_cw,
            "total_words": len(context.split())
        }
    }

def get_example(words, randle_words, num_words, common_repeats=30, uncommon_repeats=3, common_nums=10):
    """Exact copy of get_example from original RULER"""
    if num_words <= len(words):
        word_list_full = random.sample(words, num_words)
    else:
        word_list_full = random.sample(randle_words, num_words)

    common, uncommon = word_list_full[:common_nums], word_list_full[common_nums:]
    word_list = common * int(common_repeats) + uncommon * int(uncommon_repeats)
    random.shuffle(word_list)

    # Formatting the word list as "1. word1 2. word2 3. word3 ..."
    context = ' '.join([f"{i + 1}. {word}" for i, word in enumerate(word_list)])

    return context, common
