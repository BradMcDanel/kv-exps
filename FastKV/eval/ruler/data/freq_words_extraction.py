import json
import random
import string
import numpy as np
from scipy.special import zeta

def create_instance(tokenizer, max_length, task_args, prompt_template, instance_index=None):
    """
    Create a single frequency words extraction instance - matches original RULER exactly.
    """
    alpha = task_args.get('alpha', 2.0)
    coded_wordlen = 6
    tokens_to_generate = 50  # From original RULER constants
    
    # Calculate vocab size based on sequence length like original RULER
    max_seq_length = max_length - tokens_to_generate
    vocab_size = max_seq_length // 50
    
    # Use original RULER's generate_input_output function logic
    input_text, answer, num_words = generate_input_output(
        tokenizer, max_seq_length, coded_wordlen, vocab_size, alpha, prompt_template
    )
    
    # Extract context from the formatted input (remove template parts)
    # Find where context starts and ends in the template
    template_start = prompt_template.find('{context}')
    if template_start == -1:
        context = input_text
    else:
        template_before = prompt_template[:template_start]
        template_after = prompt_template[template_start + 9:]  # len('{context}') = 9
        
        # Remove template parts to get just context
        context = input_text
        if template_before:
            context = context[len(template_before):] if context.startswith(template_before) else context
        if template_after:
            context = context[:-len(template_after)] if context.endswith(template_after) else context
        context = context.strip()
    
    return {
        "context": context,
        "outputs": answer,
        "metadata": {
            "alpha": alpha,
            "vocab_size": vocab_size,
            "coded_wordlen": coded_wordlen,
            "total_words": num_words
        }
    }

def generate_input_output(tokenizer, max_len, coded_wordlen=6, vocab_size=2000, alpha=2.0, template=""):
    """Exact copy of generate_input_output from original RULER"""
    # generate vocab
    vocab = [''.join(random.choices(string.ascii_lowercase, k=coded_wordlen)) for _ in range(vocab_size)]
    while len(set(vocab)) < vocab_size:
        vocab.append(''.join(random.choices(string.ascii_lowercase, k=coded_wordlen)))
    vocab = sorted(list(set(vocab)))
    random.shuffle(vocab)
    vocab[0] = '...' # treat the top ranked as noise

    # sample words
    def gen_text(num_words):
        k = np.arange(1, len(vocab)+1)
        sampled_cnt = num_words*(k**-alpha)/zeta(alpha)
        sampled_words = [[w] * zi for w, zi in zip(vocab, sampled_cnt.astype(int))]
        sampled_words = [x for wlst in sampled_words for x in wlst]
        random.shuffle(sampled_words)
        return template.format(context=' '.join(sampled_words), query=''), vocab[1:4]

    # Binary search logic from original RULER
    num_words = max_len // coded_wordlen # init
    text, answer = gen_text(num_words)
    while len(tokenizer.encode(text)) < max_len:
        num_words += 10  # incremental from original
        text, answer = gen_text(num_words)
    num_words -= 10
    text, answer = gen_text(num_words)
    return text, answer, num_words