import json
import random
import os
import requests
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# This is a direct copy of the DOCUMENT_PROMPT from the original RULER script
DOCUMENT_PROMPT = "Document {i}:\n{document}"

def download_qa_datasets():
    """Download SQuAD and HotpotQA datasets if they don't exist."""
    data_dir = Path(__file__).parent / "json"
    data_dir.mkdir(exist_ok=True)
    
    squad_path = data_dir / "squad.json"
    hotpot_path = data_dir / "hotpotqa.json"
    
    if not squad_path.exists():
        logger.info("Downloading SQuAD dataset...")
        try:
            response = requests.get("https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json")
            response.raise_for_status()
            with open(squad_path, 'w') as f:
                f.write(response.text)
            logger.info("SQuAD dataset downloaded successfully")
        except Exception as e:
            logger.error(f"Failed to download SQuAD: {e}")
            return False, False
    
    if not hotpot_path.exists():
        logger.info("Downloading HotpotQA dataset...")
        try:
            response = requests.get("http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json")
            response.raise_for_status()
            with open(hotpot_path, 'w') as f:
                f.write(response.text)
            logger.info("HotpotQA dataset downloaded successfully")
        except Exception as e:
            logger.error(f"Failed to download HotpotQA: {e}")
            return True, False
    
    return squad_path.exists(), hotpot_path.exists()

def read_squad(file_path):
    """Read SQuAD dataset - exactly matching original RULER logic."""
    with open(file_path) as f:
        data = json.load(f)

    total_docs = [p['context'] for d in data['data'] for p in d['paragraphs']]
    total_docs = sorted(list(set(total_docs)))
    total_docs_dict = {c: idx for idx, c in enumerate(total_docs)}

    total_qas = []
    for d in data['data']:
        more_docs = [total_docs_dict[p['context']] for p in d['paragraphs']]
        for p in d['paragraphs']:
            for qas in p['qas']:
                if not qas['is_impossible']:
                    total_qas.append({
                        'query': qas['question'],
                        'outputs': [a['text'] for a in qas['answers']],
                        'context': [total_docs_dict[p['context']]],
                        'more_context': [idx for idx in more_docs if idx != total_docs_dict[p['context']]]
                    })
    
    return total_qas, total_docs

def read_hotpotqa(file_path):
    """Read HotpotQA dataset - exactly matching original RULER logic."""
    with open(file_path) as f:
        data = json.load(f)

    total_docs = [f"{t}\n{''.join(p)}" for d in data for t, p in d['context']]
    total_docs = sorted(list(set(total_docs)))
    total_docs_dict = {c: idx for idx, c in enumerate(total_docs)}

    total_qas = []
    for d in data:
        total_qas.append({
            'query': d['question'],
            'outputs': [d['answer']],
            'context': [total_docs_dict[f"{t}\n{''.join(p)}"] for t, p in d['context']],
        })

    return total_qas, total_docs

def create_instance(tokenizer, max_length, task_args, prompt_template, instance_index):
    """
    Create a single QA instance, replicating the logic from the original RULER script
    to ensure consistency. This includes deterministic QA selection and a binary
    search to maximize context length.
    """
    dataset = task_args.get('dataset', 'squad')
    
    # Ensure datasets are downloaded and load them
    squad_exists, hotpot_exists = download_qa_datasets()
    data_dir = Path(__file__).parent / "json"
    
    if dataset == 'squad':
        if not squad_exists:
            raise FileNotFoundError("SQuAD dataset not available. Please check download.")
        QAS, DOCS = read_squad(data_dir / "squad.json")
    elif dataset == 'hotpotqa':
        if not hotpot_exists:
            raise FileNotFoundError("HotpotQA dataset not available. Please check download.")
        QAS, DOCS = read_hotpotqa(data_dir / "hotpotqa.json")
    else:
        raise NotImplementedError(f'{dataset} is not implemented.')

    # This helper function is a direct adaptation of the original RULER's generate_input_output
    def _generate_and_format(qa_index, num_docs):
        curr_q = QAS[qa_index]['query']
        curr_a = QAS[qa_index]['outputs']
        curr_docs_indices = QAS[qa_index]['context']
        curr_more_indices = QAS[qa_index].get('more_context', [])
        
        # Build document list following original RULER logic
        if num_docs < len(DOCS):
            # Calculate how many distractor docs are needed
            num_distractors = num_docs - len(curr_docs_indices)
            
            # Take distractors from 'more_context' first, then from the general pool
            if num_distractors > len(curr_more_indices):
                distractors_from_more = curr_more_indices
                
                # Avoid using docs that are already in context or more_context
                avoid_indices = set(curr_docs_indices + curr_more_indices)
                remaining_distractor_pool = [i for i, d in enumerate(DOCS) if i not in avoid_indices]
                
                num_remaining_distractors = num_distractors - len(distractors_from_more)
                distractors_from_pool = random.sample(remaining_distractor_pool, min(num_remaining_distractors, len(remaining_distractor_pool)))
                
                distractor_indices = distractors_from_more + distractors_from_pool
            else:
                distractor_indices = random.sample(curr_more_indices, max(0, num_distractors))

            all_doc_indices = curr_docs_indices + distractor_indices
            all_docs_text = [DOCS[idx] for idx in all_doc_indices]
        else:
            # Repeat DOCS as many times as needed and slice to num_docs
            repeats = (num_docs + len(DOCS) - 1) // len(DOCS)
            all_docs_text = (DOCS * repeats)[:num_docs]

        random.shuffle(all_docs_text)
        
        context = '\n\n'.join([DOCUMENT_PROMPT.format(i=i+1, document=d) for i, d in enumerate(all_docs_text)])
        
        # Return components for the main script to format
        return context, curr_q, curr_a

    # --- Binary Search for Optimal Document Count (from original RULER) ---
    tokens_to_generate = 32  # From original RULER constants for qa tasks
    max_seq_length = max_length - tokens_to_generate
    incremental = 10 # A value used for estimation in the original script

    # Deterministic QA pair selection
    qa_index = instance_index % len(QAS)

    # Estimate tokens per doc for binary search bounds
    # We generate a sample prompt to estimate token usage
    _test_context, _test_q, _ = _generate_and_format(qa_index, incremental)
    _test_prompt = prompt_template.format(context=_test_context, query=_test_q)
    sample_tokens = len(tokenizer.encode(_test_prompt))
    tokens_per_doc = sample_tokens / incremental if incremental > 0 else 200 # Fallback

    # Set bounds for binary search
    estimated_max_docs = int((max_seq_length / tokens_per_doc) * 3) if tokens_per_doc > 0 else 200
    lower_bound = 1
    upper_bound = max(estimated_max_docs, lower_bound * 2)
    
    optimal_num_docs = 1

    while lower_bound <= upper_bound:
        mid = (lower_bound + upper_bound) // 2
        if mid == 0: break

        _test_context, _test_q, _ = _generate_and_format(qa_index, mid)
        _test_prompt = prompt_template.format(context=_test_context, query=_test_q)
        total_tokens = len(tokenizer.encode(_test_prompt))

        if total_tokens <= max_seq_length:
            optimal_num_docs = mid
            lower_bound = mid + 1
        else:
            upper_bound = mid - 1
            
    # Generate final instance with the optimal number of documents
    final_context, final_query, final_outputs = _generate_and_format(qa_index, optimal_num_docs)

    return {
        "context": final_context,
        "query": final_query,
        "outputs": final_outputs,
        "metadata": {
            "dataset": dataset,
            "qa_index": qa_index,
            "num_docs": optimal_num_docs,
        }
    }