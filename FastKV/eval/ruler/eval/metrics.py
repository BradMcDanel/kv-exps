# eval/ruler/eval/metrics.py
import re

def string_match_all(preds, refs):
    """Original RULER implementation - exact copy"""
    score = sum([sum([1.0 if r.lower() in pred.lower() else 0.0 for r in ref]) / len(ref) for pred, ref in zip(preds, refs)]) / len(preds) * 100
    return round(score, 2)

def string_match_part(preds, refs):
    """Original RULER implementation - exact copy"""
    score = sum([max([1.0 if r.lower() in pred.lower() else 0.0 for r in ref]) for pred, ref in zip(preds, refs)]) / len(preds) * 100
    return round(score, 2)

# Task-specific metrics matching original RULER
TASK_METRICS = {
    'niah': {
        'metric_fn': string_match_all,
    },
    'variable_tracking': {
        'metric_fn': string_match_all,
    },
    'common_words_extraction': {
        'metric_fn': string_match_all,
    },
    'freq_words_extraction': {
        'metric_fn': string_match_all
    },
    'qa': {
        'metric_fn': string_match_part,
    },
}
