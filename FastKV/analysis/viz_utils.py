# analysis/viz_utils.py

import matplotlib.pyplot as plt
import seaborn as sns

FIXED_EXIT_LAYER = 15

# ==============================================================================
# PLOTTING STYLES AND COLORS
# ==============================================================================

# Unified color scheme for all plots.
METHOD_COLORS = {
    'Oracle': '#1565c0',
    'FastKV': '#d62728',
    'GemFilter': '#ff7f0e',
    'CLAA': '#2ca02c',
    'FullKV': '#000000',
    'SpecPrefill': '#5e2b8c',
    'Speculative': '#5e2b8c',
    'Speculative (k=1)': '#d6b3ff',
    'Speculative (k=8)': '#a066cc',
    'Speculative (k=32)': '#5e2b8c',
    'Adaptive Exit': '#1f77b4',
    'Fixed Exit': '#d62728',
}

def set_publication_style():
    """Sets a consistent, high-quality plotting style with larger fonts."""
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 20,
        'axes.labelsize': 28,
        'axes.titlesize': 28,
        'axes.titlecolor': 'black',
        'axes.titlepad': 12,
        'xtick.labelsize': 24,
        'ytick.labelsize': 24,
        'legend.fontsize': 22,
        'legend.title_fontsize': 24,
        'figure.titlesize': 34,
        'grid.linestyle': ':',
        'grid.linewidth': 0.8,
        'lines.linewidth': 3.5,
        'lines.markersize': 9,
    })

# ==============================================================================
# DATASET AND TASK GROUPINGS
# ==============================================================================

# Defines which datasets belong to which high-level task category.
TASKS_AND_DATASETS = {
    'Single-Doc QA': ['narrativeqa', 'qasper', 'multifieldqa_en'],
    'Multi-Doc QA': ['hotpotqa', '2wikimqa', 'musique'],
    'Summarization': ['gov_report', 'qmsum', 'multi_news'],
    'Few-shot Learning': ['trec', 'triviaqa', 'samsum'],
    'Synthetic Task': ['passage_count', 'passage_retrieval_en'],
    'Code Completion': ['lcc', 'repobench-p'],
}

DATASET_TO_TASK_MAP = {
    dataset: task
    for task, datasets in TASKS_AND_DATASETS.items()
    for dataset in datasets
}

# A flattened list of all dataset IDs to be plotted.
ALL_DATASETS_TO_PLOT = [
    dataset for datasets in TASKS_AND_DATASETS.values() for dataset in datasets
]

# Maps full task names to shorter versions for display on individual plots.
TASK_SHORT_NAMES = {
    'Single-Doc QA': 'QA',
    'Multi-Doc QA': 'QA',
    'Summarization': 'Summ',
    'Few-shot Learning': 'Few-shot',
    'Synthetic Task': 'Synth',
    'Code Completion': 'Code',
}

# Maps dataset IDs to their proper, publication-ready display names.
DATASET_NAME_MAP = {
    'narrativeqa': 'NarrativeQA',
    'qasper': 'QASPER',
    'multifieldqa_en': 'MultiFieldQA',
    'hotpotqa': 'HotpotQA',
    '2wikimqa': '2WikiMQA',
    'musique': 'Musique',
    'gov_report': 'GovReport',
    'qmsum': 'QMSum',
    'multi_news': 'Multi-News',
    'trec': 'TREC',
    'triviaqa': 'TriviaQA',
    'samsum': 'SAMSum',
    'passage_count': 'PassageCount',
    'passage_retrieval_en': 'PassageRetrieval',
    'lcc': 'LCC',
    'repobench-p': 'RepoBench-P',
}
