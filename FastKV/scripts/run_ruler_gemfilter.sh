#!/bin/bash

# RULER benchmark script for GemFilter evaluation
# Based on the structure of run_ruler.sh

# All RULER tasks
task_list="niah_single_1 niah_single_2 niah_single_3 niah_multikey_1 niah_multikey_2 niah_multikey_3 niah_multivalue niah_multiquery vt cwe fwe qa_1 qa_2"

# Model configuration
model="meta-llama/Llama-3.1-8B-Instruct"
device=0
seq_length=4096
num_samples=10

# GemFilter specific parameters
filter_idx=15
topk_percentage=0.4

echo "Starting RULER evaluation with GemFilter:"
echo "Model: $model"
echo "Sequence Length: $seq_length"
echo "Tasks: $task_list"
echo "Samples per task: $num_samples"
echo "Filter Index: $filter_idx"
echo "TopK: $topk"
echo "TopK Percentage: $topk_percentage"
echo "=================================="

# GemFilter evaluation
path="gemfilter-$seq_length"
echo "Running GemFilter evaluation..."
for task in $task_list
do
    echo "Processing task: $task"
    CUDA_VISIBLE_DEVICES=$device python -m eval.ruler.main \
        --model $model \
        --mode gemfilter \
        --task $task \
        --save_path $path \
        --num_samples $num_samples \
        --seq_length $seq_length \
        --filter_idx $filter_idx \
        --topk_percentage $topk_percentage
    
    echo "Evaluating task: $task"
    CUDA_VISIBLE_DEVICES=$device python -m eval.ruler.eval.evaluate \
        --pred_dir "outputs/$model/ruler/$path/$seq_length/pred/"
done

echo "RULER GemFilter evaluation completed!"
echo "Results saved in: outputs/$model/ruler/$path/"
