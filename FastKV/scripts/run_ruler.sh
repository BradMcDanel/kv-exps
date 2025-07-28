#!/bin/bash

# RULER benchmark script for FastKV evaluation
# Based on the structure of run_longbench.sh

# All RULER tasks
task_list="niah_single_1 niah_single_2 niah_single_3 niah_multikey_1 niah_multikey_2 niah_multikey_3 niah_multivalue niah_multiquery vt cwe fwe qa_1 qa_2"

# Model configuration
model="meta-llama/Llama-3.1-8B-Instruct"
device=0
seq_length=4096
num_samples=10

echo "Starting RULER evaluation with:"
echo "Model: $model"
echo "Sequence Length: $seq_length"
echo "Tasks: $task_list"
echo "Samples per task: $num_samples"
echo "=================================="

# FullKV baseline
path="fullkv-$seq_length"
echo "Running FullKV baseline..."
for task in $task_list
do
    echo "Processing task: $task"
    CUDA_VISIBLE_DEVICES=$device python -m eval.ruler.main \
        --model $model \
        --mode fullkv \
        --task $task \
        --save_path $path \
        --num_samples $num_samples \
        --seq_length $seq_length
    
    echo "Evaluating task: $task"
    CUDA_VISIBLE_DEVICES=$device python -m eval.ruler.eval.evaluate \
        --pred_dir "outputs/$model/ruler/$path/$seq_length/pred/"
done

echo "RULER evaluation completed!"
echo "Results saved in: outputs/$model/ruler/$path/"
