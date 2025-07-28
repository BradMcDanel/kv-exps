#!/bin/bash

# RULER benchmark script for CLAA evaluation
# Based on the structure of run_ruler.sh

# All RULER tasks
task_list="niah_single_1 niah_single_2 niah_single_3 niah_multikey_1 niah_multikey_2 niah_multikey_3 niah_multivalue niah_multiquery vt cwe fwe qa_1 qa_2"

# Model configuration
model="meta-llama/Llama-3.1-8B-Instruct"
device=0
seq_length=4096
num_samples=10

# CLAA specific parameters
window_size=8
max_capacity_prompt_percentage=0.125
last_n_layers=4  # Number of last layers for CLAA aggregation

echo "Starting RULER evaluation with CLAA:"
echo "Model: $model"
echo "Sequence Length: $seq_length"
echo "Tasks: $task_list"
echo "Samples per task: $num_samples"
echo "Window Size: $window_size"
echo "Max Capacity Prompt Percentage: $max_capacity_prompt_percentage"
echo "Last N Layers: $last_n_layers"
echo "=================================="

# CLAA evaluation
path="claa-$seq_length"
echo "Running CLAA evaluation..."
for task in $task_list
do
    echo "Processing task: $task"
    CUDA_VISIBLE_DEVICES=$device python -m eval.ruler.main \
        --model $model \
        --mode claa \
        --task $task \
        --save_path $path \
        --num_samples $num_samples \
        --seq_length $seq_length \
        --window_size $window_size \
        --max_capacity_prompt_percentage $max_capacity_prompt_percentage \
        --last_n_layers $last_n_layers
    
    echo "Evaluating task: $task"
    CUDA_VISIBLE_DEVICES=$device python -m eval.ruler.eval.evaluate \
        --pred_dir "outputs/$model/ruler/$path/$seq_length/pred/"
done

echo "RULER CLAA evaluation completed!"
echo "Results saved in: outputs/$model/ruler/$path/"