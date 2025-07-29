#!/bin/bash

# RULER benchmark script for Speculative Prefill evaluation
# Based on the structure of run_ruler.sh

# All RULER tasks
task_list="niah_single_1 niah_single_2 niah_single_3 niah_multikey_1 niah_multikey_2 niah_multikey_3 niah_multivalue niah_multiquery vt cwe fwe qa_1 qa_2"

# Model configuration
model="meta-llama/Llama-3.1-8B-Instruct"
device=0
seq_length=4096
num_samples=10

# Speculative Prefill specific parameters
speculator_model_name="meta-llama/Llama-3.2-1B-Instruct"
max_capacity_prompt_percentage=0.4
look_ahead_k=8
kernel_size=7

echo "Starting RULER evaluation with Speculative Prefill:"
echo "Model: $model"
echo "Speculator Model: $speculator_model_name"
echo "Sequence Length: $seq_length"
echo "Tasks: $task_list"
echo "Samples per task: $num_samples"
echo "Max Capacity Prompt Percentage: $max_capacity_prompt_percentage"
echo "Look Ahead K: $look_ahead_k"
echo "Kernel Size: $kernel_size"
echo "=================================="

# Speculative Prefill evaluation
path="spec_prefill-$seq_length"
echo "Running Speculative Prefill evaluation..."
for task in $task_list
do
    echo "Processing task: $task"
    CUDA_VISIBLE_DEVICES=$device python -m eval.ruler.main \
        --model $model \
        --mode speculative_prefill \
        --task $task \
        --save_path $path \
        --num_samples $num_samples \
        --seq_length $seq_length \
        --speculator_model_name $speculator_model_name \
        --max_capacity_prompt_percentage $max_capacity_prompt_percentage \
        --look_ahead_k $look_ahead_k \
        --kernel_size $kernel_size 
    
    echo "Evaluating task: $task"
    CUDA_VISIBLE_DEVICES=$device python -m eval.ruler.eval.evaluate \
        --pred_dir "outputs/$model/ruler/$path/$seq_length/pred/"
done

echo "RULER Speculative Prefill evaluation completed!"
echo "Results saved in: outputs/$model/ruler/$path/"
