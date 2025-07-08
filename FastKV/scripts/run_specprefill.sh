#!/bin/bash

model="meta-llama/Llama-3.1-8B-Instruct"
dataset_list="qasper"
speculator="meta-llama/Llama-3.2-1B-Instruct" 
device=0
max_prompt_pct=0.1
look_ahead_k=8
kernel_size=7
path="speculative_prefill"

for dataset in $dataset_list
do
    CUDA_VISIBLE_DEVICES=$device python -m eval.longbench.main \
    --model $model \
    --mode speculative_prefill \
    --save_path $path \
    --dataset $dataset \
    --max_capacity_prompt_pct $max_prompt_pct \
    --speculator_model_name $speculator \
    --look_ahead_k $look_ahead_k \
    --kernel_size $kernel_size

    CUDA_VISIBLE_DEVICES=$device python -m eval.longbench.evaluate \
        --model $model \
        --eval_path $path
done
