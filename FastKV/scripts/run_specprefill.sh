#!/bin/bash

model="meta-llama/Llama-3.1-8B-Instruct"
dataset_list="qasper"
speculator="meta-llama/Llama-3.2-1B-Instruct" 
device=0
max_prompt=512
path="speculative_prefill"

for dataset in $dataset_list
do
    CUDA_VISIBLE_DEVICES=$device python -m eval.longbench.main \
    --model $model \
    --speculator_model_name $speculator \
    --mode speculative_prefill \
    --save_path $path \
    --dataset $dataset \
    --max_capacity_prompt $max_prompt
done
