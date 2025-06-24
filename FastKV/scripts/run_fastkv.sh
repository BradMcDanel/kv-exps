#!/bin/bash

model="meta-llama/Llama-3.1-8B-Instruct"
dataset_list="qasper"
device=0
path="fastkv-512"
max_prompt=512

for dataset in $dataset_list
do
    CUDA_VISIBLE_DEVICES=$device python -m eval.longbench.main \
    --model $model \
    --mode fastkv \
    --save_path $path \
    --dataset $dataset \
    --max_capacity_prompt "$max_prompt"\ 
    --tsp_idx 15 \
    --tsp_len 2048
    CUDA_VISIBLE_DEVICES=$device python -m eval.longbench.evaluate \
    --model $model \
    --eval_path $path
done
