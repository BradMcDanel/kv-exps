#!/bin/bash

model="meta-llama/Llama-3.1-8B-Instruct"
dataset_list="qasper"
device=0
keep_percentage=0.1
first_k=32
last_k=64
path="uniform"

for dataset in $dataset_list
do
    CUDA_VISIBLE_DEVICES=$device python -m eval.longbench.main \
        --model $model \
        --mode uniform \
        --save_path $path \
        --dataset $dataset \
        --keep_percentage $keep_percentage \
        --uniform_first_k $first_k \
        --uniform_last_k $last_k

    CUDA_VISIBLE_DEVICES=$device python -m eval.longbench.evaluate \
        --model $model \
        --eval_path $path
done
