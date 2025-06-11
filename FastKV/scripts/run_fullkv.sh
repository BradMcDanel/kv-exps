#!/bin/bash

model="meta-llama/Llama-3.1-8B-Instruct"
dataset_list="qasper"
device=0
path="fullkv"
for dataset in $dataset_list
do
    # Run main evaluation
    CUDA_VISIBLE_DEVICES=$device python -m eval.longbench.main \
        --model $model \
        --mode fullkv \
        --save_path $path \
        --dataset $dataset
    
    # Run evaluation
    CUDA_VISIBLE_DEVICES=$device python -m eval.longbench.evaluate \
        --model $model \
        --eval_path $path
done
