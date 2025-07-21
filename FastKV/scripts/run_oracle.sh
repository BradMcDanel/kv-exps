#!/bin/bash

model="meta-llama/Llama-3.1-8B-Instruct"
dataset_list="qasper"
device=0
keep_percentage=0.1
oracle_rankings_path="analysis_results/oracles"
path="oracle"
tsp_idx=15

for dataset in $dataset_list
do
    CUDA_VISIBLE_DEVICES=$device python -m eval.longbench.main \
        --model $model \
        --mode oracle \
        --save_path $path \
        --dataset $dataset \
        --keep_percentage $keep_percentage \
        --oracle_rankings_path $oracle_rankings_path \
        --tsp_idx $tsp_idx

    CUDA_VISIBLE_DEVICES=$device python -m eval.longbench.evaluate \
        --model $model \
        --eval_path $path
done
