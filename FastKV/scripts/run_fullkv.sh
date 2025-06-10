#!/bin/bash

model="meta-llama/Llama-3.1-8B-Instruct"
dataset_list="qasper"
device=0
path="fullkv"

# Check if with-proxy alias exists
cmd_prefix=""
if alias with-proxy &>/dev/null; then
    cmd_prefix="with-proxy "
    echo "Detected with-proxy alias - using it automatically"
fi

echo "Running with configuration:"
echo "  Model: $model"
echo "  Dataset(s): $dataset_list"
echo "  Device: $device"
echo "  Path: $path"
echo "  Python: $(which python)"
echo

for dataset in $dataset_list
do
    echo "Processing dataset: $dataset"
    
    # Run main evaluation
    ${cmd_prefix}CUDA_VISIBLE_DEVICES=$device python -m eval.longbench.main \
        --model $model \
        --mode fullkv \
        --save_path $path \
        --dataset $dataset
    
    # Run evaluation
    ${cmd_prefix}CUDA_VISIBLE_DEVICES=$device python -m eval.longbench.evaluate \
        --model $model \
        --eval_path $path
done
