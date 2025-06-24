#!/bin/bash

model="meta-llama/Llama-3.1-8B-Instruct"
dataset_list="qasper"
device=0
max_prompt=512
tsp_schedule="3:8192,7:4096,11:3072,15:2048,19:1536,23:1024"

# --- Execution ---
# Define a unique path for the results to avoid overwriting other experiments
path="hfastkv-3-$max_prompt"
for dataset in $dataset_list
do

    CUDA_VISIBLE_DEVICES=$device python -m eval.longbench.main \
        --model "$model" \
        --mode hfastkv \
        --save_path "$path" \
        --dataset "$dataset" \
        --max_capacity_prompt "$max_prompt" \
        --tsp_schedule "$tsp_schedule"

    CUDA_VISIBLE_DEVICES=$device python -m eval.longbench.evaluate \
        --model "$model" \
        --eval_path "$path"
done

echo "--- HFastKV Evaluation Complete ---"
