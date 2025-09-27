#!/bin/bash
# run_models.sh

model_paths=(
    "Qwen/Qwen2.5-VL-7B-Instruct"
)

file_names=(
    "FileName"
)

output_path=("src/r1-v/Video-Ours-data/xxx.json")
dataset_path=("src/r1-v/Video-Ours-data/xxx.json")

export DECORD_EOF_RETRY_MAX=20480


for i in "${!model_paths[@]}"; do
    model="${model_paths[$i]}"
    file_name="${file_names[$i]}"
    CUDA_VISIBLE_DEVICES=0 python ./src/eval_bench.py --model_path "$model" --file_name "$file_name" --output_path "$output_path" --dataset_path "$dataset_path"
done
