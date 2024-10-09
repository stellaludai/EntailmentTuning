#!/bin/bash
model_file="/path/to/your/biencoder_model.pt(downloaded from our huggingface repo)"
output_folder="generated_embeddings/dpr_nq_singe_base_entail_tuned"

if [ ! -d $output_folder ]; then
    mkdir -p $output_folder
fi

total_shards=8

# 指定GPU编号数组
gpus=(0 1 2 3 4 5 6 7)

set -e

for i in {0..7}; do
    # 分配每个shard到一个GPU
    CUDA_VISIBLE_DEVICES=${gpus[$i]} python generate_dense_embeddings.py \
        model_file=${model_file}\
        ctx_src=dpr_wiki \
        shard_id=$i num_shards=$total_shards \
        batch_size=1024 \
        out_file=$output_folder/wikipedia_passages &
done

wait
