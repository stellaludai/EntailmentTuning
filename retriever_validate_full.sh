#!/bin/bash

model_file="/path/to/your/biencoder_model.pt(downloaded from our huggingface repo)"
model_name=$(basename ${model_file})

encoded_ctx_files=[\"generated_embeddings/dpr_nq_singe_base_entail_tuned/wikipedia_passages_*\"]

test_dataset=nq_test

CUDA_VISIBLE_DEVICES=0,1,2,3 python dense_retriever.py \
	model_file=${model_file} \
	qa_dataset=${test_dataset} \
	ctx_datatsets=[dpr_wiki] \
    encoded_ctx_files=${encoded_ctx_files} \
	out_file=inference_output/inference_output_${test_dataset}_${model_name}.json \
    n_gpu=4