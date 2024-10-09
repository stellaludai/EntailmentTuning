#!/bin/bash

# first run entailment tuning
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.run --nproc_per_node 4 --master_port 28866 entail_tuning.py

wait

# second run dpr

CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port 28006 \
    train_dense_encoder.py \
    train_datasets=[/data/dailu/dpr/downloads/data/retriever/nq-train.json] \
    dev_datasets=[/data/dailu/dpr/downloads/data/retriever/nq-dev.json] \
    train=biencoder_nq \
    output_dir=/data/dailu/dpr/checkpoints \
    train.batch_size=32