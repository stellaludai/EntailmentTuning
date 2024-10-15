#!/bin/bash
train_data_path=/path/to/your/nq-train.json
dev_data_path=/path/to/your/nq-dev.json
python -m torch.distributed.launch --nproc_per_node=8 --master_port 28006 \
    train_dense_encoder.py \
    train_datasets=[$train_data_path] \
    dev_datasets=[$dev_data_path] \
    train=biencoder_nq \
    output_dir=/data/dailu/dpr/checkpoints \
    train.batch_size=32