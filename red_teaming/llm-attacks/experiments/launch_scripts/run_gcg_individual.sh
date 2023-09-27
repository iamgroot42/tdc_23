#!/bin/bash

export WANDB_MODE=disabled

# Optionally set the cache for transformers
# export TRANSFORMERS_CACHE='YOUR_PATH/huggingface'

for data_offset in 0 5 10 15 20 25 30 35 40
do

    knocky python -u ../main.py \
        --config="../configs/individual_llama2.py" \
        --config.attack=gcg \
        --config.train_data="../../data/advbench/harmful_behaviors.csv" \
        --config.result_prefix="../results/indi/behaviors_llama2_gcg_offset${data_offset}" \
        --config.n_train_data=5 \
        --config.data_offset=$data_offset \
        --config.n_steps=1000 \
        --config.test_steps=50 \
        --config.batch_size=256
done
