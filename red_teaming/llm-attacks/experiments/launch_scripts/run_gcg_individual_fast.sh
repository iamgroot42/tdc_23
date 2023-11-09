#!/bin/bash

export WANDB_MODE=disabled

# Optionally set the cache for transformers
# export TRANSFORMERS_CACHE='YOUR_PATH/huggingface'

for data_offset in $(seq 4 1 50)
do
    knocky python -u ../main.py \
        --config="../configs/individual_llama2.py" \
        --config.attack=gcg \
        --config.train_data="../../data/advbench/harmful_behaviors.csv" \
        --config.result_prefix="../results/indi_fast/behaviors_llama2_gcg_offset${data_offset}" \
        --config.n_train_data=1 \
        --config.data_offset=$data_offset \
        --config.n_steps=500 \
        --config.test_steps=100 \
        --config.batch_size=256 \
        --config.stop_on_success=False \
        --config.verbose=True
done
