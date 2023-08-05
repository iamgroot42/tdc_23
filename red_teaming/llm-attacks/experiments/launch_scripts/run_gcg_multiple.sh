#!/bin/bash

export WANDB_MODE=disabled

# Optionally set the cache for transformers
# export TRANSFORMERS_CACHE='YOUR_PATH/huggingface'

export n=50
export model=llama2

knocky python -u ../main.py \
    --config="../configs/transfer_llama2.py" \
    --config.attack=gcg \
    --config.train_data="../../data/advbench/harmful_behaviors_de.csv" \
    --config.result_prefix="../results/transfer_llama2_gcg_${n}_progressive" \
    --config.progressive_goals=True \
    --config.stop_on_success=True \
    --config.num_train_models=1 \
    --config.allow_non_ascii=False \
    --config.n_train_data=$n \
    --config.n_test_data=$n \
    --config.n_steps=1000 \
    --config.test_steps=50 \
    --config.batch_size=256
