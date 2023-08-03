#!/bin/bash

export WANDB_MODE=disabled

# Optionally set the cache for transformers
# export TRANSFORMERS_CACHE='YOUR_PATH/huggingface'

export model=$1 # llama2 or vicuna
export setup=$2 # behaviors or strings

# for data_offset in 0 10 20 30 40 50
for data_offset in 0 4 8 12 16 20 24 28 32 36 40 44 48
do

    knocky python -u ../main.py \
        --config="../configs/individual_${model}.py" \
        --config.attack=gcg \
        --config.train_data="../../data/advbench/harmful_${setup}.csv" \
        --config.result_prefix="../results/individual_${setup}_${model}_gcg_offset${data_offset}" \
        --config.n_train_data=4 \
        --config.data_offset=$data_offset \
        --config.n_steps=1000 \
        --config.test_steps=50 \
        --config.batch_size=256

done
