#!/bin/bash

#SBATCH --job-name="tdc_large"
#SBATCH --gres=gpu:1
#SBATCH --partition="gpu"
#SBATCH --nodelist=cheetah01
#SBATCH --reservation=as9rw_18
#SBATCH --output=logs/job_%x-%j.out
#SBATCH --error=logs/job_%x-%j.err

# 1. First time run
knocky python gcg_trojan_gen.py --setting large --n_iters 20 --break_on_success
# 2. Collect known triggers
python generate_known_triggers.py large
# Run steps 1-2 until you have enough known triggers
# Re-run, but with focus on increasing recall

# knocky python gcg_trojan_gen.py --setting large --n_iters 50 --keep_all_success

# knocky python gcg_trojan_gen.py --setting large --n_iters 50 --keep_all_success --use_negative_feedback --n_iters_min 20
