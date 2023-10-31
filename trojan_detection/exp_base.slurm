#!/bin/bash

#SBATCH --job-name="tdc_base"
#SBATCH --gres=gpu:1
#SBATCH --partition="gpu"
#SBATCH --nodelist=cheetah01
#SBATCH --reservation=as9rw_18
#SBATCH --output=logs/job_%x-%j.out
#SBATCH --error=logs/job_%x-%j.err

# Run 1
knocky python gcg_trojan_gen.py --setting base --n_iters 20 --break_on_success --use_negative_feedback
python generate_known_triggers.py base

# Run 2
knocky python gcg_trojan_gen.py --setting base --n_iters 25 --break_on_success --use_negative_feedback
python generate_known_triggers.py base

# Run 3
knocky python gcg_trojan_gen.py --setting base --n_iters 30 --break_on_success --use_negative_feedback
python generate_known_triggers.py base

# Run 4
knocky python gcg_trojan_gen.py --setting base --n_iters 50 --keep_all_success --use_negative_feedback --n_tries 10
python generate_known_triggers.py base
