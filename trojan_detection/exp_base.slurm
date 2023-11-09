#!/bin/bash

#SBATCH --job-name="tdc_base"
#SBATCH --gres=gpu:1
#SBATCH --partition="gpu"
#SBATCH --nodelist=cheetah01
#SBATCH --reservation=as9rw_18
#SBATCH --output=logs/job_%x-%j.out
#SBATCH --error=logs/job_%x-%j.err

# Plus old incorrect runs, etc
# Run 1
# knocky python gcg_trojan_gen.py --setting base --n_iters 25 --break_on_success --use_negative_feedback --n_tries 20
# Run 2
# knocky python gcg_trojan_gen.py --setting base --n_iters 30 --break_on_success --use_negative_feedback --n_tries 20
# 46 / 80
# 44 of these have < 20 triggers: [5, 3, 13, 6, 15, 8, 1, 12, 12, 1, 1, 12, 1, 2, 4, 1, 1, 1, 2, 3, 1, 1, 3, 6, 3, 6, 2, 5, 5, 4, 2, 2, 9, 2, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1]

# New run (def not any better)
# Run 1
# knocky python gcg_trojan_gen.py --setting base --n_iters 25 --break_on_success --use_negative_feedback --n_tries 25
# 36 / 80
# 35 of these have < 20 triggers: [3, 2, 8, 10, 11, 3, 2, 5, 1, 2, 7, 1, 1, 2, 2, 1, 4, 2, 1, 2, 3, 1, 3, 1, 1, 5, 1, 2, 2, 1, 2, 1, 1, 1, 1]
# Run 2
# knocky python gcg_trojan_gen.py --setting base --n_iters 30 --break_on_success --use_negative_feedback --n_tries 25
# 41 / 80
# 40 of these have < 20 triggers: [3, 2, 8, 10, 11, 3, 2, 5, 1, 2, 7, 4, 1, 3, 3, 3, 8, 5, 3, 4, 5, 3, 7, 2, 2, 8, 3, 3, 3, 3, 6, 1, 1, 1, 1, 1, 1, 1, 1, 2]


# Newer run
# Run 1-3
knocky python gcg_trojan_gen.py --setting base --n_iters 20 --break_on_success --use_negative_feedback --n_tries 10 --n_want 1
knocky python gcg_trojan_gen.py --setting base --n_iters 25 --break_on_success --use_negative_feedback --n_tries 15 --n_want 1
knocky python gcg_trojan_gen.py --setting base --n_iters 30 --break_on_success --use_negative_feedback --n_tries 20 --n_want 1
# Run 4
knocky python gcg_trojan_gen.py --setting base --n_iters 25 --break_on_success --use_negative_feedback --n_tries 20 --n_want 4
# Run 5
knocky python gcg_trojan_gen.py --setting base --n_iters 30 --break_on_success --use_negative_feedback --n_tries 20 --n_want 4
