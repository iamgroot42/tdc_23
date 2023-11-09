#!/bin/bash

#SBATCH --job-name="tdc_large"
#SBATCH --gres=gpu:1
#SBATCH --partition="gpu"
#SBATCH --nodelist=cheetah01
#SBATCH --reservation=as9rw_18
#SBATCH --output=logs/job_%x-%j.out
#SBATCH --error=logs/job_%x-%j.err

# Run 1
# knocky python gcg_trojan_gen.py --setting large --n_iters 25 --break_on_success --use_negative_feedback --n_tries 20
# Run 2
# knocky python gcg_trojan_gen.py --setting large --n_iters 30 --break_on_success --use_negative_feedback --n_tries 20
# 58 / 80
# 58 of these have < 20 triggers: [5, 6, 3, 1, 7, 5, 3, 10, 4, 2, 18, 5, 2, 4, 6, 2, 2, 5, 4, 3, 9, 1, 6, 8, 1, 5, 3, 3, 3, 12, 1, 1, 3, 6, 7, 7, 2, 3, 1, 8, 3, 1, 1, 1, 1, 1, 4, 4, 3, 2, 1, 2, 2, 6, 1, 1, 2, 1]

# Run 3 (partial)
# knocky python gcg_trojan_gen.py --setting large --n_iters 35 --break_on_success --use_negative_feedback --n_tries 20
# 61 / 80
# 59 of these have < 20 triggers: [12, 13, 5, 1, 9, 8, 3, 16, 5, 2, 8, 4, 8, 8, 2, 4, 7, 7, 7, 17, 5, 7, 10, 3, 9, 8, 5, 3, 1, 1, 6, 11, 11, 15, 2, 3, 1, 8, 4, 2, 3, 1, 5, 5, 11, 7, 6, 2, 3, 3, 3, 13, 1, 3, 2, 1, 1, 1, 3]

# Run 4 (and beyond)
# knocky python gcg_trojan_gen.py --setting large --n_iters 35 --break_on_success --use_negative_feedback --n_tries 20 --n_want 1
# knocky python gcg_trojan_gen.py --setting large --n_iters 40 --break_on_success --use_negative_feedback --n_tries 20 --n_want 1
# knocky python gcg_trojan_gen.py --setting large --n_iters 45 --break_on_success --use_negative_feedback --n_tries 20 --n_want 1
# 72 / 80
# 70 of these have < 20 triggers: [12, 13, 5, 1, 9, 8, 3, 16, 5, 2, 8, 4, 8, 8, 2, 4, 7, 7, 7, 17, 5, 7, 10, 3, 9, 8, 5, 3, 1, 1, 6, 11, 11, 15, 2, 3, 1, 8, 4, 2, 3, 1, 5, 5, 11, 7, 6, 2, 3, 3, 3, 13, 1, 3, 2, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# knocky python gcg_trojan_gen.py --setting large --n_iters 50 --break_on_success --use_negative_feedback --n_tries 20 --n_want 1

# Extra run
# knocky python gcg_trojan_gen.py --setting large --n_iters 25 --break_on_success --use_negative_feedback --n_tries 20 --n_want 5

# More extra run
knocky python gcg_trojan_gen.py --setting large --n_iters 40 --break_on_success --use_negative_feedback --n_tries 25 --n_want 1
