#!/bin/bash

#SBATCH --job-name="large_troj"
#SBATCH --gres=gpu:1
#SBATCH --partition="gpu"
#SBATCH --nodelist=jaguar06
#SBATCH --output=logs/job_%x-%j.out
#SBATCH --error=logs/job_%x-%j.err

# knocky python gcg_trojan_gen.py base
knocky python gcg_trojan_gen.py large
