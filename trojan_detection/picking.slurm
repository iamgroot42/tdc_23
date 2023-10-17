#!/bin/bash

#SBATCH --job-name="pick_base"
#SBATCH --gres=gpu:1
#SBATCH --partition="gpu"
#SBATCH --nodelist=jaguar06
#SBATCH --output=logs/job_%x-%j.out
#SBATCH --error=logs/job_%x-%j.err

knocky python generate_known_triggers.py base
