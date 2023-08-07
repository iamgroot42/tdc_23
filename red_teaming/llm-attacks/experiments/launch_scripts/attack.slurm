#!/bin/bash

#SBATCH --job-name="tdc_multi"
#SBATCH --gres=gpu:1
#SBATCH --partition="gpu"
#SBATCH --nodelist=cheetah01
#SBATCH --output=job_%x-%j.out
#SBATCH --error=job_%x-%j.err

# bash run_gcg_individual.sh llama2 behaviors
bash run_gcg_multiple.sh