#!/bin/bash
#
#SBATCH --job-name=test_tone_mapping
#SBATCH --output=res_tone_mapping.md
#SBATCH --tasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

TMO = "gamma"

echo "imagen|tiempo"
echo ":---:|:---:"

srun tonemap ../images results_$TMO \| $TMO 1.2 0.4
