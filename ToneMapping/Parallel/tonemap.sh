#!/bin/bash
#
#SBATCH --job-name=tone_mapping
#SBATCH --tasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

TMO="log"

echo "imagen|tiempo"
echo ":---:|:---:"

#srun tonemap ../images results/results_$TMO \| $TMO 1.2 0.4 #gamma
srun tonemap ../images results/results_$TMO \| $TMO 1 1 #log
#srun tonemap ../images results/results_$TMO \| $TMO 1 #adap_log