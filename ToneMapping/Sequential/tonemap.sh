#!/bin/bash
#
#SBATCH --job-name=tone_mapping
#SBATCH --tasks=1
#SBATCH --nodes=1

TMO="gamma"

echo "imagen|tiempo"
echo ":---:|:---:"

srun tonemap ../images results/results_$TMO \| $TMO 1.2 0.4 #gamma
#srun tonemap ../images results/results_$TMO \| $TMO 1 1 #log
#srun tonemap ../images results/results_$TMO \| $TMO 1 150 #adap_log