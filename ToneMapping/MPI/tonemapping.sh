#!/bin/bash
#
#SBATCH --job-name=tone_mapping
#SBATCH --ntasks=3
#SBATCH --gres=gpu:1

TMO="gamma"

echo "imagen|tiempo|id worker"
echo ":---:|:---:|:---:"

mpirun build/tonemapping ../images results/results_$TMO \| $TMO 1.2 0.4 #gamma
#mpirun build/tonemapping ../images results/results_$TMO \| $TMO 1 1 #log
#mpirun build/tonemapping ../images results/results_$TMO \| $TMO 1 150 #adap_log