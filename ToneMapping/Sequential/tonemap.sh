#!/bin/bash
#
#SBATCH --job-name=tonemap
#SBATCH --output=res_tonemap.md
#SBATCH --tasks=1
#SBATCH --nodes=1


echo "imagen|tiempo"
echo ":---:|:---:"

srun tonemap 0.4 1.2 ../images results \|
