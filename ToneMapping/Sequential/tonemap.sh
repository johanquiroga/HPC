#!/bin/bash
#
#SBATCH --job-name=tonemap
#SBATCH --output=res_tonemap.md

echo "imagen|tiempo"
echo ":---:|:---:"

srun tonemap 0.4 1.2 images results \|
