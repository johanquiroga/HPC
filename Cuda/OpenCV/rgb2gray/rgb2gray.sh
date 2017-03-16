#!/bin/bash

#SBATCH --job-name=rgb2gray
#SBATCH --output=res_rgb2gray.out
#
#SBATCH --tasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

srun rgb2gray ../test_files/Kylo_Ren.jpg
