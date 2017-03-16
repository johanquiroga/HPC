#!/bin/bash

#SBATCH --job-name=rgb2gray
#SBATCH --output=res_rgb2gray.out
#
#SBATCH --tasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

for run in {1..20}
do
	echo $run
	srun rgb2gray ../test_files/Kylo_Ren.jpg
done
