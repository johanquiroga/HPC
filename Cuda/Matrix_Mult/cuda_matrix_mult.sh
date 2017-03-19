#!/bin/bash
#
#SBATCH --job-name=cuda_matrix_mult
#SBATCH --output=res_cuda_matrix_mult.csv
#SBATCH --tasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

#echo $CUDA_VISIBLE_DEVICES
echo "Tamaño,secuencial,cuda,aceleracion,resultado"
for run in {1..20}
do
	argv=$((run*100))
	#echo $argv
	./build/cuda_matrix_mult $argv
done
