#!/bin/bash
#
#SBATCH --job-name=cuda_vector_add
#SBATCH --output=res_cuda_vector_add.out
#SBATCH --tasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

echo $CUDA_VISIBLE_DEVICES
mpirun cuda_vector_add
