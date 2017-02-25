#!/bin/bash
#
#SBATCH --job-name=vector_add
#SBATCH --output=res_vector_add.out
#SBATCH --nodes=3
#SBATCH --ntasks=4
#SBATCH --time=10:00
#SBATCH --mem-per-cpu=100

mpirun vector_add
