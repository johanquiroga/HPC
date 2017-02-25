#!/bin/bash
#
#SBATCH --job-name=mpi_matrix_mult
#SBATCH --output=res_mpi_matrix_mult.out
#SBATCH --nodes=3
#SBATCH --ntasks=4
#SBATCH --time=10:00
#SBATCH --mem-per-cpu=100

mpirun mpi_matrix_mult
