#!/bin/bash

#SBATCH --job-name=rgb2gray
#SBATCH --output=res_rgb2gray.csv
#
#SBATCH --tasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

for i in {1..3}
do
	echo "Imagen: img$i.jpg,,,"
	echo "Iteracion,opencv,cuda,aceleracion"
	for run in {1..20}
	do
		echo -n "$run,"
		srun rgb2gray ../test_files/img$i.jpg
	done
	echo
done
