#!/bin/bash
#
#SBATCH --job-name=test_tone_mapping
#SBATCH --output=res_test_tone_mapping.md
#SBATCH --ntasks=3
#SBATCH --gres=gpu:1

echo "imagen,tiempo,id worker"
echo ":---:|:---:|:---:"

mpirun build/tonemapping 0.4 1.2 images results \|

#for i in {1..5}
#do
	#echo "**Image**: img$i.jpg"
	#echo
	#echo "iteracion|Host|OpenCV|aceleracion Host-OCV|OpenCVGPU|aceleracion OCV-OCVGPU|Cuda|aceleracion OCV-Cuda|aceleracion OCVGPU-Cuda"
	#echo ":---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:"
#	for run in {1..20}
#	do
		#argv=$((run*100))
		#echo -n "$run|"
#		mpirun ./build/tonemapping 0.4 1.2 ./images ./results
#	done
	#echo "Promedios:| | | | | | | | "
	#echo
#done
