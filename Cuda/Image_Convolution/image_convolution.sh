#!/bin/bash
#
#SBATCH --job-name=image_convolution
#SBATCH --output=res_image_convolution.md
#SBATCH --tasks=1
#SBATCH --nodes=1
#SBATCH --time=10:00
#SBATCH --gres=gpu:1

#echo $CUDA_VISIBLE_DEVICES
for i in {1..3}
do
	echo "**Image**: img$i.jpg"
	echo
	echo "iteracion|Host|OpenCV|aceleracion Host-OCV|OpenCVGPU|aceleracion OCV-OCVGPU|Cuda|aceleracion OCV-Cuda|aceleracion OCVGPU-Cuda"
	echo ":---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:"
	for run in {1..20}
	do
		#argv=$((run*100))
		echo -n "$run|"
		srun image_convolution ../OpenCV/test_files/img$i.jpg
	done
	echo
done
