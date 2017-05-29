#include <stdio.h>
#include <cuda.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;

__global__ void find_maximum_kernel(float *array, float *max, int *mutex, unsigned int n, int blockSize)
{
	unsigned int index = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int stride = gridDim.x*blockDim.x;
	unsigned int offset = 0;
//	const int size = blockSize;

	__shared__ float cache[256];


	float temp = -1.0;
	while(index + offset < n){
		temp = fmaxf(temp, array[index + offset]);

		offset += stride;
	}

	cache[threadIdx.x] = temp;

	__syncthreads();


	// reduction
	unsigned int i = blockDim.x/2;
	while(i != 0){
		if(threadIdx.x < i){
			cache[threadIdx.x] = fmaxf(cache[threadIdx.x], cache[threadIdx.x + i]);
		}

		__syncthreads();
		i /= 2;
	}

	if(threadIdx.x == 0){
		while(atomicCAS(mutex,0,1) != 0);  //lock
		*max = fmaxf(*max, cache[0]);
		atomicExch(mutex, 0);  //unlock
	}
}

int main(int argc, char** argv)
{
	char* image_name = argv[1];
	float *h_ImageData, *d_ImageData, *h_max, *d_max;
	int *d_mutex;
	int width, height, channels, sizeImage, N;

	Mat image, gray_image;

	image = imread(image_name, -1);
	if(argc !=2 || !image.data) {
		printf("No image Data \n");
		printf("Usage: ./search <file_path>\n");
		return -1;
	}

	if(image.empty()) {
		printf("Couldn't find or open the image...\n");
		return -1;
	}

	cvtColor(image, gray_image, CV_BGR2GRAY);

	width = gray_image.cols;
	height = gray_image.rows;
	channels = gray_image.channels();
	N = width*height*channels;
	sizeImage = sizeof(float)*width*height*channels;

	//h_ImageData = (float *) malloc (sizeImage);
	h_ImageData = (float *)gray_image.data;
	h_max = (float*)malloc(sizeof(float));

	cudaMalloc((void **)&d_ImageData, sizeImage);
	cudaMalloc((void**)&d_max, sizeof(float));
	cudaMalloc((void**)&d_mutex, sizeof(int));
	cudaMemset(d_max, 0, sizeof(float));
	cudaMemset(d_mutex, 0, sizeof(int));

	cudaMemcpy(d_ImageData, h_ImageData, sizeImage, cudaMemcpyHostToDevice);

	int blockSize = 256;
	dim3 dimBlock(blockSize, 1, 1);
	dim3 dimGrid(ceil(width/float(blockSize)), 1, 1);
	find_maximum_kernel<<< dimGrid, dimBlock >>>(d_ImageData, d_max, d_mutex, N, blockSize);

	cudaMemcpy(h_max, d_max, sizeof(float), cudaMemcpyDeviceToHost);
	printf("Maximum number found on gpu was: %f\n", *h_max);

	*h_max = -1.0;
	for(unsigned int i=0;i<N;i++){
		if(h_ImageData[i] > *h_max){
			*h_max = h_ImageData[i];
		}
	}

	printf("Maximum number found on cpu was: %f\n", *h_max);

	free(h_max);
	cudaFree(d_ImageData);
	cudaFree(d_max);
	cudaFree(d_mutex);
	return 0;
}
