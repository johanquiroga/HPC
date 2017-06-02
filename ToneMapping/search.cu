#include <stdio.h>
#include <cuda.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
__global__ void find_maximum_kernel(float *array, float *max, int *mutex, unsigned int n)
{
	unsigned int index = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int stride = gridDim.x*blockDim.x;
	unsigned int offset = 0;
//	const int size = blockSize;

	extern __shared__ float cache[];

	float temp = -1.0;
	while(((index + offset)*3 + 2) < n) {
		float B, G, R, L;

		B = array[(index + offset)*3 + BLUE];
		G = array[(index + offset)*3 + GREEN];
		R = array[(index + offset)*3 + RED];
		L = rgb2Lum(B, G, R);

		temp = fmaxf(temp, L);

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

	Mat image, xyz_image, xyz[3];

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

	cvtColor(image, xyz_image, CV_BGR2XYZ);
//	split(xyz_image,xyz);

	Mat y_channel( xyz_image.rows, xyz_image.cols, CV_32FC1 );
	int from_to[] = { 1,0 };
	mixChannels( &xyz_image, 1, &y_channel, 1, from_to, 1 );

	width = y_channel.cols;
	height = y_channel.rows;
	channels = y_channel.channels();
	N = width*height*channels;
	sizeImage = sizeof(float)*width*height*channels;

	//h_ImageData = (float *) malloc (sizeImage);
	h_ImageData = (float *)y_channel.data;
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
	find_maximum_kernel<<< dimGrid, dimBlock, sizeof(float)*blockSize >>>(d_ImageData, d_max, d_mutex, N, blockSize);

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
