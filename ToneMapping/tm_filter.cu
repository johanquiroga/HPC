#include <stdio.h>
#include <iostream>
#include <math.h>
#include <cuda.h>

#include "lib/tone_mapping.h"

#define CUDA_CHECK(call) \
    if((call) != cudaSuccess) { \
        cudaError_t err = cudaGetLastError(); \
        std::cerr << "CUDA error calling \""#call"\", code is " << err << std::endl; \
        my_abort(err); }

// Iout(x,y)=(Iin(x,y)⋅2ᶺf)ᶺ(1/g)

__device__ float gamma_correction(float f_stop, float gamma, float val)
{
	return powf((val*powf(2,f_stop)),(1.0/gamma));
}

__global__ void tonemap_kernel(float* imageIn, float* imageOut, int width, int height, int channels, float f_stop,
						float gamma)
{
	int Row = blockDim.y * blockIdx.y + threadIdx.y;
	int Col = blockDim.x * blockIdx.x + threadIdx.x;

	if(Row < height && Col < width) {
        for(int i=0; i<channels; i++) {
            imageOut[(Row*width+Col)*channels+i] = gamma_correction(f_stop, gamma, imageIn[(Row*width+Col)*channels+i]);
        }
	}
}

float tonemap(float *h_ImageData, float *h_ImageOut, int width, int height, int channels, float f_stop, float gamma, int blockSize,
             int sizeImage)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;

    float *d_ImageData, *d_ImageOut;

    CUDA_CHECK(cudaMalloc((void **)&d_ImageData, sizeImage));
    CUDA_CHECK(cudaMalloc((void **)&d_ImageOut, sizeImage));
    CUDA_CHECK(cudaMemcpy(d_ImageData, h_ImageData, sizeImage, cudaMemcpyHostToDevice));

    dim3 dimBlock(blockSize, blockSize, 1);
    dim3 dimGrid(ceil(width/float(blockSize)), ceil(height/float(blockSize)), 1);

    cudaEventRecord(start);
    tonemap_kernel<<<dimGrid, dimBlock>>>(d_ImageData, d_ImageOut, width, height, channels, f_stop, gamma);
    cudaEventRecord(stop);

    cudaDeviceSynchronize();

    CUDA_CHECK(cudaMemcpy(h_ImageOut, d_ImageOut, sizeImage, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_ImageData));
    CUDA_CHECK(cudaFree(d_ImageOut));

    cudaEventElapsedTime(&milliseconds, start, stop);
    return milliseconds/1000.0;
}