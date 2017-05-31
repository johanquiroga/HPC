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

__global__ void find_maximum_kernel(float *array, float *max, int *mutex, unsigned int n)
{
    unsigned int index = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int stride = gridDim.x*blockDim.x;
    unsigned int offset = 0;
//	const int size = blockSize;

    extern	__shared__ float cache[];


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

__device__ float logarithmic_mapping(float k, float q, float val_pixel, float maxLum){
    return (log10(1 + q * val_pixel))/(log10(1 + k * maxLum));
}

__device__ float gamma_correction(float f_stop, float gamma, float val)
{
    return powf((val*powf(2,f_stop)),(1.0/gamma));
}

__device__ float adaptive_logarithmic_mapping(float lw_max, float ld_max, float lw, float b)
{
    float ld = ((ld_max)/(100*log10f(1+lw_max)))*((logf(1+lw))/(logf(2+8*powf((lw/lw_max),(logf(b)/logf(0.5))))));
    return ld;
}

__global__ void gamma_tonemap_kernel(float* imageIn, float* imageOut, int width, int height, int channels, float f_stop,
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

__global__ void log_tonemap_kernel(float* imageIn, float* imageOut, int width, int height, int channels, float k,
                                   float q, float* max)
{
    int Row = blockDim.y * blockIdx.y + threadIdx.y;
    int Col = blockDim.x * blockIdx.x + threadIdx.x;

    if(Row < height && Col < width) {
        for(int i=0; i<channels; i++) {
            imageOut[(Row*width+Col)*channels+i] =  logarithmic_mapping(k, q, imageIn[(Row*width+Col)*channels+i], *max);
        }
    }
}

__global__ void adaptive_log_tonemap_kernel(float* imageIn, float* imageOut, int width, int height, int channels, float b,
                                            float ld_max, float* max)
{
    int Row = blockDim.y * blockIdx.y + threadIdx.y;
    int Col = blockDim.x * blockIdx.x + threadIdx.x;

    if(Row < height && Col < width) {
        for(int i=0; i<channels; i++) {
            imageOut[(Row*width+Col)*channels+i] =  adaptive_logarithmic_mapping(*max, ld_max, imageIn[(Row*width+Col)*channels+i], b);
        }
    }
}

float log_tonemap(float *h_ImageData, float *h_ImageOut, int width, int height, int channels, float k, float q, int blockSize,
                  int sizeImage)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;

    float *d_ImageData, *d_ImageOut, *d_max;
    int *d_mutex;

    CUDA_CHECK(cudaMalloc((void **)&d_ImageData, sizeImage));
    CUDA_CHECK(cudaMalloc((void **)&d_ImageOut, sizeImage));
    CUDA_CHECK(cudaMalloc((void**)&d_max, sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_mutex, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_max, 0, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_mutex, 0, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_ImageData, h_ImageData, sizeImage, cudaMemcpyHostToDevice));

    cudaEventRecord(start);

    int search_blockSize = 256;
    dim3 dimBlock_search(search_blockSize, 1, 1);
    dim3 dimGrid_search(ceil(width/float(search_blockSize)), 1, 1);
    find_maximum_kernel<<< dimGrid_search, dimBlock_search, sizeof(float)*search_blockSize >>>(d_ImageData, d_max, d_mutex,
            sizeImage/sizeof(float));
    cudaDeviceSynchronize();

    dim3 dimBlock(blockSize, blockSize, 1);
    dim3 dimGrid(ceil(width/float(blockSize)), ceil(height/float(blockSize)), 1);
    log_tonemap_kernel<<<dimGrid, dimBlock>>>(d_ImageData, d_ImageOut, width, height, channels, k, q, d_max);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);

    CUDA_CHECK(cudaMemcpy(h_ImageOut, d_ImageOut, sizeImage, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_ImageData));
    CUDA_CHECK(cudaFree(d_ImageOut));
    CUDA_CHECK(cudaFree(d_max));
    CUDA_CHECK(cudaFree(d_mutex));

    cudaEventElapsedTime(&milliseconds, start, stop);
    return milliseconds/1000.0;

}

float gamma_tonemap(float *h_ImageData, float *h_ImageOut, int width, int height, int channels, float f_stop, float gamma, int blockSize,
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
    gamma_tonemap_kernel<<<dimGrid, dimBlock>>>(d_ImageData, d_ImageOut, width, height, channels, f_stop, gamma);
    cudaEventRecord(stop);

    cudaDeviceSynchronize();

    CUDA_CHECK(cudaMemcpy(h_ImageOut, d_ImageOut, sizeImage, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_ImageData));
    CUDA_CHECK(cudaFree(d_ImageOut));

    cudaEventElapsedTime(&milliseconds, start, stop);
    return milliseconds/1000.0;
}

float adaptive_log_tonemap(float *h_ImageData, float *h_ImageOut, int width, int height, int channels, float b, float ld_max, int blockSize,
                           int sizeImage)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;

    float *d_ImageData, *d_ImageOut, *d_max;
    int *d_mutex;

    CUDA_CHECK(cudaMalloc((void **)&d_ImageData, sizeImage));
    CUDA_CHECK(cudaMalloc((void **)&d_ImageOut, sizeImage));
    CUDA_CHECK(cudaMalloc((void**)&d_max, sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_mutex, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_max, 0, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_mutex, 0, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_ImageData, h_ImageData, sizeImage, cudaMemcpyHostToDevice));

    cudaEventRecord(start);

    int search_blockSize = 256;
    dim3 dimBlock_search(search_blockSize, 1, 1);
    dim3 dimGrid_search(ceil(width/float(search_blockSize)), 1, 1);
    find_maximum_kernel<<< dimGrid_search, dimBlock_search, sizeof(float)*search_blockSize >>>(d_ImageData, d_max, d_mutex,
            sizeImage/sizeof(float));
    cudaDeviceSynchronize();

    dim3 dimBlock(blockSize, blockSize, 1);
    dim3 dimGrid(ceil(width/float(blockSize)), ceil(height/float(blockSize)), 1);
    adaptive_log_tonemap_kernel<<<dimGrid, dimBlock>>>(d_ImageData, d_ImageOut, width, height, channels, b, ld_max, d_max);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);

    CUDA_CHECK(cudaMemcpy(h_ImageOut, d_ImageOut, sizeImage, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_ImageData));
    CUDA_CHECK(cudaFree(d_ImageOut));
    CUDA_CHECK(cudaFree(d_max));
    CUDA_CHECK(cudaFree(d_mutex));

    cudaEventElapsedTime(&milliseconds, start, stop);
    return milliseconds/1000.0;
}