#include <stdio.h>
#include <cuda.h>
#include <math.h>

#define TAM 5

__global__ void vecAdd(float* d_A, float* d_B, float* d_C) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if(i<TAM)
		d_C[i] = d_A[i] + d_B[i];
}

void printVector(float* A) {
	for(int i=0; i<TAM; i++) {
		printf("%f ", A[i]);
	}
	printf("\n");
}

int main() {
	cudaError_t err;
	int size = TAM*sizeof(float);
	float *h_A = (float *) malloc(size);
	float *h_B = (float *) malloc(size);
	float *h_C = (float *) malloc(size);
	
	float *d_A, *d_B, *d_C;
	
	//init vectors
	for(int i=0; i<TAM; i++) {
		h_A[i]=rand()%10;
		h_B[i]=rand()%10;
		h_C[i]=0;
	}
	
	err = cudaMalloc((void **)&d_A, size);
	if(err!=cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	err = cudaMalloc((void **)&d_B, size);
	if(err!=cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	err = cudaMalloc((void **)&d_C, size);
	if(err!=cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}
	
	cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_A, size, cudaMemcpyHostToDevice);
	
	vecAdd<<<ceil(TAM/256.0), 256>>>(d_A, d_B, d_C);
	
	cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
	
	printVector(h_A);
	printVector(h_B);
	printVector(h_C);
	free(h_A); free(h_B); free(h_C);
	cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
	return 0;
}
