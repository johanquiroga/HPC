#include <stdio.h>
#include <cuda.h>
#include <math.h>

#define TAM 5

__global__ void vecAdd(float* d_A, float* d_B, float* d_C) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i<TAM)
		d_C[i] = d_A[i] + d_B[i];
}

void SerialvecAdd(float* h_A, float* h_B, float* compare_C) {
	for(int i=0; i<TAM; i++) {
		compare_C[i] = h_A[i] + h_B[i];
	}
}

bool compareResults(float* h_C, float* d_C) {
	for(int i=0; i<TAM; i++) {
		if(h_C[i] != d_C[i]) {
			return false;
		}
	}
	return true;
}

void printVector(float A[TAM]) {
	for(int i=0; i<TAM; i++) {
		printf("%.2f ", A[i]);
	}
	printf("\n");
}

int main() {
	cudaError_t err = cudaSuccess;
	size_t size = TAM*sizeof(float);
	float *h_A = (float *) malloc(size);
	float *h_B = (float *) malloc(size);
	float *h_C = (float *) malloc(size);
	float *compare_C = (float *) malloc(size);
	
	float *d_A = NULL, *d_B = NULL, *d_C = NULL;
	
	//init vectors
	for(int i=0; i<TAM; i++) {
		h_A[i]=rand()%10;
		h_B[i]=rand()%10;
//		h_C[i]=0;
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
	cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
	
	vecAdd<<<ceil(TAM/256.0), 256>>>(d_A, d_B, d_C);
	
	cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
	
	SerialvecAdd(h_A, h_B, compare_C);
	//printVector(h_A);
	//printVector(h_B);
	//printVector(h_C);
	//printVector(compare_C);
	if(compareResults(compare_C, h_C)) {
		printf("Good calculation");
	} else {
		printf("Bad calculation");
	}
	free(h_A); free(h_B); free(h_C); free(compare_C);
	cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
	return 0;
}
