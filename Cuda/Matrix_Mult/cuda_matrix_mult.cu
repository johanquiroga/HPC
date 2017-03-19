#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include <time.h>

#define TAM 2

__global__ void matrixMultDevice(float* d_A, float* d_B, float* d_C, int width) {
	int Row = blockDim.y * blockIdx.y + threadIdx.y;
	int Col = blockDim.x * blockIdx.x + threadIdx.x;
	if(Row < width && Col < width) {
		float ans = 0.0;
		for(int k=0; k<width; k++) {
			ans += d_A[Row*width+k]*d_B[k*width+Col];
		}
		d_C[Row*width+Col]=ans;
	}
}

void matrixMultHost(float *h_A, float *h_B, float *h_C, int width) {
	float ans;
	for(int i=0; i<width; i++) {
		for(int j=0; j<width; j++) {
			ans = 0.0;
			for(int k=0; k<width; k++) {
				ans += h_A[i*width+k]*h_B[k*width+j];
			}
			h_C[i*width+j]=ans;
		}
	}
}

bool compareResults(float* h_C, float* d_C, int width) {
	for(int i=0; i<width*width; i++) {
		if(h_C[i] != d_C[i]) {
			return false;
		}
	}
	return true;
}

void printMatrix(float *A, int width) {
	for(int i=0; i<width; i++) {
		for(int j=0; j<width; j++) {
			printf("%.2f ", A[i*width+j]);
		}
		printf("\n");
	}
}

void checkError(cudaError_t err) {
	if(err!=cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}
}
	
int main(int argc, char** argv) {
	int width = TAM;
	if(argc != 2) {
		//printf("Matrix size missing!!, using %d\n", TAM);
		//exit(EXIT_FAILURE);
	} else {
		width = atoi(argv[1]);
		//printf("Matrix size: %d\n", width);
	}
	printf("%d,", width);
	clock_t start_serial, end_serial, start_cuda, end_cuda;
	double time_used_serial, time_used_cuda;
	cudaError_t err = cudaSuccess;
	size_t size = width*width*sizeof(float);
	float *h_A = (float *) malloc(size);
	float *h_B = (float *) malloc(size);
	float *h_C = (float *) malloc(size);
	float *compare_C = (float *) malloc(size);
	
	float *d_A = NULL, *d_B = NULL, *d_C = NULL;
	
	//init matrix
	for(int i=0; i<width*width; i++) {
		h_A[i]=rand()%10;
		h_B[i]=rand()%10;
	}
	
	start_serial = clock();
	matrixMultHost(h_A, h_B, compare_C, width);
	end_serial = clock();
	time_used_serial = ((double)(end_serial - start_serial))/CLOCKS_PER_SEC;
	//printf("Tiempo en serial: %.10f\n", time_used_serial);
	printf("%.10f,", time_used_serial);

	err = cudaMalloc((void **)&d_A, size);
	checkError(err);

	err = cudaMalloc((void **)&d_B, size);
	checkError(err);

	err = cudaMalloc((void **)&d_C, size);
	checkError(err);
	
	start_cuda = clock();
	err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
	checkError(err);

	err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
	checkError(err);
	
	dim3 dimGrid(ceil(width/32.0), ceil(width/32.0), 1);
	dim3 dimBlock(32, 32, 1);
	matrixMultDevice<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, width);
	
	err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
	checkError(err);
	end_cuda = clock();
	time_used_cuda = ((double)(end_cuda - start_cuda))/CLOCKS_PER_SEC;
	//printf("Tiempo en CUDA: %.10f\n", time_used_cuda);
	printf("%.10f,", time_used_cuda);

	//printMatrix(h_A, width);
	//printMatrix(h_B, width);
	//printMatrix(h_C, width);
	//printMatrix(compare_C, width);
	
	//printf("\nAceleración obtenida: %.10f\n", time_used_serial/time_used_cuda);
	printf("%.10f,", time_used_serial/time_used_cuda);

	if(compareResults(compare_C, h_C, width)) {
		printf("Good calculation\n");
	} else {
		printf("Bad calculation\n");
	}
	
	free(h_A); free(h_B); free(h_C); free(compare_C);
	cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
	return 0;
}
