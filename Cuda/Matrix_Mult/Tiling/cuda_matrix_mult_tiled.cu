#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include <time.h>
#include <string>

#define TILE_WIDTH 32

__global__ void matrixMultTiled(float* d_A, float* d_B, float* d_C, int width) {
	__shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
	__shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];
	
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	// Identify the row and column of the d_P element to work on
	int Row = by * TILE_WIDTH + ty;
	int Col = bx * TILE_WIDTH + tx; 
	
	float ans = 0;	
	// Loop over the d_M and d_N tiles required to compute d_P element
	for(int m=0; m < (TILE_WIDTH + width - 1)/TILE_WIDTH; m++) {
		// Coolaborative loading of d_M and d_N tiles into shared memory
		if(m*TILE_WIDTH + tx < width && Row < width)
			ds_A[ty][tx] = d_A[Row*width + m*TILE_WIDTH + tx];
		else
			ds_A[ty][tx] = 0.0;
		if(m*TILE_WIDTH + ty < width && Col < width)
			ds_B[ty][tx] = d_B[(m*TILE_WIDTH + ty) * width + Col];
		else
			ds_B[ty][tx] = 0.0;

		__syncthreads();
	
		for(int k=0; k<TILE_WIDTH; k++) {
			ans += ds_A[ty][k] * ds_B[k][tx];
		}
		__syncthreads();
	}
	if(Row < width && Col < width)
		d_C[Row*width+Col]=ans;
}

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

int compareResults(float* h_C, float* d_C, int width) {
	for(int i=0; i<width*width; i++) {
		if(h_C[i] != d_C[i]) {
			return 0;
		}
	}
	return 1;
}

void printMatrix(float *A, int width) {
	for(int i=0; i<width; i++) {
		for(int j=0; j<width; j++) {
			printf("%.2f ", A[i*width+j]);
		}
		printf("\n");
	}
}

void checkError(cudaError_t err, int line) {
	if(err!=cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, line);
		exit(EXIT_FAILURE);
	}
}

void cudaKernelCall(float* h_A, float* h_B, float* h_C, int width, int size, double &time_used, int op) {
	float *d_A, *d_B, *d_C;
	clock_t start, end;
	cudaError_t err = cudaSuccess;
	
	err = cudaMalloc((void **)&d_A, size);
	checkError(err, __LINE__);

	err = cudaMalloc((void **)&d_B, size);
	checkError(err, __LINE__);

	err = cudaMalloc((void **)&d_C, size);
	checkError(err, __LINE__);
	
	start = clock();
	err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
	checkError(err, __LINE__);

	err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
	checkError(err, __LINE__);
	
	int blockSize = 32;
	dim3 dimBlock(blockSize, blockSize, 1);
	dim3 dimGrid(ceil(width/float(blockSize)), ceil(width/float(blockSize)), 1);
	if(op == 0) {
		matrixMultDevice<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, width);
		cudaDeviceSynchronize();
	} else if(op == 1) {
		matrixMultTiled<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, width);
		cudaDeviceSynchronize();
	}
	
	err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
	checkError(err, __LINE__);
	end = clock();
	time_used = ((double)(end - start))/CLOCKS_PER_SEC;
	
	cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}

int main(int argc, char** argv) {
	int width;
	bool f_serial, f_cuda, f_cudaTiled;
	int result = -1;
	f_serial = f_cuda = f_cudaTiled = false;
	if(argc < 2) {
		printf("Matrix size missing!!\n");
		return -1;
	} else {
		width = atoi(argv[1]);
		//printf("Matrix size: %d\n", width);
	}
	//printf("%d,", width);
	for(int i = 2; i < argc; i++) {
		std::string arg = argv[i];
		if(arg == "s")
			f_serial = true;
		else if(arg == "c")
			f_cuda = true;
		else if(arg == "t")
			f_cudaTiled = true;
	}
	clock_t start_serial, end_serial;
	double time_used_serial, time_used_cuda, time_used_tiled, acc;
	time_used_serial = time_used_cuda = time_used_tiled = acc = 0.0;
	int size = width*width*sizeof(float);
	float *h_A = (float *) malloc(size);
	float *h_B = (float *) malloc(size);
	float *h_C = (float *) malloc(size);
	float *h_C_tiled = (float *) malloc(size);
	float *compare_C = (float *) malloc(size);
	
	//init matrix
	for(int i=0; i<width*width; i++) {
		h_A[i]=rand()%10;
		h_B[i]=rand()%10;
	}
		       
	if(f_serial) {
		
		/////////////////////Algoritmo serial//////////////////////////////////
		start_serial = clock();
		matrixMultHost(h_A, h_B, compare_C, width);
		end_serial = clock();
		time_used_serial = ((double)(end_serial - start_serial))/CLOCKS_PER_SEC;
		//printf("Tiempo en serial: %.10f\n", time_used_serial);
		printf("%.10f,", time_used_serial);	
		///////////////////////////////////////////////////////////////////////
	} else
		printf("-,");
	
	if(f_cuda) {
		
		/////////////////////Algoritmo paralelo///////////////////////////
		cudaKernelCall(h_A, h_B, h_C, width, size, time_used_cuda, 0);
		//printf("Tiempo en CUDA: %.10f\n", time_used_cuda);
		printf("%.10f,", time_used_cuda);
		//////////////////////////////////////////////////////////////////
		if(f_serial) {
			acc = time_used_serial/time_used_cuda;
			result = compareResults(compare_C, h_C, width);
		}
	} else
		printf("-,");
	
	if(f_cudaTiled) {
		
		/////////Algoritmo paralelo con memmoria compartida///////////////
		cudaKernelCall(h_A, h_B, h_C_tiled, width, size, time_used_tiled, 1);
		//printf("Tiempo en CUDA: %.10f\n", time_used_cuda);
		printf("%.10f,", time_used_tiled);
		//////////////////////////////////////////////////////////////////
		if(f_cuda) {
			if(f_serial) {
				acc = time_used_serial/time_used_tiled;
				result = compareResults(compare_C, h_C_tiled, width);
			}
			else {
				acc = time_used_cuda/time_used_tiled;
				result = compareResults(h_C_tiled, h_C, width);
			}
		} else {
			if(f_serial) {
				acc = time_used_serial/time_used_tiled;
				result = compareResults(compare_C, h_C_tiled, width);
			} else {
				acc = 0.0;
			}
		}
	} else
		printf("-,");

	//printMatrix(h_A, width);
	//printMatrix(h_B, width);
	//printMatrix(h_C, width);
	//printMatrix(h_C_tiled, width);
	//printMatrix(compare_C, width);
	
	//printf("\nAceleración obtenida: %.10f\n", time_used_serial/time_used_cuda);
	if(acc != 0.0)
		printf("%.10f,", acc);
	else
		printf("-,");

	if(result < 0) {
		printf("-\n");
	} else if(result > 0) {
		printf("Good calculation\n");
	} else {
		printf("Bad calculation\n");
	}
	
	free(h_A); free(h_B); free(h_C); free(compare_C); free(h_C_tiled);
	return 0;
}
