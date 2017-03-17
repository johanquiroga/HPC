#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define SIZE 2000
#define CHUNKSIZE 10

void printMatrix(int *A) {
	for(int i=0; i<SIZE; i++) {
		printf("|");
		for(int j=0; j<SIZE; j++) {
			printf("%-d ",A[i*SIZE+j]);
		}
		printf("|\n");
	}
	printf("\n");
}

int main() {
	int i, j, k, chunk, ans, nthreads, tid;
	double begin, end;
	int *A = (int *) malloc(SIZE*SIZE*sizeof(int));
	int *B = (int *) malloc(SIZE*SIZE*sizeof(int));
	int *C = (int *) malloc(SIZE*SIZE*sizeof(int));
	int *C_p = (int *) malloc(SIZE*SIZE*sizeof(int));

	printf("Matrix size: %dx%d\n", SIZE, SIZE);
	
	for(i=0; i<SIZE*SIZE; i++) {
		A[i] = rand()%10;
		B[i] = rand()%10;
		C[i] = 0;
		C_p[i] = 0;
	}	

	//Serial version
	begin=omp_get_wtime();

	for(i=0; i<SIZE; i++) {
		for(j=0; j<SIZE; j++) {
			ans=0;
			for(k=0; k<SIZE; k++) {
				ans += A[i*SIZE+k]*B[k*SIZE+j];
			}
			C[i*SIZE+j]=ans;
		}
	}

	end=omp_get_wtime();
	printf("\nTime in serial version: %.5f\n", end-begin);

	//Parallel version
	begin=omp_get_wtime();
	chunk = CHUNKSIZE;
	#pragma omp parallel shared(A, B, C_p, chunk, nthreads) private(i, j, k, ans, tid)
	{
		tid = omp_get_thread_num();
  		if (tid == 0) {
    			nthreads = omp_get_num_threads();
	    		printf("\nNumber of threads = %d\nChunk size: %d\n", nthreads, chunk);
    		}		

		#pragma omp for schedule(static, chunk)
		for(i=0; i<SIZE; i++) {
			for(j=0; j<SIZE; j++) {
				ans=0;
				for(k=0; k<SIZE; k++) {
					ans += A[i*SIZE+k]*B[k*SIZE+j];
				}
				C_p[i*SIZE+j]=ans;
			}
		}
	}
	end=omp_get_wtime();
	printf("\nTime in Parallel version: %.5f\n", end-begin);

	/*printMatrix(A);
	printMatrix(B);
	printMatrix(C);
	printMatrix(C_p);*/

	free(A);
	free(B);
	free(C);
	free(C_p);
	return 0;
}
