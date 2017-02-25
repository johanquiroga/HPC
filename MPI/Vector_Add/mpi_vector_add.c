#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

#define SIZE 100
#define FROM_MASTER 1
#define FROM_WORKER 2

void print_vec(double A[SIZE]) {
	printf("\n");
	for(int i=0; i<SIZE; i++) {
		printf("%.3f	", A[i]);
	}
}

int main(int argc, char** argv) {
	int numtasks, taskid, nworkers, chunk, extra_data, offset, data_to_send;
	double A[SIZE], B[SIZE], C[SIZE];
	
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
	MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
	
	if (numtasks < 2 ) {
		printf("Need at least two MPI tasks. Quitting...\n");
		MPI_Abort(MPI_COMM_WORLD, 1);
		exit(1);
	}
	nworkers = numtasks - 1;

	if(taskid == 0) {
	
		//initialize vectors
		printf("Initializing vectors...\n");
		for(int i=0; i<SIZE; i++) {
			A[i]=rand()%10;
			B[i]=rand()%10;
		}
	
		//send data chunks of vectors to workers

		data_to_send = SIZE/nworkers;
		extra_data = SIZE%nworkers;
		offset = 0;
		for(int i=1; i<=nworkers; i++) {
			if(i <= extra_data) {
				chunk = data_to_send + 1; 	
			} else {
				chunk = data_to_send;
			}
			printf("Sendind %d columns to task %d, offset=%d\n",chunk,i,offset);
			MPI_Send(&offset, 1, MPI_INT, i, FROM_MASTER, MPI_COMM_WORLD);
			MPI_Send(&chunk, 1, MPI_INT, i, FROM_MASTER, MPI_COMM_WORLD);
			MPI_Send(&A[offset], chunk, MPI_DOUBLE, i, FROM_MASTER, MPI_COMM_WORLD);
			MPI_Send(&B[offset], chunk, MPI_DOUBLE, i, FROM_MASTER, MPI_COMM_WORLD);
			offset += chunk;
		}

		//receive results from workers
		for(int i=1; i<=nworkers; i++) {
			MPI_Recv(&offset, 1, MPI_INT, i, FROM_WORKER, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			MPI_Recv(&chunk, 1, MPI_INT, i, FROM_WORKER, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			MPI_Recv(&C[offset], chunk, MPI_DOUBLE, i, FROM_WORKER, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			printf("Received results from task %d\n", i);
		}

		//print results
		//print_vec(A);
		//print_vec(B);
		print_vec(C);
		printf("\nDone\n");
	}
	
	if(taskid > 0) {
		//receive data to process
		MPI_Recv(&offset, 1, MPI_INT, 0, FROM_MASTER, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(&chunk, 1, MPI_INT, 0, FROM_MASTER, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(&A, chunk, MPI_DOUBLE, 0, FROM_MASTER, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(&B, chunk, MPI_DOUBLE, 0, FROM_MASTER, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		
		//vector add
		for(int i=0; i<chunk; i++) {
			C[i]=A[i]+B[i];
		}

		//send results
		MPI_Send(&offset, 1, MPI_INT, 0, FROM_WORKER, MPI_COMM_WORLD);          	
		MPI_Send(&chunk, 1, MPI_INT, 0, FROM_WORKER, MPI_COMM_WORLD);
		MPI_Send(&C, chunk, MPI_DOUBLE, 0, FROM_WORKER, MPI_COMM_WORLD);
	}	
	MPI_Finalize();
	return 0;
}
