#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

#define SIZE 100
#define FROM_MASTER 1 //msg type
#define FROM_WORKER 2 //msg_type

int main(int argc, char** argv) {
	int numtasks, taskid, nworkers, rows, extra, offset, rows_per_task, err_code;
	double A[SIZE][SIZE], B[SIZE][SIZE], C[SIZE][SIZE];
	int i, j, k;
	
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
	MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
	
	if(numtasks < 2) {
		printf("Need at least two tasks...\n");
		MPI_Abort(MPI_COMM_WORLD, err_code);
		exit(1);
	}
	nworkers = numtasks - 1;
	
	if(taskid == 0) {
		
		//Initialize Matrix
		for(i=0; i<SIZE; i++) {
			for(j=0; j<SIZE; j++) {
				A[i][j]=rand()%10;
				B[i][j]=rand()%10;
			}
		}
		
		//send data to workers
		rows_per_task = SIZE/nworkers;
		extra = SIZE%nworkers;
		offset = 0;
		
		for(i=1; i<=nworkers; i++) {
			if(i <= extra) {
				rows = rows_per_task + 1;
			} else {
				rows = rows_per_task;
			}
			printf("Sending %d rows to task %d, offset=%d\n",rows,i,offset);
			MPI_Send(&offset, 1, MPI_INT, i, FROM_MASTER, MPI_COMM_WORLD);
			MPI_Send(&rows, 1, MPI_INT, i, FROM_MASTER, MPI_COMM_WORLD);
			MPI_Send(&A[offset][0], rows*SIZE, MPI_DOUBLE, i, FROM_MASTER, MPI_COMM_WORLD);
			MPI_Send(&B, SIZE*SIZE, MPI_DOUBLE, i, FROM_MASTER, MPI_COMM_WORLD);
			offset += rows;
		}

		//Receive results from workers
		for(i=1; i<=nworkers; i++) {
			MPI_Recv(&offset, 1, MPI_INT, i, FROM_WORKER, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			MPI_Recv(&rows, 1, MPI_INT, i, FROM_WORKER, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			MPI_Recv(&C[offset][0], rows*SIZE, MPI_DOUBLE, i, FROM_WORKER, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			printf("Received results from task %d\n", i);
		}
		
		//print results
		/*printf("\nMatrix A:");
		for(i=0; i<SIZE; i++) {
			printf("\n");
			for(j=0; j<SIZE; j++) {
				printf("%.3f	", A[i][j]);
			}
		}
		printf("\n");
			
		printf("\nMatrix B:");
		for(i=0; i<SIZE; i++) {
			printf("\n");
			for(j=0; j<SIZE; j++) {
				printf("%.3f	", B[i][j]);
			}
		}
		printf("\n");*/
		
		printf("\nMatrix C:");
		for(i=0; i<SIZE; i++) {
			printf("\n");
			for(j=0; j<SIZE; j++) {
				printf("%.3f	", C[i][j]);
			}
		}
		
		printf("\nDone\n");
	}
	
	if(taskid > 0) {
		//receive data from master
		MPI_Recv(&offset, 1, MPI_INT, 0, FROM_MASTER, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(&rows, 1, MPI_INT, 0, FROM_MASTER, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(&A, rows*SIZE, MPI_DOUBLE, 0, FROM_MASTER, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(&B, SIZE*SIZE, MPI_DOUBLE, 0, FROM_MASTER, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		
		//do matrix mult with data
		for(i=0; i<SIZE; i++) {
			for(j=0; j<SIZE; j++) {
				C[i][j]=0.0;
				for(k=0; k<SIZE; k++) {
					C[i][j] += A[i][k] * B[k][j];
				}
			}
		}
				
		//send back results to master
		MPI_Send(&offset, 1, MPI_INT, 0, FROM_WORKER, MPI_COMM_WORLD);
		MPI_Send(&rows, 1, MPI_INT, 0, FROM_WORKER, MPI_COMM_WORLD);
		MPI_Send(&C, rows*SIZE, MPI_DOUBLE, 0, FROM_WORKER, MPI_COMM_WORLD);
	}
	MPI_Finalize();
	return 0;
}
