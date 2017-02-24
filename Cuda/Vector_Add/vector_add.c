#include <stdio.h>
#include <stdlib.h>

#define SIZE 10

void vec_add(int A[SIZE], int B[SIZE], int C[SIZE]) {
	for(int i=0; i<SIZE; i++) {
		C[i]=A[i]+B[i];
	}
}

void print_vec(int A[SIZE]) {
	for(int i=0; i<SIZE; i++) {
		printf("%d\t", A[i]);
	}
}

int main() {
	int A[SIZE], B[SIZE], C[SIZE];
	
	//initialize vectors
	for(int i=0; i<SIZE; i++) {
		A[i]=rand()%10;
		B[i]=rand()%10;
		C[i]=0;
	}
	
	vec_add(A, B, C);
	
	print_vec(C);
	printf("Done");
	
	return 0;
}

