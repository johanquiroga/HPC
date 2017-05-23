#CUDAC = nvcc
CFLAGS = `pkg-config --cflags opencv`
CUDAFLAGS = `pkg-config --cflags cuda-8.0`
CUDALIBS = `pkg-config --libs cuda-8.0`
LIBS = `pkg-config --libs opencv`

# tone_mapping: tone_mapping.cu
# 	$(CUDAC) $(LIBS) -o tone_mapping tone_mapping.cu $(CFLAGS)

#test: test.cu
#	$(CUDAC) $(LIBS) -o test test.cu $(CFLAGS)
#clean:
#	rm test

all: mpi cuda final

mpi: tonemap.cpp
	mpicc -c tonemap.cpp -o tonemap.o

cuda: test.cu
	nvcc -c test.cu -o test.o

final: tonemap.o test.o
	mpicc tonemap.o test.o -lcudart $(LIBS) $(CUDALIBS) -o tonemap $(CFLAGS) $(CUDAFLAGS)

clean:
	rm *.o tonemap