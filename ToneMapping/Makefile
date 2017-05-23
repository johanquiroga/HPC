#CUDAC = nvcc
CFLAGS = `pkg-config --cflags opencv`
LIBS = `pkg-config --libs opencv`

# tone_mapping: tone_mapping.cu
# 	$(CUDAC) $(LIBS) -o tone_mapping tone_mapping.cu $(CFLAGS)

#test: test.cu
#	$(CUDAC) $(LIBS) -o test test.cu $(CFLAGS)
#clean:
#	rm test

all: mpi cuda final

mpi: tonemap.cpp
	mpicc $(LIBS) -c tonemap.cpp -o tonemap.o $(CFLAGS)

cuda: test.cu
	nvcc -c test.cu -o test.o

final: tonemap.o test.o
	mpicc tonemap.o test.o -lm -o tonemap

clean:
	rm *.o tonemap