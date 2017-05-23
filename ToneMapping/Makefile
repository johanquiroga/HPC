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
	mpicc -c tonemap.cpp $(LIBS) -o tonemap.o $(CFLAGS)

cuda: test.cu
	nvcc -c test.cu -o test.o

final: tonemap.o test.o
	mpicc tonemap.o test.o -lm $(LIBS) -o tonemap $(CFLAGS)

clean:
	rm *.o tonemap