#CUDAC = nvcc
CFLAGS = `pkg-config --cflags opencv`
LIBS = `pkg-config --libs opencv`

# tone_mapping: tone_mapping.cu
# 	$(CUDAC) $(LIBS) -o tone_mapping tone_mapping.cu $(CFLAGS)

#test: test.cu
#	$(CUDAC) $(LIBS) -o test test.cu $(CFLAGS)
#clean:
#	rm test

all: tone_mapping.cu tonemap.cpp
    mpicc $(LIBS) -c tonemap.cpp -o tonemap.o
    nvcc -c tone_mapping.cu -o tone_mapping.o
    mpicc tonemap.o tone_mapping.o $(CFLAGS) -o tonemap

clean:
    rm *.o tonemap