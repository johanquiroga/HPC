CUDAC = nvcc
CFLAGS = `pkg-config --cflags opencv`
LIBS = `pkg-config --libs opencv`

rgb2gray: rgb2gray.cu
	$(CUDAC) $(LIBS) -o rgb2gray rgb2gray.cu $(CFLAGS)
clean:
	rm rgb2gray
