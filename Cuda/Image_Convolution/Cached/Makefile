CUDAC = nvcc
CFLAGS = `pkg-config --cflags opencv`
LIBS = `pkg-config --libs opencv`

image_convolution_cached: image_convolution_cached.cu
	$(CUDAC) $(LIBS) -O3 -o image_convolution_cached image_convolution_cached.cu $(CFLAGS)
clean:
	rm image_convolution_cached *.jpg
