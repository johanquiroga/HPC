CUDAC = nvcc
CFLAGS = `pkg-config --cflags opencv`
LIBS = `pkg-config --libs opencv`

image_convolution: image_convolution.cu
	$(CUDAC) $(LIBS) -o image_convolution image_convolution.cu $(CFLAGS)
clean:
	rm image_convolution *.jpg
