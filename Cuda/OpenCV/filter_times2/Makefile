CUDAC = nvcc
CFLAGS = `pkg-config --cflags opencv`
LIBS = `pkg-config --libs opencv`

grey_scale: grey_scale.cu
	$(CUDAC) $(LIBS) -o grey_scale grey_scale.cu $(CFLAGS)
clean:
	rm grey_scale
