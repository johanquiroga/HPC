CUDAC = nvcc
CFLAGS = `pkg-config --cflags opencv`
LIBS = `pkg-config --libs opencv`

tonemap: tonemap.cu
	$(CUDAC) $(LIBS) -o tonemap tonemap.cu $(CFLAGS)
clean:
	rm tonemap