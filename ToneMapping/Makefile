CUDAC = nvcc
CFLAGS = `pkg-config --cflags opencv`
LIBS = `pkg-config --libs opencv`

search: search.cu
	$(CUDAC) $(LIBS) -o search search.cu $(CFLAGS)
clean:
	rm search