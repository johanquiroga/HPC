CFLAGS = `pkg-config --cflags opencv`
LIBS = `pkg-config --libs opencv`

tonemap: tonemap.cpp
	gcc $(LIBS) -o tonemap tonemap.cpp $(CFLAGS)
clean:
	rm tonemap