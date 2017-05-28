CFLAGS = `pkg-config --cflags opencv`
LIBS = `pkg-config --libs opencv`

tonemap: tonemap.cpp
	g++ -o tonemap tonemap.cpp $(LIBS) $(CFLAGS)
clean:
	rm tonemap