CFLAGS = `pkg-config --cflags opencv`
LIBS = `pkg-config --libs opencv`

tonemap: tonemap.cpp
	g++ $(LIBS) $(CFLAGS) -o tonemap tonemap.cpp
clean:
	rm tonemap