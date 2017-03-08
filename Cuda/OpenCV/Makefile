CC = g++ -I/usr/local/include/opencv -I/usr/local/include -L/usr/local/lib -L/usr/local/cuda/lib64 
LDFLAGS = -lopencv_calib3d -lopencv_contrib -lopencv_core -lopencv_features2d -lopencv_flann -lopencv_gpu -lopencv_highgui -lopencv_imgproc -lopencv_legacy -lopencv_ml -lopencv_nonfree -lopencv_objdetect -lopencv_ocl -lopencv_photo -lopencv_stitching -lopencv_superres -lopencv_ts -lopencv_video -lopencv_videostab -lcufft -lnpps -lnppi -lnppc -lcudart -latomic -ltbb -lGL -lGLU -lrt -lpthread -lm -ldl

all: grey_scale

grey_scale: grey_scale.cpp
	$(CC) -o grey_scale grey_scale.cpp $(LDFLAGS)

clean:
	rm -f grey_scale
