#include <cuda.h>
#include <cv.h>
#include <highgui.h>

using namespace cv;

void showImage(Mat &image, const char *window) {
	namedWindow(window, CV_WINDOW_NORMAL);
	imshow(window, image);
}

int main(int argc, char** argv)
{
	char* image_name = argv[1];
	Mat hdr;

	image = imread(image_name, CV_LOAD_IMAGE_ANYDEPTH);
	if(argc !=2 || !image.data){
	        printf("No image Data \n");
        	return -1;
	}

	if(image.empty()) {
		printf("Couldn't find or open the image...\n");
		return -1;
	}

	return 0;
}