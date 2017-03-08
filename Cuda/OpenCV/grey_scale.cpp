#include <cv.h>
#include <highgui.h>
#include <iostream>

using namespace std;
using namespace cv;

int main(int argc, char** argv) {
	//cudaError_t err = cudaSuccess;
	char* image_name = argv[1];
	Mat image;
	image = imread(image_name, 0);
	
	if(image.empty()) {
		cout << "Couldn't find or open the image..." << endl;
	}
	
	namedWindow("Display window", WINDOW_AUTOSIZE);
	imshow("Display window", image);
	waitKey(0);
	return 0;
}
