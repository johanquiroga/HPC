#include <stdio.h>
#include <cuda.h>
//#include <cv.h>
//#include <highgui.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/photo.hpp>
#include <vector>
#include <string>

using namespace cv;

__device__ void compute_intensity(float Red, float Green, float Blue)
{

}

void showImage(Mat &image, const char *window) {
	namedWindow(window, CV_WINDOW_NORMAL);
	imshow(window, image);
}

std::string type2str(int type) {
  std::string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

int main(int argc, char** argv)
{
	char* image_name = argv[1];
	Mat hdr;
	Size imageSize;
	int width, height;
//	std::vector<Mat>images;

	printf("%s\n", image_name);
	hdr = imread(image_name, CV_LOAD_IMAGE_ANYDEPTH);
	if(argc !=2 || !hdr.data){
	        printf("No image Data \n");
      	return -1;
	}

//	images.push_back(hdr);
//	Mat ldr;
//	Ptr<TonemapDurand> tonemap = createTonemapDurand(2.2f);
//	tonemap->process(images[0], ldr);
//	imwrite("ldr.png", ldr * 255);

	if(hdr.empty()) {
		printf("Couldn't find or open the image...\n");
		return -1;
	}
	imageSize = hdr.size();
	width = imageSize.width;
	height = imageSize.height;

	//printf("Width: %d\nHeight: %d\n", width, height);
	std::string ty =  type2str( M.type() );
	printf("Matrix: %s %dx%d \n", ty.c_str(), M.cols, M.rows );

	//printf("Channels: %d\nDepth: %d\n", hdr.channels(), hdr.depth());

	return 0;
}