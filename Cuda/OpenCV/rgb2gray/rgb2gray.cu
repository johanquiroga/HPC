#include <cv.h>
#include <highgui.h>
#include <iostream>
#include <cuda.h>

#define BLUE 0
#define GREEN 1
#define RED 2

using namespace cv;

__global__ void rgb2gray(unsigned char* d_Pin, unsigned char* d_Pout, int n, int m) {
	int Row = blockIdx.y*blockDim.y + threadIdx.y;
	int Col = blockIdx.x*blockDim.x + threadIdx.x;
	
	if((Row < m) && (Col < n)) {
		d_Pout[Row*n+Col] = d_Pin[(Row*n+Col)*3+BLUE]*0.114 + d_Pin[(Row*n+Col)*3+GREEN]*0.587 + d_Pin[(Row*n+Col)*3+RED]*0.299;

	}
}

void showImage(Mat &image, const char *window) {
	namedWindow(window, CV_WINDOW_NORMAL);
	imshow(window, image);
}

int main(int argc, char** argv) {
	cudaError_t err = cudaSuccess;
	char* image_name = argv[1];
	int width, height, sizeImageGrey;
	Mat image, image_out;
	unsigned char *h_ImageData, *d_ImageData, *d_ImageOut, *h_ImageOut;
	Size imageSize; 

	printf("Image name: %s\n", image_name);
	image = imread(image_name, 1);
	if(argc !=2 || !image.data){
	        printf("No image Data \n");
        	return -1;
	}

	if(image.empty()) {
		printf("Couldn't find or open the image...\n");
		return -1;
	}

	imageSize = image.size();
	width = imageSize.width;
	height = imageSize.height;
	sizeImageGrey = sizeof(unsigned char)*width*height;
	
	h_ImageData = (unsigned char *) malloc (sizeImageGrey);
	h_ImageData = image.data;
	h_ImageOut = (unsigned char *) malloc (sizeImageGrey);
	err = cudaMalloc((void **)&d_ImageData, sizeImageGrey);
	if(err != cudaSuccess){
        	printf("Error reservando memoria para d_ImageDara\n");
	 	exit(-1);
	}
	err = cudaMalloc((void **)&d_ImageOut, sizeImageGrey);
	if(err != cudaSuccess){
        	printf("Error reservando memoria para d_ImageOut\n");
	 	exit(-1);
	}
	err = cudaMemcpy(d_ImageData, h_ImageData, sizeImageGrey, cudaMemcpyHostToDevice);
	if(err != cudaSuccess){
        	printf("Error copiando los datos de h_ImageData a d_ImageData\n");
	 	exit(-1);
	}
	
	dim3 dimBlock(32, 32, 1);
	dim3 dimGrid(ceil(width/32.0), ceil(height/32.0));
	PictureKernell<<<dimGrid, dimBlock>>>(d_ImageData, d_ImageOut, width, height);
	
	err = cudaMemcpy(h_ImageOut, d_ImageOut, sizeImageGrey, cudaMemcpyDeviceToHost);
	if(err != cudaSuccess){
        	printf("Error copiando los datos de d_ImageOut a h_ImageOut\n");
	 	exit(-1);
	}
	
	image_out.create(height, width, CV_8UC1);
	image_out.data = h_ImageOut;
	imwrite("image_out.jpg", image_out);

	printf("Done\n");
	showImage(image, "Image In");
	showImage(image_out, "Image out");
	waitKey(0);
	free(h_ImageOut); cudaFree(d_ImageData); cudaFree(d_ImageOut);
	return 0;
}
