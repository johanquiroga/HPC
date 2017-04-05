#include <cv.h>
#include <highgui.h>
#include <iostream>
#include <cuda.h>
#include <time.h>

#define BLUE 0
#define GREEN 1
#define RED 2

using namespace cv;

__global__ void rgb2gray(unsigned char* d_Pin, unsigned char* d_Pout, int width, int height) {
	int Row = blockIdx.y*blockDim.y + threadIdx.y;
	int Col = blockIdx.x*blockDim.x + threadIdx.x;
	
	if((Row < height) && (Col < width)) {
		d_Pout[Row*width+Col] = d_Pin[(Row*width+Col)*3+BLUE]*0.114 + d_Pin[(Row*width+Col)*3+GREEN]*0.587 + d_Pin[(Row*width+Col)*3+RED]*0.299;

	}
}

void showImage(Mat &image, const char *window) {
	namedWindow(window, CV_WINDOW_NORMAL);
	imshow(window, image);
}

void h_sobelFilter(unsigned char* imageIn, unsigned char* imageOut, int width, int height, int maskWidth, char* M) {
	for(int i=0; i<height; i++) {
		for(int j=0; j<width; j++) {
			int ni = i - (maskWidth/2);		//-1
			int nj = j - (maskWidth/2);		//-1
			int res = 0;
			for(int k=0; k < maskWidth; k++) {
				for(int w=0; w < maskWidth; w++) {
					if((ni + k >=0 && ni + k <height) && (nj + w >=0 && nj + w <width)) {
						res += imageIn[(ni + k)*width + (nj + w)] * M[k*maskWidth + w];
					}
				}
			}
			if(res < 0)
				res = 0;
			else
				if(res > 255)
					res = 255;
			imageOut[i*width+j] = (unsigned char)res;
		}
	}
}

int main(int argc, char** argv) {
	cudaError_t err = cudaSuccess;
	char* image_name = argv[1], *d_M;
	char h_M[] = {-1,0,1,-2,0,2,-1,0,1};
	clock_t start_cuda, end_cuda, start_opencv, end_opencv;
	double time_used_cuda, time_used_opencv;
	int width, height, sizeImageGrey, sizeImage;
	Mat image, image_out_cuda, image_out_opencv, image_out_host, image_out_opencv_gpu;
	unsigned char *h_ImageData, *d_ImageData, *d_ImageOut, *h_ImageOut, *image_sobel, *d_image_Sobel, *cv_gray_image;
	Size imageSize; 

	//printf("Image name: %s\n", image_name);
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
	sizeImage = sizeof(unsigned char)*width*height*image.channels();
	sizeImageGrey = sizeof(unsigned char)*width*height;
	
	h_ImageData = (unsigned char *) malloc (sizeImage);
	h_ImageData = image.data;
	h_ImageOut = (unsigned char *) malloc (sizeImageGrey);
	err = cudaMalloc((void **)&d_ImageData, sizeImage);
	if(err != cudaSuccess){
        	printf("Error reservando memoria para d_ImageData\n");
	 	exit(-1);
	}
	err = cudaMalloc((void **)&d_ImageOut, sizeImageGrey);
	if(err != cudaSuccess){
        	printf("Error reservando memoria para d_ImageOut\n");
	 	exit(-1);
	}

	// Start conversion with OpenCV
	start_opencv = clock();
	cvtColor(image, image_out_opencv, CV_BGR2GRAY);
	cv_gray_image = (unsigned char *) malloc (sizeImageGrey);
	cv_gray_image = image_out_opencv.data;
	h_sobelFilter(cv_gray_image, image_sobel, width, height, 3, h_M);
	image_out_host.create(height, width, CV_8UC1);
	image_out_host.data = image_sobel;
	imwrite("image_out_host.jpg", image_out_host);
	end_opencv = clock();
	// End conversion
	time_used_opencv = ((double) (end_opencv - start_opencv)) /CLOCKS_PER_SEC;
	//printf("Tiempo algoritmo OpenCV: %.10f\n", time_used_opencv);
	printf("%.10f,", time_used_opencv);

	imwrite("image_out_opencv.jpg", image_out_opencv);

	// Start conversion with cuda	
	start_cuda = clock();
	err = cudaMemcpy(d_ImageData, h_ImageData, sizeImage, cudaMemcpyHostToDevice);
	if(err != cudaSuccess){
        	printf("Error copiando los datos de h_ImageData a d_ImageData\n");
	 	exit(-1);
	}
	
	dim3 dimBlock(32, 32, 1);
	dim3 dimGrid(ceil(width/32.0), ceil(height/32.0));
	rgb2gray<<<dimGrid, dimBlock>>>(d_ImageData, d_ImageOut, width, height);
		
	err = cudaMemcpy(h_ImageOut, d_ImageOut, sizeImageGrey, cudaMemcpyDeviceToHost);
	if(err != cudaSuccess){
        	printf("Error copiando los datos de d_ImageOut a h_ImageOut\n");
	 	exit(-1);
	}
	end_cuda = clock();
	// End conversion
	time_used_cuda = ((double) (end_cuda - start_cuda)) /CLOCKS_PER_SEC;
	//printf("Tiempo algoritmo en CUDA: %.10f\n", time_used_cuda);
	printf("%.10f,", time_used_cuda);

	image_out_cuda.create(height, width, CV_8UC1);
	image_out_cuda.data = h_ImageOut;
	imwrite("image_out_cuda.jpg", image_out_cuda);

	printf("%.10f\n", time_used_opencv/time_used_cuda);
		
	printf("Done\n\n");
	//showImage(image, "Image In");
	//showImage(image_out_cuda, "Image out CUDA");
	//showImage(image_out_opencv, "Image out OpenCV");
	showImage(image_out_host, "Image out OpenCV");
	//waitKey(0);
	free(h_ImageOut); cudaFree(d_ImageData); cudaFree(d_ImageOut);
	return 0;
}
