#include <cv.h>
#include <highgui.h>
#include <cuda.h>
#include <time.h>
#include "opencv2/gpu/gpu.hpp"

#define BLUE 0
#define GREEN 1
#define RED 2

using namespace cv;

__global__ void d_sobelFilter(unsigned char* imageIn, unsigned char* imageOut, int width, int height, int maskWidth, char* M) {
	int Row = blockDim.y * blockIdx.y + threadIdx.y;
	int Col = blockDim.x * blockIdx.x + threadIdx.x;

	int nrow = Row - (maskWidth/2);
	int ncol = Col - (maskWidth/2);
	int res = 0;

	if(Row < height && Col < width) {
		for(int i=0; i<maskWidth; i++) {
			for(int j=0; j<maskWidth; j++) {
				if((nrow + i >= 0 && nrow + i < height) && (ncol + j >= 0 && ncol + j < width)) {
					res += imageIn[(nrow + i)*width + (ncol + j)] * M[i*maskWidth + j];
				}
			}
		}
		if(res < 0)
			res = 0;
		else
			if(res > 255)
				res = 255;
		imageOut[Row*width+Col] = (unsigned char)res;
	}
}

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
					if((ni + k >= 0 && ni + k < height) && (nj + w >= 0 && nj + w < width)) {
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
	clock_t start_cuda, end_cuda, start_opencv, end_opencv, start_host, end_host, start_opencv_gpu, end_opencv_gpu;
	double time_used_cuda, time_used_opencv, time_used_host, time_used_opencv_gpu;
	int width, height, sizeImageGrey, sizeImage;
	Mat image, image_out_cuda, image_gray_opencv, image_gray_host, image_out_host, image_out_opencv_gpu, image_out_opencv, abs_image_out_opencv;
	unsigned char *h_ImageData, *d_ImageData, *d_ImageOut, *h_ImageOut, *image_sobel, *d_image_Sobel, *cv_gray_image;
	Size imageSize;
	gpu::GpuMat src, dst, gray;
	bool show_flag = false;

	//printf("Image name: %s\n", image_name);
	image = imread(image_name, 1);
	if(argc < 2 || !image.data){
	        printf("No image Data \n");
        	return -1;
	} else if(argc == 3) {
		std::string arg = argv[2];
		if(arg == "p") {
			show_flag = true;
		} else {
			printf("unknown flag: %s\n", arg.c_str());
			return -1;
		}
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
	printf("Width: %d, Height: %d, sizeof(unsigned char): %d, sizeImage: %d, sizeImageGrey: %d, sizeM: %d\n", width, height, sizeof(unsigned char), sizeImage, sizeImageGrey, sizeof(char)*9);
	
	cv_gray_image = (unsigned char *) malloc (sizeImageGrey);
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
	err = cudaMalloc((void **)&d_M, sizeof(char)*9);
	if(err != cudaSuccess){
        	printf("Error reservando memoria para d_M\n");
	 	exit(-1);
	}
	err = cudaMalloc((void **)&d_image_Sobel, sizeImageGrey);
	if(err != cudaSuccess){
        	printf("Error reservando memoria para d_image_Sobel\n");
	 	exit(-1);
	}

	// Start conversion from Host
	start_host = clock();
	image_sobel = (unsigned char *) malloc (sizeImageGrey);
	cvtColor(image, image_gray_host, CV_BGR2GRAY);
	cv_gray_image = image_gray_host.data;
	h_sobelFilter(cv_gray_image, image_sobel, width, height, 3, h_M);
	end_host = clock();
	// End Host conversion
	image_out_host.create(height, width, CV_8UC1);
	image_out_host.data = image_sobel;
	imwrite("image_out_host.jpg", image_out_host);
	time_used_host = ((double) (end_host - start_host)) /CLOCKS_PER_SEC;
	printf("%.10f|", time_used_host);

	// Start conversion with OpenCV
	start_opencv = clock();
	cvtColor(image, image_gray_opencv, CV_BGR2GRAY);
	Sobel(image_gray_opencv, image_out_opencv, CV_8UC1, 1, 0, 3); //src, dst, format(color depth), gradientx, gradienty, M size
	convertScaleAbs(image_out_opencv, abs_image_out_opencv); //src, dst -> convert to 8 bit, normalize
	end_opencv = clock();
	imwrite("image_out_opencv.jpg", abs_image_out_opencv);
	time_used_opencv = ((double) (end_opencv - start_opencv)) /CLOCKS_PER_SEC;
	printf("%.10f|", time_used_opencv);
	printf("%.10f|", time_used_host/time_used_opencv);
	//End OpenCV conversion

	// Start conversion with OpenCVCuda
	gpu::setDevice(0);
	start_opencv_gpu = clock();
	src.upload(image);
	gpu::cvtColor(src, gray, CV_BGR2GRAY);
	gpu::Sobel(gray, dst, CV_8UC1, 1, 0, 3);
	dst.download(image_out_opencv_gpu);
	end_opencv_gpu = clock();
	imwrite("image_out_opencv_gpu.jpg", image_out_opencv_gpu);
	time_used_opencv_gpu = ((double) (end_opencv_gpu - start_opencv_gpu)) /CLOCKS_PER_SEC;
	printf("%.10f|", time_used_opencv_gpu);
	printf("%.10f|", time_used_opencv/time_used_opencv_gpu);
	// End OpenCVCuda conversion

	// Start conversion with cuda	
	start_cuda = clock();
	err = cudaMemcpy(d_ImageData, h_ImageData, sizeImage, cudaMemcpyHostToDevice);
	if(err != cudaSuccess){
        	printf("Error copiando los datos de h_ImageData a d_ImageData\n");
	 	exit(-1);
	}
	err = cudaMemcpy(d_M, h_M, sizeof(char)*9, cudaMemcpyHostToDevice);
	if(err != cudaSuccess){
        	printf("Error copiando los datos de h_M a d_M\n");
	 	exit(-1);
	}
	
	int blockSize = 32;
	dim3 dimBlock(blockSize, blockSize, 1);
	dim3 dimGrid(ceil(width/float(blockSize)), ceil(width/float(blockSize)), 1);
	rgb2gray<<<dimGrid, dimBlock>>>(d_ImageData, d_ImageOut, width, height);
	cudaDeviceSynchronize();
	d_sobelFilter<<<dimGrid, dimBlock>>>(d_ImageOut, d_image_Sobel, width, height, 3, d_M);
	cudaDeviceSynchronize();
		
	err = cudaMemcpy(h_ImageOut, d_image_Sobel, sizeImageGrey, cudaMemcpyDeviceToHost);
	if(err != cudaSuccess){
        	printf("Error copiando los datos de d_image_Sobel a h_ImageOut\n");
	 	exit(-1);
	}
	end_cuda = clock();
	// End CUDA conversion
	time_used_cuda = ((double) (end_cuda - start_cuda)) /CLOCKS_PER_SEC;
	//printf("Tiempo algoritmo en CUDA: %.10f\n", time_used_cuda);
	printf("%.10f|", time_used_cuda);

	image_out_cuda.create(height, width, CV_8UC1);
	image_out_cuda.data = h_ImageOut;
	imwrite("image_out_cuda.jpg", image_out_cuda);

	printf("%.10f|", time_used_opencv/time_used_cuda);
	printf("%.10f\n", time_used_opencv_gpu/time_used_cuda);
		
	//printf("Done\n\n");
	//showImage(image, "Image In");
	if(show_flag) {
		showImage(image_out_host, "Image out Host");
		showImage(abs_image_out_opencv, "Image out OpenCV");
		showImage(image_out_opencv_gpu, "Image out OpenCVCuda");
		showImage(image_out_cuda, "Image out CUDA");
		waitKey(0);
	}
	free(h_ImageOut); free(image_sobel); cudaFree(d_ImageData); cudaFree(d_ImageOut); cudaFree(d_image_Sobel);
	return 0;
}
