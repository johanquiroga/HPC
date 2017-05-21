#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/photo.hpp>
#include <vector>
#include <string>

#define BLUE 0
#define GREEN 1
#define RED 2

using namespace cv;

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

void checkError(cudaError_t err) {
	if(err!=cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}
}

///***********************************
//I = 1/61*(Red*20 + Green*40 + Blue)
//***********************************/
//__device__ float compute_intensity(float Blue, float Green, float Red)
//{
//	return ((1.0/61.0) * (Blue + (Green * 40) + (Red * 20)));
//}
//
///**********************
//Red/I, Green/I, Blue/I
//**********************/
//__device__ float compute_chrominance(float Channel, float I)
//{
//	return Channel/I;
//}
//
///***********
//L = log2(I)
//***********/
//__device__ float compute_intensity_log(float I)
//{
//	return log2f(I);
//}
//
///*********
//B = bf(L)
//*********/
//__device__ float apply_billateral_filter(float L)
//{
//	//@TODO
//}
//
///*******
//D = L-B
//*******/
//__device__ float compute_detail_layer(float L, float B)
//{
//	return L-B;
//}
//
///********************
//nB = (B-offset)*scale
//********************/
//__device__ float apply_offset_scale_base(float B, int offset, int scale)
//{
//	return (B-offset)*scale;
//}
//
///*************
//O = exp(nB+D)
//*************/
//__device__ float reconstruct_log_intensity(float nB, float D)
//{
//	return expf(nB+D);
//}
//
///****************************
//nR,nG,nB = O*(R/I, G/I, B/I)
//****************************/
//__device__ float put_colors_back(float Channel, float I, float O)
//{
//	return O*(compute_chrominance(Channel, I));
//}
//
///*********************
//
//*********************/
//__global__ void tonemap(float* imageIn, float* imageOut, int width, int height, int channels, int depth)
//{
//	//Each thread reads each pixel and puts it through the pipeline
//}
//

// Iout(x,y)=(Iin(x,y)⋅2ᶺf)ᶺ(1/g)

__device__ float gamma_correction(float f_stop, float gamma, float val)
{
	return powf((val*powf(2,f_stop)),(1.0/gamma));
}

__global__ void tonemap(float* imageIn, float* imageOut, int width, int height, int channels, int depth, float f_stop,
						float gamma)
{
	int Row = blockDim.y * blockIdx.y + threadIdx.y;
	int Col = blockDim.x * blockIdx.x + threadIdx.x;

	if(Row < height && Col < width) {
		imageOut[(Row*width+Col)*3+BLUE] = gamma_correction(f_stop, gamma, imageIn[(Row*width+Col)*3+BLUE]);
		imageOut[(Row*width+Col)*3+GREEN] = gamma_correction(f_stop, gamma, imageIn[(Row*width+Col)*3+GREEN]);
		imageOut[(Row*width+Col)*3+RED] = gamma_correction(f_stop, gamma, imageIn[(Row*width+Col)*3+RED]);
	}
}

 void showImage(Mat &image, const char *window) {
	namedWindow(window, CV_WINDOW_NORMAL);
	imshow(window, image);
 }

int main(int argc, char** argv)
{
	char* image_name = argv[1];
	float *h_ImageData, *d_ImageData, *d_ImageOut, *h_ImageOut;
	Mat hdr, ldr;
	Size imageSize;
	int width, height, channels, sizeImage;
	float f_stop=0.0, gamma=0.0;
	int show_flag;
//	std::vector<Mat>images;

	printf("%s\n", image_name);
	hdr = imread(image_name, -1);
	if(argc !=5 || !hdr.data) {
		printf("No image Data \n");
		printf("Usage: ./test <file_path> <f_stop> <gamma> <show_flag>");
		return -1;
	}

	f_stop = atof(argv[2]);
	gamma = atof(argv[3]);
	show_flag = atoi(argv[4]);

	if(hdr.empty()) {
		printf("Couldn't find or open the image...\n");
		return -1;
	}
	imageSize = hdr.size();
	width = imageSize.width;
	height = imageSize.height;
	channels = hdr.channels();
	sizeImage = sizeof(float)*width*height*channels;

	//printf("Width: %d\nHeight: %d\n", width, height);
	//std::string ty =  type2str( hdr.type() );
	//printf("Image: %s %dx%d \n", ty.c_str(), hdr.cols, hdr.rows );

	//printf("Channels: %d\nDepth: %d\n", hdr.channels(), hdr.depth());

	h_ImageData = (float *) malloc (sizeImage);
	h_ImageData = (float *)hdr.data;
	h_ImageOut = (float *) malloc (sizeImage);

	checkError(cudaMalloc((void **)&d_ImageData, sizeImage));
	checkError(cudaMalloc((void **)&d_ImageOut, sizeImage));
	checkError(cudaMemcpy(d_ImageData, h_ImageData, sizeImage, cudaMemcpyHostToDevice));

	int blockSize = 32;
	dim3 dimBlock(blockSize, blockSize, 1);
	dim3 dimGrid(ceil(width/float(blockSize)), ceil(height/float(blockSize)), 1);
	tonemap<<<dimGrid, dimBlock>>>(d_ImageData, d_ImageOut, width, height, channels, 32, f_stop, gamma);
	cudaDeviceSynchronize();

	checkError(cudaMemcpy(h_ImageOut, d_ImageOut, sizeImage, cudaMemcpyDeviceToHost));

	ldr.create(height, width, CV_8UC3);
	ldr.data = (unsigned char *)h_ImageOut;
//	ldr.convertTo(ldr, CV_8UC3, 255);
	imwrite("ldr.png", ldr*255);

	if(show_flag) {
		showImage(ldr, "Image out LDR");
		waitKey(0);
	}

	free(h_ImageOut); cudaFree(d_ImageData); cudaFree(d_ImageOut);

	return 0;
}
