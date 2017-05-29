#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
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

__global__ void find_maximum_kernel(float *array, float *max, int *mutex, unsigned int n)
{
	unsigned int index = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int stride = gridDim.x*blockDim.x;
	unsigned int offset = 0;
//	const int size = blockSize;

	extern	__shared__ float cache[];


	float temp = -1.0;
	while(index + offset < n){
		temp = fmaxf(temp, array[index + offset]);

		offset += stride;
	}

	cache[threadIdx.x] = temp;

	__syncthreads();


	// reduction
	unsigned int i = blockDim.x/2;
	while(i != 0){
		if(threadIdx.x < i){
			cache[threadIdx.x] = fmaxf(cache[threadIdx.x], cache[threadIdx.x + i]);
		}

		__syncthreads();
		i /= 2;
	}

	if(threadIdx.x == 0){
		while(atomicCAS(mutex,0,1) != 0);  //lock
		*max = fmaxf(*max, cache[0]);
		atomicExch(mutex, 0);  //unlock
	}
}

__device__ float gamma_correction(float f_stop, float gamma, float val)
{
	return powf((val*powf(2,f_stop)),(1.0/gamma));
}

__device__ float logarithmic_mapping(float lw_max, float ld_max, float lw, float b)
{
	float ld = (ld_max)/(100*log10f(1+lw_max)*((logf(1+lw))/(logf(2+8*powf((lw/lw_max),(logf(b)/logf(0.5)))))));
	return ld;
}

__global__ void tonemap(float* imageIn, float* imageOut, int width, int height, int channels, int depth, float f_stop,
						float gamma, float* max, float b)
{
	int Row = blockDim.y * blockIdx.y + threadIdx.y;
	int Col = blockDim.x * blockIdx.x + threadIdx.x;

	if(Row < height && Col < width) {
		imageOut[Row*width+Col] =  logarithmic_mapping(*max, 100.0, imageIn[Row*width+Col], b);
//		imageOut[(Row*width+Col)*3+BLUE] = gamma_correction(f_stop, gamma, imageIn[(Row*width+Col)*3+BLUE]);
//		imageOut[(Row*width+Col)*3+GREEN] = gamma_correction(f_stop, gamma, imageIn[(Row*width+Col)*3+GREEN]);
//		imageOut[(Row*width+Col)*3+RED] = gamma_correction(f_stop, gamma, imageIn[(Row*width+Col)*3+RED]);
	}
}

void showImage(Mat &image, const char *window) {
	namedWindow(window, CV_WINDOW_NORMAL);
	imshow(window, image);
}

int main(int argc, char** argv)
{
	char* image_name = argv[1];
    char* image_out_name = argv[3];
	float *h_ImageData, *d_ImageData, *d_ImageOut, *h_ImageOut, *h_max, *d_max;
	int *d_mutex, N;
	Mat hdr, ldr, gray_hdr;
	Size imageSize;
	int width, height, channels, sizeImage;
	float f_stop=0.0, gamma=0.0, b;
	int show_flag = 0;
//	std::vector<Mat>images;

//	printf("%s\n", image_name);
	hdr = imread(image_name, -1);
	if(argc !=4 || !hdr.data) {
		printf("No image Data \n");
		//printf("Usage: ./test <file_path> <f_stop> <gamma> <show_flag> <output_file_path>");
		printf("Usage: ./test <file_path> <b> <output_file_path>");
		return -1;
	}

	b = atof(argv[2]);
//	f_stop = atof(argv[2]);
//	gamma = atof(argv[3]);
//	show_flag = atoi(argv[4]);

	if(hdr.empty()) {
		printf("Couldn't find or open the image...\n");
		return -1;
	}

	cvtColor(hdr, gray_hdr, CV_BGR2GRAY);
	normalize(gray_hdr, gray_hdr, 0.0, 1.0, NORM_MINMAX, CV_32FC1);

	imageSize = gray_hdr.size();
	width = imageSize.width;
	height = imageSize.height;
	channels = gray_hdr.channels();
	N = width*height*channels;
	sizeImage = sizeof(float)*width*height*channels;

	//printf("Width: %d\nHeight: %d\n", width, height);
	std::string ty =  type2str( gray_hdr.type() );
	printf("Image: %s %dx%d \n", ty.c_str(), hdr.cols, hdr.rows );

	//printf("Channels: %d\nDepth: %d\n", hdr.channels(), hdr.depth());

	//h_ImageData = (float *) malloc (sizeImage);
	h_ImageData = (float *)gray_hdr.data;
	h_max = (float*)malloc(sizeof(float));
	h_ImageOut = (float *) malloc (sizeImage);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float milliseconds = 0;

	checkError(cudaMalloc((void **)&d_ImageData, sizeImage));
	checkError(cudaMalloc((void **)&d_ImageOut, sizeImage));
	checkError(cudaMalloc((void**)&d_max, sizeof(float)));
	checkError(cudaMalloc((void**)&d_mutex, sizeof(int)));
	checkError(cudaMemset(d_max, 0, sizeof(float)));
	checkError(cudaMemset(d_mutex, 0, sizeof(int)));
	checkError(cudaMemcpy(d_ImageData, h_ImageData, sizeImage, cudaMemcpyHostToDevice));

	int blockSize = 256;
	dim3 dimBlock(blockSize, 1, 1);
	dim3 dimGrid(ceil(width/float(blockSize)), 1, 1);
	find_maximum_kernel<<< dimGrid, dimBlock, sizeof(float)*blockSize >>>(d_ImageData, d_max, d_mutex, N);
	cudaDeviceSynchronize();
	//cudaMemcpy(h_max, d_max, sizeof(float), cudaMemcpyDeviceToHost);

	blockSize = 32;
	dim3 dimBlock2(blockSize, blockSize, 1);
	dim3 dimGrid2(ceil(width/float(blockSize)), ceil(height/float(blockSize)), 1);
	cudaEventRecord(start);
	tonemap<<<dimGrid2, dimBlock2>>>(d_ImageData, d_ImageOut, width, height, channels, 32, f_stop, gamma, d_max, b);
	cudaEventRecord(stop);
	cudaDeviceSynchronize();
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("%s|%.10f\n", image_name, milliseconds/1000.0);

	checkError(cudaMemcpy(h_ImageOut, d_ImageOut, sizeImage, cudaMemcpyDeviceToHost));

	ldr.create(height, width, CV_32FC1);
	ldr.data = (unsigned char *)h_ImageOut;
	ldr.convertTo(ldr, CV_8UC1, 1);
	//cvtColor(ldr, ldr, CV_GRAY2BGR);
	imwrite(image_out_name, ldr);

    ty =  type2str( ldr.type() );
    printf("Image result: %s %dx%d \n", ty.c_str(), ldr.cols, ldr.rows );

	if(show_flag) {
		showImage(ldr, "Image out LDR");
		waitKey(0);
	}

	free(h_max); free(h_ImageOut); cudaFree(d_ImageData); cudaFree(d_ImageOut); cudaFree(d_max); cudaFree(d_mutex);

	return 0;
}
