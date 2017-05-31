#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <string>
#include <time.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "lib/helpers.h"
#include "lib/tonemap.cuh"

#define BLOCK_SIZE 32
#define BLUE 0
#define GREEN 1
#define RED 2

using namespace cv;

//void checkError(cudaError_t err) {
//	if(err!=cudaSuccess) {
//		printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
//		exit(EXIT_FAILURE);
//	}
//}

void my_abort(int err)
{
	std::cout << "Program FAILED\n";
    exit(err);
}

float task(std::string image_name, std::string images_path, std::string dst_path, std::string tmo, int blockSize,
           float f_stop, float gamma, float q, float k, float b)
{
	float *h_ImageData, *h_ImageOut;
	std::string image_out_name;
	Mat hdr, ldr;
	int width, height, channels, sizeImage;

	std::string path = images_path + "/" + image_name;

	hdr = imread(path.c_str(), -1);

	if(!hdr.data) {
		printf("No image Data \n");
		return EXIT_FAILURE;
//        return -1;
	}

	if(hdr.empty()) {
		printf("Couldn't find or open the image...\n");
		return EXIT_FAILURE;
//        return -1;
	}

	width = hdr.cols;
	height = hdr.rows;
	channels = hdr.channels();
	sizeImage = sizeof(float)*width*height*channels;

	h_ImageData = (float *)hdr.data;
	h_ImageOut = (float *) malloc (sizeImage);
	float elapsed_time = 0.0;

	if(tmo == "log") {
		elapsed_time = log_tonemap(h_ImageData, h_ImageOut, width, height, channels, k, q, blockSize, sizeImage);
	} else if(tmo == "gamma") {
		elapsed_time = gamma_tonemap(h_ImageData, h_ImageOut, width, height, channels, f_stop, gamma, blockSize,
		                             sizeImage);
	} else {
		elapsed_time = adaptive_log_tonemap(h_ImageData, h_ImageOut, width, height, channels, b, blockSize, sizeImage);
	}

	ldr.create(height, width, CV_32FC3);
	ldr.data = (unsigned char *)h_ImageOut;
	ldr.convertTo(ldr, CV_8UC3, 255);
	image_out_name = dst_path + "/" + change_image_extension(image_name);
	imwrite(image_out_name.c_str(), ldr);

	free(h_ImageOut);

	return elapsed_time;
}

void Usage()
{
	printf("Usage: ./test <images_src> <results_dst> <output_separator> <TMO>(log/gamma)\n");
	printf("If TMO = log, add: <k> <q>\n");
	printf("If TMO = gamma, add: <gamma> <f_stop>\n");
//	printf("If TMO = adap_log, add: <b>\n");
}

int main(int argc, char** argv)
{
	float f_stop=0.0, gamma=0.0, q=0.0, k=0.0, b=1.0;

	if(argc == 1 || argc < 5) {
		Usage();
		return EXIT_FAILURE;
	}
	std::string images_path(argv[1]);
	std::string dst_path(argv[2]);
	std::string separator(argv[3]);
	std::string tmo(argv[4]);

	if(tmo == "log") {
		if(argc != 7) {
			Usage();
			return EXIT_FAILURE;
		}
		k = atof(argv[5]);
		q = atof(argv[6]);
	} else if(tmo == "gamma") {
		if(argc != 7) {
			Usage();
			return EXIT_FAILURE;
		}
		gamma = atof(argv[5]);
		f_stop = atof(argv[6]);
	} else if(tmo == "adap_log") {
		if(argc != 6) {
			Usage();
			return EXIT_FAILURE;
		}
		b = atof(argv[5]);
	} else {
		Usage();
		return EXIT_FAILURE;
	}

	clock_t start, end;
	double batch_time;
	std::vector<std::string> files;
	read_files(files, images_path);
	int blockSize = BLOCK_SIZE;

	start = clock();
	while(!files.empty()) {
		float elapsed_time = 0.0;
		std::string file_name = files.back();
		elapsed_time = task(file_name, images_path, dst_path, tmo, blockSize, f_stop, gamma, q, k, b);
		printTime(file_name, elapsed_time, separator);
		files.pop_back();
	}
	end = clock();
	batch_time = ((double)(end - start))/CLOCKS_PER_SEC;
	printTime("batch time", batch_time, separator);

	return EXIT_SUCCESS;

/*	imageSize = gray_hdr.size();
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

	return 0;*/
}
