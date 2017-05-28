#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include <time.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "lib/helpers.h"

using namespace cv;

float task(std::string image_name, float f_stop, float gamma, std::string images_path, std::string dst_path)
{
	float *h_ImageData, *h_ImageOut;
	std::string image_out_name;
	Mat hdr, ldr;
	Size imageSize;
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

	h_ImageData = (float *) malloc (sizeImage);
	h_ImageData = (float *)hdr.data;
	h_ImageOut = (float *) malloc (sizeImage);

	float elapsed_time = tonemap(h_ImageData, h_ImageOut, width, height, channels, f_stop, gamma, sizeImage);

	ldr.create(height, width, CV_32FC3);
	ldr.data = (unsigned char *)h_ImageOut;
	ldr.convertTo(ldr, CV_8UC3, 255);
	image_out_name = dst_path + "/" + change_image_extension(image_name);
	imwrite(image_out_name.c_str(), ldr);

	free(h_ImageOut);

	return elapsed_time;
}

int main(int argc, char** argv)
{
	if(argc !=6) {
		printf("Usage: ./tonemap <f_stop> <gamma> <image_src> <results_dst> <output_separator>");
		return EXIT_FAILURE;
	}

	clock_t start, end;
	double batch_time;

	float f_stop = atof(argv[1]);
	float gamma = atof(argv[2]);
	std::string images_path(argv[3]);
	std::string dst_path(argv[4]);
	std::string separator(argv[5]);

	std::vector<std::string> files;
	read_files(files, images_path);

	start = clock();
	while(!files.empty()) {
		float elapsed_time = 0.0;
		std::string file_name = files.bak();
		elapsed_time = task(file_name, f_stop, gamma, images_path, dst_path);
		printTime(file_name, elapsed_time, separator);
		files.pop_back();
	}
	end = clock();
	batch_time = ((double)(end - start))/CLOCKS_PER_SEC;
	printTime("batch time", batch_time, separator);

	return EXIT_SUCCESS;
}