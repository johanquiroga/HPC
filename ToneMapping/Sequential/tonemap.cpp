#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include <time.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "lib/helpers.h"

using namespace cv;

float task(std::string image_name, std::string images_path, std::string dst_path, std::string tmo,
           float f_stop, float gamma, float q, float k, float b, float ld_max)
{
	float *h_ImageData, *h_ImageOut;
	std::string image_out_name;
	Mat hdr, ldr, xyz_hdr;
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

//	cvtColor(hdr, xyz_hdr, CV_BGR2XYZ);

//	Mat y_channel( xyz_hdr.rows, xyz_hdr.cols, CV_32FC1 );
//	int from_to[] = { 1,0 };
//	mixChannels( &xyz_hdr, 1, &y_channel, 1, from_to, 1 );

	width = hdr.cols;
	height = hdr.rows;
	channels = hdr.channels();
	sizeImage = sizeof(float)*width*height*channels;

	h_ImageData = (float *)hdr.data;
	h_ImageOut = (float *) malloc (sizeImage);

	float elapsed_time = 0.0;

	if(tmo == "log") {
		elapsed_time = log_tonemap(h_ImageData, h_ImageOut, width, height, channels, k, q, sizeImage);
	} else if(tmo == "gamma") {
		elapsed_time = gamma_tonemap(h_ImageData, h_ImageOut, width, height, channels, f_stop, gamma, sizeImage);
	} else {
		elapsed_time = adaptive_log_tonemap(h_ImageData, h_ImageOut, width, height, channels, b, ld_max, sizeImage);
	}

//	Mat xyz_ldr(xyz_hdr.rows, xyz_hdr.cols, CV_32FC3);
//	Mat y_channel_out = Mat::zeros( xyz_hdr.rows, xyz_hdr.cols, CV_32FC1);
//	Mat y_channel_out( xyz_hdr.rows, xyz_hdr.cols, CV_32FC1, h_ImageOut);
//	y_channel_out.data = (unsigned char *)h_ImageOut;
//	showImage(y_channel_out, "y_channel_out");
//	waitKey(0);
	
//	Mat out[] = { xyz_hdr, y_channel_out};

//	int from_to_ldr[] = { 0,0, 3,1, 2,2 };
//	mixChannels( out, 2, &xyz_ldr, 1, from_to_ldr, 3 );

//	cvtColor(xyz_ldr, ldr, CV_XYZ2BGR);

//	printf("Image: %s\n", type2str(ldr.type()).c_str());

	ldr.create(height, width, CV_32FC3, h_ImageOut);
//	ldr.data = (unsigned char *)h_ImageOut;
	ldr.convertTo(ldr, CV_8UC3, 255);

	printf("Image: %s\n", type2str(ldr.type()).c_str());

	image_out_name = dst_path + "/" + change_image_extension(image_name);
	imwrite(image_out_name.c_str(), ldr);

	free(h_ImageOut);

	return elapsed_time;
}

void Usage()
{
	printf("Usage: ./tonemap <images_src> <results_dst> <output_separator> <TMO>(log/gamma/adap_log)\n");
	printf("If TMO = log, add: <k> <q>\n");
	printf("If TMO = gamma, add: <gamma> <f_stop>\n");
	printf("If TMO = adap_log, add: <b> <ld_max>\n");
}

int main(int argc, char** argv)
{
	float f_stop=0.0, gamma=0.0, q=0.0, k=0.0, b=1.0, ld_max=0.0;

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
		if(argc != 7) {
			Usage();
			return EXIT_FAILURE;
		}
		b = atof(argv[5]);
		ld_max = atof(argv[6]);
	} else {
		Usage();
		return EXIT_FAILURE;
	}

	clock_t start, end;
	double batch_time;
	std::vector<std::string> files;
	read_files(files, images_path);

	start = clock();
	while(!files.empty()) {
		float elapsed_time = 0.0;
		std::string file_name = files.back();
		elapsed_time = task(file_name,  images_path, dst_path, tmo, f_stop, gamma, q, k, b, ld_max);
		printTime(file_name, elapsed_time, separator);
		files.pop_back();
	}
	end = clock();
	batch_time = ((double)(end - start))/CLOCKS_PER_SEC;
	printTime("batch time", batch_time, separator);

	return EXIT_SUCCESS;
}
