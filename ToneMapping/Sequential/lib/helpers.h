#ifndef HPC_HELPERS_H
#define HPC_HELPERS_H

//#include <iostream>
//#include <stdio.h>
//#include <stdlib.h>
//#include <string>
#include <dirent.h>
//#include <time.h>
#include <math.h>

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

void showImage(Mat &image, const char *window)
{
    namedWindow(window, CV_WINDOW_NORMAL);
    imshow(window, image);
}

std::string change_image_extension(std::string target)
{
	int pos = target.rfind(".");
	target.replace(pos, 4, ".png");
	return target;
}

void read_files(std::vector<std::string> &files, std::string path)
{
	DIR *dir;
	struct dirent *ent;
	if((dir = opendir (path.c_str())) != NULL) {
		/* print all the files and directories within directory */
		while ((ent = readdir (dir)) != NULL) {
			//files += "\t";
			std::string tmp = ent->d_name;
			if(tmp != "." && tmp != "..") {
				files.push_back(tmp);
			}
		}
		closedir (dir);
	} else {
		/* could not open directory */
		perror ("");
		exit(EXIT_FAILURE);
	}
}

void printTime(std::string file_name, float time, std::string separator)
{
	printf("%s%s%.10f\n", file_name.c_str(), separator.c_str(), time);

}

float gamma_correction(float f_stop, float gamma, float val)
{
	return powf((val*powf(2, f_stop)), (1.0/gamma));
}

float gamma_tonemap(float* h_ImageData, float* h_ImageOut, int width, int height, int channels, float f_stop, float gamma, int sizeImage)
{
	clock_t start, end;
	double elapsed_time;

	start = clock();
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			for (int k = 0; k < channels; k++) {
				h_ImageOut[(i * width + j) * channels + k] = gamma_correction(f_stop, gamma, h_ImageData[(i * width + j) * channels + k]);
			}
		}
		
	}
	end = clock();

	elapsed_time = ((double)(end - start))/CLOCKS_PER_SEC;
	return elapsed_time;
}

float logarithmic_mapping(float k, float q, float val_pixel, float maxLum)
{
	return (log10(1 + q * val_pixel))/(log10(1 + k * maxLum));
}

float log_tonemap(float* h_ImageData, float* h_ImageOut, int width, int height, int channels, float k, float q, int sizeImage)
{
	clock_t start, end;
	double elapsed_time;
	int N = sizeImage/sizeof(float);

	start = clock();

	float* h_max = -1.0;
	for(unsigned int i=0;i<N;i++){
		if(h_ImageData[i] > *h_max){
			*h_max = h_ImageData[i];
		}
	}

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			for (int k = 0; k < channels; k++) {
				h_ImageOut[(i * width + j) * channels + k] = logarithmic_mapping(k, q, h_ImageData[(i * width + j) * channels + k], *h_max);
			}
		}

	}

	end = clock();

	elapsed_time = ((double)(end - start))/CLOCKS_PER_SEC;
	return elapsed_time;
}

float adaptive_logarithmic_mapping(float lw_max, float ld_max, float lw, float b)
{
	float ld = ((ld_max)/(100*log10f(1+lw_max)))*((logf(1+lw))/(logf(2+8*powf((lw/lw_max),(logf(b)/logf(0.5))))));
	return ld;
}

float adaptive_log_tonemap(float* h_ImageData, float* h_ImageOut, int width, int height, int channels, float b, float ld_max, int sizeImage)
{
	clock_t start, end;
	double elapsed_time;
	int N = sizeImage/sizeof(float);

	start = clock();

	float* h_max = -1.0;
	for(unsigned int i=0;i<N;i++){
		if(h_ImageData[i] > *h_max){
			*h_max = h_ImageData[i];
		}
	}

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			for (int k = 0; k < channels; k++) {
				h_ImageOut[(i * width + j) * channels + k] = adaptive_logarithmic_mapping(*h_max, ld_max, h_ImageData[(i * width + j) * channels + k], b);
			}
		}

	}

	end = clock();

	elapsed_time = ((double)(end - start))/CLOCKS_PER_SEC;
	return elapsed_time;
}

#endif //HPC_HELPERS_H
