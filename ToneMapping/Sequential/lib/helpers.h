#ifndef HPC_HELPERS_H
#define HPC_HELPERS_H

#include <dirent.h>
#include <math.h>

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
	return powf((val*powf(2.0, f_stop)), (1.0/gamma));
}

float rgb2Lum(float B, float G, float R)
{
	return B * 0.0722 + G * 0.7152 + R * 0.2126;
}

float find_max(float* array, int N)
{
	float h_max = -1.0;

	for(unsigned int i=0;i<N;i+=3) {
		float L = rgb2Lum(h_ImageData[i], h_ImageData[i+1], h_ImageData[i+2]);
		if(L > h_max) {
			h_max = L;
		}
	}

	return h_max;
}

float gamma_tonemap(float* h_ImageData, float* h_ImageOut, int width, int height, int channels, float f_stop, float gamma, int sizeImage)
{
	clock_t start, end;
	double elapsed_time;

	start = clock();
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			float B, G, R, L, nL, scale;
			B = h_ImageData[(i * width + j) * 3 + BLUE];
			G = h_ImageData[(i * width + j) * 3 + GREEN];
			R = h_ImageData[(i * width + j) * 3 + RED];

			L = rgb2Lum(B, G, R);
			nL = gamma_correction(f_stop, gamma, L);
			scale = nL/L;

			h_ImageOut[(i * width + j) * 3 + BLUE] = B * scale;
			h_ImageOut[(i * width + j) * 3 + GREEN] = G * scale;
			h_ImageOut[(i * width + j) * 3 + RED] = R * scale;
		}
		
	}
	end = clock();

	elapsed_time = ((double)(end - start))/CLOCKS_PER_SEC;
	return elapsed_time;
}

float logarithmic_mapping(float k, float q, float val_pixel, float maxLum)
{
	return (log10f(1.0 + q * val_pixel))/(log10f(1.0 + k * maxLum));
}

float log_tonemap(float* h_ImageData, float* h_ImageOut, int width, int height, int channels, float k, float q, int sizeImage)
{
	clock_t start, end;
	double elapsed_time;
	int N = sizeImage/sizeof(float);

	start = clock();

	float h_max = find_max(h_ImageData, N);

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			float B, G, R, L, nL, scale;
			B = h_ImageData[(i * width + j) * 3 + BLUE];
			G = h_ImageData[(i * width + j) * 3 + GREEN];
			R = h_ImageData[(i * width + j) * 3 + RED];

			L = rgb2Lum(B, G, R);
			nL = logarithmic_mapping(k, q, L, h_max);
			scale = nL/L;

			h_ImageOut[(i * width + j) * 3 + BLUE] = B * scale;
			h_ImageOut[(i * width + j) * 3 + GREEN] = G * scale;
			h_ImageOut[(i * width + j) * 3 + RED] = R * scale;
		}

	}

	end = clock();

	elapsed_time = ((double)(end - start))/CLOCKS_PER_SEC;
	return elapsed_time;
}

float adaptive_logarithmic_mapping(float lw_max, float ld_max, float lw, float b)
{
	float ld = ((ld_max)/(100.0*log10f(1.0+lw_max)))*((logf(1.0+lw))/(logf(2.0+8.0*powf((lw/lw_max),(logf(b)/logf(0.5))))));
	return ld;
}

float adaptive_log_tonemap(float* h_ImageData, float* h_ImageOut, int width, int height, int channels, float b, float ld_max, int sizeImage)
{
	clock_t start, end;
	double elapsed_time;
	int N = sizeImage/sizeof(float);

	start = clock();

	float h_max = find_max(h_ImageData, N);

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			float B, G, R, L, nL, scale;
			B = h_ImageData[(i * width + j) * 3 + BLUE];
			G = h_ImageData[(i * width + j) * 3 + GREEN];
			R = h_ImageData[(i * width + j) * 3 + RED];

			L = rgb2Lum(B, G, R);
			nL = adaptive_logarithmic_mapping(h_max, ld_max, L, b);
			scale = nL/L;

			h_ImageOut[(i * width + j) * 3 + BLUE] = B * scale;
			h_ImageOut[(i * width + j) * 3 + GREEN] = G * scale;
			h_ImageOut[(i * width + j) * 3 + RED] = R * scale;
		}

	}

	end = clock();

	elapsed_time = ((double)(end - start))/CLOCKS_PER_SEC;
	return elapsed_time;
}

#endif //HPC_HELPERS_H
