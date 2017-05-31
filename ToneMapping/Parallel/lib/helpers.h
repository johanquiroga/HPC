#ifndef HPC_HELPERS_H
#define HPC_HELPERS_H

#include <iostream>
#include <dirent.h>

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

#endif //HPC_HELPERS_H
