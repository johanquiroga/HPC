#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string>
#include <vector>
#include <mpi.h>
//#include <dirent.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "lib/helpers.h"
#include "lib/tone_mapping.h"

#define BLOCK_SIZE 32
#define FROM_MASTER 1
#define FROM_WORKER 2

// Error handling macros
#define MPI_CHECK(call) \
	if((call) != MPI_SUCCESS) { \
		std::cerr << "MPI error calling \""#call"\"\n"; \
		my_abort(-1); }

using namespace cv;

//void showImage(Mat &image, const char *window) {
//    namedWindow(window, CV_WINDOW_NORMAL);
//    imshow(window, image);
//}

//std::string change_image_extension(std::string target)
//{
//    int pos = target.rfind(".");
//    target.replace(pos, 4, ".png");
//    return target;
//}
//
//void read_files(std::vector<std::string> &files, std::string path)
//{
//    DIR *dir;
//    struct dirent *ent;
//    if((dir = opendir (path.c_str())) != NULL) {
//        /* print all the files and directories within directory */
//        while ((ent = readdir (dir)) != NULL) {
//            //files += "\t";
//            std::string tmp = ent->d_name;
//            if(tmp != "." && tmp != "..") {
//                files.push_back(tmp);
//            }
//        }
//        closedir (dir);
//    } else {
//        /* could not open directory */
//        perror ("");
//        exit(EXIT_FAILURE);
//    }
//}

// Shut down MPI cleanly if something goes wrong
void my_abort(int err)
{
	std::cout << "Program FAILED\n";
	MPI_Abort(MPI_COMM_WORLD, err);
//    exit(err);
}

void task(std::string image_name, float f_stop, float gamma, int block_size, std::string images_path, std::string dst_path)
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
		my_abort(EXIT_FAILURE);
//        return -1;
	}

	if(hdr.empty()) {
		printf("Couldn't find or open the image...\n");
		my_abort(EXIT_FAILURE);
//        return -1;
	}

	width = hdr.cols;
	height = hdr.rows;
	channels = hdr.channels();
	sizeImage = sizeof(float)*width*height*channels;

	h_ImageData = (float *) malloc (sizeImage);
	h_ImageData = (float *)hdr.data;
	h_ImageOut = (float *) malloc (sizeImage);

	tonemap(h_ImageData, h_ImageOut, width, height, channels, f_stop, gamma, block_size, sizeImage);

	ldr.create(height, width, CV_32FC3);
	ldr.data = (unsigned char *)h_ImageOut;
	ldr.convertTo(ldr, CV_8UC3, 255);
	image_out_name = dst_path + "/" + change_image_extension(image_name);
	imwrite(image_out_name.c_str(), ldr);

	free(h_ImageOut);
}

int main(int argc, char** argv)
{
	int numtasks, taskid;
	MPI_CHECK(MPI_Init(&argc, &argv));
	MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &taskid));
	MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &numtasks));

	if(numtasks < 2) {
		printf("Need at least two tasks...\n");
		my_abort(EXIT_FAILURE);
	}

	if(argc !=5) {
//        printf("No image Data \n");
		printf("Usage: ./tonemapping <f_stop> <gamma> <image_src> <results_dst>");
		my_abort(EXIT_FAILURE);
//        return -1;
	}

	int nworkers = numtasks - 1;
//    float f_stop = 0.0;
//    float gamma = 0.0;

	//char* image_name = argv[1];
	if(taskid == 0) {
		int block_size = BLOCK_SIZE;
		float f_stop = atof(argv[1]);
		float gamma = atof(argv[2]);
		std::string images_path(argv[3]);
		std::string dst_path(argv[4]);

		std::vector<std::string> files;
		read_files(files, images_path);
		std::cout << "f_stop: " << f_stop << std::endl;
		std::cout << "gamma: " << gamma << std::endl;
		std::cout << "images_path: " << images_path << std::endl;
		std::cout << "dst_path: " << dst_path << std::endl;
		std::cout << "size: " << files.size() << std::endl;

		if (files.size() >= nworkers) {
			int j = 1;
			std::cout << "more files than workers" << std::endl;
			while (j <= nworkers) {
				// send
				std::string op("work");
				std::string tmp = files.back();
//				std::cout << "worker: " << j << std::endl;
//				std::cout << "File: " << tmp << std::endl;
				MPI_CHECK(MPI_Send(op.c_str(), op.length()+1, MPI_CHAR, j, FROM_MASTER, MPI_COMM_WORLD));
//				std::cout << "op message sended" << std::endl;
				MPI_CHECK(MPI_Send(images_path.c_str(), images_path.size()+1, MPI_CHAR, j, FROM_MASTER, MPI_COMM_WORLD));
				MPI_CHECK(MPI_Send(dst_path.c_str(), dst_path.size()+1, MPI_CHAR, j, FROM_MASTER, MPI_COMM_WORLD));
				MPI_CHECK(MPI_Send(tmp.c_str(), tmp.size()+1, MPI_CHAR, j, FROM_MASTER, MPI_COMM_WORLD));
				MPI_CHECK(MPI_Send(&f_stop, 1, MPI_FLOAT, j, FROM_MASTER, MPI_COMM_WORLD));
				MPI_CHECK(MPI_Send(&gamma, 1, MPI_FLOAT, j, FROM_MASTER, MPI_COMM_WORLD));
				MPI_CHECK(MPI_Send(&block_size, 1, MPI_INT, j, FROM_MASTER, MPI_COMM_WORLD));
				files.pop_back();
				j++;
			}
			j = 1;
			int count = nworkers;
			while (j <= count) {
				int tmpid;
				if(j > nworkers) {
					tmpid = j - nworkers;
				} else {
					tmpid = j;
				}
				// receive
				int workerid;
				MPI_CHECK(MPI_Recv(&workerid, 1, MPI_INT, MPI_ANY_SOURCE, FROM_WORKER, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
				std::cout << "Finished: worker: " << tmpid << " workerid: " << workerid << std::endl;
				tmpid = workerid;
				if (!files.empty()) {
					// send
					std::string op = "work";
					std::string tmp = files.back();
//					std::cout << "worker: " << tmpid << std::endl;
//					std::cout << "File: " << tmp << std::endl;
					MPI_CHECK(MPI_Send(op.c_str(), op.size()+1, MPI_CHAR, tmpid, FROM_MASTER, MPI_COMM_WORLD));
					MPI_CHECK(MPI_Send(images_path.c_str(), images_path.size()+1, MPI_CHAR, tmpid, FROM_MASTER, MPI_COMM_WORLD));
					MPI_CHECK(MPI_Send(dst_path.c_str(), dst_path.size()+1, MPI_CHAR, tmpid, FROM_MASTER, MPI_COMM_WORLD));
					MPI_CHECK(MPI_Send(tmp.c_str(), tmp.size()+1, MPI_CHAR, tmpid, FROM_MASTER, MPI_COMM_WORLD));
					MPI_CHECK(MPI_Send(&f_stop, 1, MPI_FLOAT, tmpid, FROM_MASTER, MPI_COMM_WORLD));
					MPI_CHECK(MPI_Send(&gamma, 1, MPI_FLOAT, tmpid, FROM_MASTER, MPI_COMM_WORLD));
					MPI_CHECK(MPI_Send(&block_size, 1, MPI_INT, tmpid, FROM_MASTER, MPI_COMM_WORLD));
					files.pop_back();
					count++;
				} else {
					// finish
					std::string op = "finish";
					std::cout << "worker: " << tmpid << std::endl;
					std::cout << "Terminate" << std::endl;
					MPI_CHECK(MPI_Send(op.c_str(), op.size()+1, MPI_CHAR, tmpid, FROM_MASTER, MPI_COMM_WORLD));
				}
				j++;
			}
		} else {
			int i = 0;
			std::cout << "more workers than files"  << std::endl;
			while (i < files.size()) {
				// send to i+1
				std::string op = "work";
				std::string tmp = files[i];
//				std::cout << "worker: " << i << std::endl;
//				std::cout << "File: " << tmp << std::endl;
				MPI_CHECK(MPI_Send(op.c_str(), op.size()+1, MPI_CHAR, i+1, FROM_MASTER, MPI_COMM_WORLD));
				MPI_CHECK(MPI_Send(images_path.c_str(), images_path.size()+1, MPI_CHAR, i+1, FROM_MASTER, MPI_COMM_WORLD));
				MPI_CHECK(MPI_Send(dst_path.c_str(), dst_path.size()+1, MPI_CHAR, i+1, FROM_MASTER, MPI_COMM_WORLD));
				MPI_CHECK(MPI_Send(tmp.c_str(), tmp.size()+1, MPI_CHAR, i+1, FROM_MASTER, MPI_COMM_WORLD));
				MPI_CHECK(MPI_Send(&f_stop, 1, MPI_FLOAT, i+1, FROM_MASTER, MPI_COMM_WORLD));
				MPI_CHECK(MPI_Send(&gamma, 1, MPI_FLOAT, i+1, FROM_MASTER, MPI_COMM_WORLD));
				MPI_CHECK(MPI_Send(&block_size, 1, MPI_INT, i+1, FROM_MASTER, MPI_COMM_WORLD));
				i++;
			}
			int j = 1;
			while (j <= i) {
				// receive
				int workerid;
				MPI_CHECK(MPI_Recv(&workerid, 1, MPI_INT, j, FROM_WORKER, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
				// send finish
				std::string op = "finish";
				std::cout << "worker: " << j << "workerid: " << workerid << std::endl;
				std::cout << "Terminate" << std::endl;
				MPI_CHECK(MPI_Send(op.c_str(), op.size()+1, MPI_CHAR, j, FROM_MASTER, MPI_COMM_WORLD));
				j++;
			}
		}

		/*while(true) {
			if(!files.empty()) {
				std::string op = "work";
				if(j<=nworkers) {
					std::cout << "Sending \"" << files.back() << "\" to \"" << j << "\"" << std::endl;
					std::string tmp = files.back();
					MPI_Send(op.c_str(), op.size(), MPI_CHAR, j, FROM_MASTER, MPI_COMM_WORLD);
					MPI_Send(images_path.c_str(), images_path.size(), MPI_CHAR, j, FROM_MASTER, MPI_COMM_WORLD);
					MPI_Send(dst_path.c_str(), dst_path.size(), MPI_CHAR, j, FROM_MASTER, MPI_COMM_WORLD);
					MPI_Send(tmp.c_str(), tmp.size(), MPI_CHAR, j, FROM_MASTER, MPI_COMM_WORLD);
					MPI_Send(&f_stop, 1, MPI_FLOAT, j, FROM_MASTER, MPI_COMM_WORLD);
					MPI_Send(&gamma, 1, MPI_FLOAT, j, FROM_MASTER, MPI_COMM_WORLD);
					MPI_Send(&block_size, 1, MPI_INT, j, FROM_MASTER, MPI_COMM_WORLD);
					files.pop_back();
					// send op = "work";
					// MPI_Send(op.c_str(), op.length(), MPI_CHAR, j, FROM_MASTER, MPI_COMM_WORLD);
					// send file name: file.back();
					// send f_stop
					// send gamma
					// send block_size
					// files.pop_back();
					j++;
				}
				int workerid;
				MPI_Recv(&workerid, 1, MPI_INT, 0, FROM_WORKER, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

				std::string tmp = files.back();
				MPI_Send(op.c_str(), op.size(), MPI_CHAR, workerid, FROM_MASTER, MPI_COMM_WORLD);
				MPI_Send(images_path.c_str(), images_path.size(), MPI_CHAR, workerid, FROM_MASTER, MPI_COMM_WORLD);
				MPI_Send(dst_path.c_str(), dst_path.size(), MPI_CHAR, workerid, FROM_MASTER, MPI_COMM_WORLD);
				MPI_Send(tmp.c_str(), tmp.size(), MPI_CHAR, workerid, FROM_MASTER, MPI_COMM_WORLD);
				MPI_Send(&f_stop, 1, MPI_FLOAT, workerid, FROM_MASTER, MPI_COMM_WORLD);
				MPI_Send(&gamma, 1, MPI_FLOAT, workerid, FROM_MASTER, MPI_COMM_WORLD);
				MPI_Send(&block_size, 1, MPI_INT, workerid, FROM_MASTER, MPI_COMM_WORLD);
				files.pop_back();
				//receive file name
				//receive time
				//j--;
			} else {
				for (int i = 1; i <= nworkers; i++) {
					std::string op = "finish";
					MPI_Send(op.c_str(), op.size(), MPI_CHAR, j, FROM_MASTER, MPI_COMM_WORLD);
				}
				// send op = finish to all workers
				break;
			}
		}*/
	}

	if(taskid > 0) {
		while(true) {
			MPI_Status status_op, status_images, status_dst, status_file;
			int length_op = 0, length_images = 0, length_dst = 0, length_file = 0;
			char* op = NULL;
			char* images_path = NULL;
			char* dst_path = NULL;
			char* file_name = NULL;
			float f_stop;
			float gamma;
			int block_size;

			// Receive op to perform
			// Probe for an incoming message from process zero
			MPI_CHECK(MPI_Probe(0, FROM_MASTER, MPI_COMM_WORLD, &status_op));
			// When probe returns, the status object has the size and other
			// attributes of the incoming message. Get the message size
			MPI_CHECK(MPI_Get_count(&status_op, MPI_CHAR, &length_op));
			// Allocate a buffer to hold the incoming string
			op = (char*)malloc(sizeof(char) * length_op);
			// Now receive the message with the allocated buffer
			MPI_CHECK(MPI_Recv(op, length_op, MPI_CHAR, 0, FROM_MASTER, MPI_COMM_WORLD, MPI_STATUS_IGNORE));

			std::cout << "op: " << op << std::endl;

			if(op == "finish") {
				free(op);
				break;
			}
			free(op);

			// Receive images_path
			MPI_CHECK(MPI_Probe(0, FROM_MASTER, MPI_COMM_WORLD, &status_images));
			MPI_CHECK(MPI_Get_count(&status_images, MPI_CHAR, &length_images));
			images_path = (char*)malloc(sizeof(char) * length_images);
			MPI_CHECK(MPI_Recv(images_path, length_images, MPI_CHAR, 0, FROM_MASTER, MPI_COMM_WORLD, MPI_STATUS_IGNORE));

//			std::cout << "images_path: " << images_path << std::endl;

			// Receive dst_path
			MPI_CHECK(MPI_Probe(0, FROM_MASTER, MPI_COMM_WORLD, &status_dst));
			MPI_CHECK(MPI_Get_count(&status_dst, MPI_CHAR, &length_dst));
			dst_path = (char*)malloc(sizeof(char) * length_dst);
			MPI_CHECK(MPI_Recv(dst_path, length_dst, MPI_CHAR, 0, FROM_MASTER, MPI_COMM_WORLD, MPI_STATUS_IGNORE));

			// Receive file_name
			MPI_CHECK(MPI_Probe(0, FROM_MASTER, MPI_COMM_WORLD, &status_file));
			MPI_CHECK(MPI_Get_count(&status_file, MPI_CHAR, &length_file));
			file_name = (char*)malloc(sizeof(char) * length_file);
			MPI_CHECK(MPI_Recv(file_name, length_file, MPI_CHAR, 0, FROM_MASTER, MPI_COMM_WORLD, MPI_STATUS_IGNORE));

			MPI_CHECK(MPI_Recv(&f_stop, 1, MPI_FLOAT, 0, FROM_MASTER, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
			MPI_CHECK(MPI_Recv(&gamma, 1, MPI_FLOAT, 0, FROM_MASTER, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
			MPI_CHECK(MPI_Recv(&block_size, 1, MPI_INT, 0, FROM_MASTER, MPI_COMM_WORLD, MPI_STATUS_IGNORE));

			std::cout << "Process \"" << taskid << "\" received file \"" << file_name << "\" and settings: " <<
					  images_path << ", " << dst_path << ", " << f_stop << ", " << gamma << ", " << block_size << std::endl;

			//task(std::string(file_name), f_stop, gamma, block_size, std::string(images_path), std::string(dst_path));

			MPI_CHECK(MPI_Send(&taskid, 1, MPI_INT, 0, FROM_WORKER, MPI_COMM_WORLD));

			free(images_path); free(dst_path); free(file_name);
			// receive op
			// MPI_RECV();
			// if(op == "finish") {
			//      break;
			//    }
			// receive file name
			// receive f_stop
			// receive gamma
			// receive block_size

			// tic
			// call to task();
			// toc

			// send name of file processed
			// send time
		}
	}

	MPI_CHECK(MPI_Finalize());
	return 0;
}
