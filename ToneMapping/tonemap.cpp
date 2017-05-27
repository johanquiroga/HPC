#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string>
#include <cstring>
#include <vector>
#include <time.h>
#include <mpi.h>
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

	float time_result = tonemap(h_ImageData, h_ImageOut, width, height, channels, f_stop, gamma, block_size, sizeImage);

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

	if(argc !=6) {
//        printf("No image Data \n");
		printf("Usage: ./tonemapping <f_stop> <gamma> <image_src> <results_dst> <output_separator>");
		my_abort(EXIT_FAILURE);
//        return -1;
	}

	int nworkers = numtasks - 1;
	if(taskid == 0) {
		clock_t start, end;
		double batch_time;
		int block_size = BLOCK_SIZE;
		float f_stop = atof(argv[1]);
		float gamma = atof(argv[2]);
		std::string images_path(argv[3]);
		std::string dst_path(argv[4]);
		std::string separator(argv[5]);

		std::vector<std::string> files;
		read_files(files, images_path);

		if (files.size() >= nworkers) {
			start = clock();
			int j = 1;
			while (j <= nworkers) {
				// send
				std::string op("work");
				std::string tmp = files.back();
//				std::cout << "Process \"" << taskid << "\" sending file \"" << tmp << " to: " << j << std::endl;
				MPI_CHECK(MPI_Send(op.c_str(), op.length()+1, MPI_CHAR, j, FROM_MASTER, MPI_COMM_WORLD));
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
			while(true) {
				// receive
				MPI_Status status_file;
				int length_file = 0;
				char* file_name = NULL;
				int workerid;
				float elapsed_time = 0.0;

				MPI_CHECK(MPI_Recv(&workerid, 1, MPI_INT, MPI_ANY_SOURCE, FROM_WORKER, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
//				MPI_CHECK(MPI_Recv(&elapsed_time, 1, MPI_FLOAT, MPI_ANY_SOURCE, FROM_WORKER, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
				// Receive file_name
				MPI_CHECK(MPI_Probe(MPI_ANY_SOURCE, FROM_WORKER, MPI_COMM_WORLD, &status_file));
				MPI_CHECK(MPI_Get_count(&status_file, MPI_CHAR, &length_file));
				file_name = (char*)malloc(sizeof(char) * length_file);
				MPI_CHECK(MPI_Recv(file_name, length_file, MPI_CHAR, MPI_ANY_SOURCE, FROM_WORKER, MPI_COMM_WORLD, MPI_STATUS_IGNORE));

//				printTime(std::string(file_name), elapsed_time, separator);
				std::cout << "Process \"" << taskid << "\" received file \"" << std::string(file_name) << "\" from: " << workerid << std::endl;

				int tmpid = workerid;

				if (!files.empty()) {
					// send
					std::string op = "work";
					std::string tmp = files.back();
//					std::cout << "Process \"" << taskid << "\" sending file \"" << tmp << " to: " << tmpid << std::endl;
					MPI_CHECK(MPI_Send(op.c_str(), op.size()+1, MPI_CHAR, tmpid, FROM_MASTER, MPI_COMM_WORLD));
					MPI_CHECK(MPI_Send(images_path.c_str(), images_path.size()+1, MPI_CHAR, tmpid, FROM_MASTER, MPI_COMM_WORLD));
					MPI_CHECK(MPI_Send(dst_path.c_str(), dst_path.size()+1, MPI_CHAR, tmpid, FROM_MASTER, MPI_COMM_WORLD));
					MPI_CHECK(MPI_Send(tmp.c_str(), tmp.size()+1, MPI_CHAR, tmpid, FROM_MASTER, MPI_COMM_WORLD));
					MPI_CHECK(MPI_Send(&f_stop, 1, MPI_FLOAT, tmpid, FROM_MASTER, MPI_COMM_WORLD));
					MPI_CHECK(MPI_Send(&gamma, 1, MPI_FLOAT, tmpid, FROM_MASTER, MPI_COMM_WORLD));
					MPI_CHECK(MPI_Send(&block_size, 1, MPI_INT, tmpid, FROM_MASTER, MPI_COMM_WORLD));
					files.pop_back();
				} else {
					// finish
					break;
				}
			}
			j = 1;
			while (j <= nworkers) {
				// send finish
				std::string op = "finish";
//				std::cout << "Process \"" << taskid << "\" sending op \"" << op << "\" to: " << j << std::endl;
				MPI_CHECK(MPI_Send(op.c_str(), op.size()+1, MPI_CHAR, j, FROM_MASTER, MPI_COMM_WORLD));
				j++;
			}
//			std::cout << "Process \"" << taskid << "\" Finishing up..." << std::endl;
			end = clock();
		} else {
			start = clock();
			int i = 0;
			while (i < files.size()) {
				// send to i+1
				std::string op = "work";
				std::string tmp = files[i];
//				std::cout << "Process \"" << taskid << "\" sending file \"" << tmp << " to: " << i+1 << std::endl;
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
				MPI_Status status_file;
				int length_file = 0;
				char* file_name = NULL;
				int workerid;
				float elapsed_time = 0.0;

				MPI_CHECK(MPI_Recv(&workerid, 1, MPI_INT, j, FROM_WORKER, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
//				MPI_CHECK(MPI_Recv(&elapsed_time, 1, MPI_FLOAT, j, FROM_WORKER, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
				// Receive file_name
				MPI_CHECK(MPI_Probe(j, FROM_WORKER, MPI_COMM_WORLD, &status_file));
				MPI_CHECK(MPI_Get_count(&status_file, MPI_CHAR, &length_file));
				file_name = (char*)malloc(sizeof(char) * length_file);
				MPI_CHECK(MPI_Recv(file_name, length_file, MPI_CHAR, j, FROM_WORKER, MPI_COMM_WORLD, MPI_STATUS_IGNORE));

//				printTime(std::string(file_name), elapsed_time, separator);
				std::cout << "Process \"" << taskid << "\" received file \"" << std::string(file_name) << "\" from: " << workerid << std::endl;

				// send finish
				std::string op = "finish";
//				std::cout << "Process \"" << taskid << "\" sending op \"" << op << "\" to: " << j << std::endl;
				MPI_CHECK(MPI_Send(op.c_str(), op.size()+1, MPI_CHAR, j, FROM_MASTER, MPI_COMM_WORLD));
				j++;
			}
//			std::cout << "Process \"" << taskid << "\" Finishing up..." << std::endl;
			end = clock();
		}
		batch_time = ((double)(end - start))/CLOCKS_PER_SEC;
		printf("\nbatch time\n%.10f\n", batch_time);
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

//			std::cout << "op: " << op << std::endl;

			if(strcmp(op,"finish") == 0) {
				free(op);
//				std::cout << "Process \"" << taskid << "\" Finishing up..." << std::endl;
				break;
			}
			free(op);

			// Receive images_path
			MPI_CHECK(MPI_Probe(0, FROM_MASTER, MPI_COMM_WORLD, &status_images));
			MPI_CHECK(MPI_Get_count(&status_images, MPI_CHAR, &length_images));
			images_path = (char*)malloc(sizeof(char) * length_images);
			MPI_CHECK(MPI_Recv(images_path, length_images, MPI_CHAR, 0, FROM_MASTER, MPI_COMM_WORLD, MPI_STATUS_IGNORE));

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

//			std::cout << "Process \"" << taskid << "\" received file \"" << std::string(file_name) << "\" and settings: " <<
//			          std::string(images_path) << ", " << std::string(dst_path) << ", " << f_stop << ", " << gamma << ", " << block_size << std::endl;

			//task(std::string(file_name), f_stop, gamma, block_size, std::string(images_path), std::string(dst_path));

			MPI_CHECK(MPI_Send(&taskid, 1, MPI_INT, 0, FROM_WORKER, MPI_COMM_WORLD));
//			MPI_CHECK(MPI_Send(&elapsed_time, 1, MPI_FLOAT, 0, FROM_WORKER, MPI_COMM_WORLD));
			MPI_CHECK(MPI_Send(file_name, std::strlen(file_name)+1, MPI_CHAR, 0, FROM_WORKER, MPI_COMM_WORLD));

			free(images_path); free(dst_path); free(file_name);
		}
	}

	MPI_CHECK(MPI_Finalize());
//	std::cout << "Finished" << std::endl;
	return 0;
}
