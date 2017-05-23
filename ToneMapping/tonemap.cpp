#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string>
#include <vector>
#include <mpi.h>
#include <opencv2/core/core.hpp>

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

void task(std::string image_name, float f_stop, float gamma, int block_size)
{
    float *h_ImageData, *h_ImageOut;
    std::string image_out_name;
    Mat hdr, ldr;
    Size imageSize;
    int width, height, channels, sizeImage;

    hdr = imread(image_name.c_str(), -1);

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
    image_out_name = "Results/" + change_image_extension(image_name);
    imwrite(image_out_name.c_str(), ldr);

    free(h_ImageOut);
}

int main(int argc, char** argv)
{
    int numtasks, taskid;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

    if(numtasks < 2) {
        printf("Need at least two tasks...\n");
        my_abort(EXIT_FAILURE);
    }

    if(argc !=3) {
//        printf("No image Data \n");
        printf("Usage: ./test <f_stop> <gamma>");
        my_abort(EXIT_FAILURE);
//        return -1;
    }

    int nworkers = numtasks - 1;
    float f_stop = 0.0;
    float gamma = 0.0;

    //char* image_name = argv[1];
    if(taskid == 0) {
        float f_stop = atof(argv[1]);
        float gamma = atof(argv[2]);
        std::vector<std::string> files;
        read_files(files, "./images");

        int j=1;
        while(true) {
            if(!files.empty()) {
                if(j<=nworkers) {
                    // send op = "work";
                    // MPI_Send(op.c_str(), op.length(), MPI_CHAR, j, FROM_MASTER, MPI_COMM_WORLD);
                    // send file name: file.back();
                    // send f_stop
                    // send gamma
                    // send BLOCK_SIZE
                    // files.pop_back();
                    j++;
                } else {
                    //receive file name
                    //receive time
                    j--;
                }
            } else {
                // send op = finish to all workers
                break;
            }
        }
    }

    if(taskid > 0) {
        while(true) {
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

    MPI_Finalize();
    return 0;
}
