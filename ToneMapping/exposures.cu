#include <cv.h>
#include <highgui.h>
#include <iostream>
#include <cuda.h>
#include <time.h>

#define BLUE 0
#define GREEN 1
#define RED 2

using namespace cv;

__device__ unsigned char clamp(int pixel) {
	if(pixel < 0) {
		pixel = 0;
	} else if (pixel > 255) {
		pixel = 255;
	}
	return (unsigned char)pixel;
}

__global__ void under_exposed(unsigned char* imageIn, unsigned char* imageOut, int width, int height, int channels) {
	int Row = blockIdx.y*blockDim.y + threadIdx.y;
	int Col = blockIdx.x*blockDim.x + threadIdx.x;
	
	if((Row < height) && (Col < width)) {
		for(int i=0; i<channels; i++) {
			imageOut[Row*width+Col*channels+i] = clamp(imageIn[Row*width+Col*channels+i] - 51);
			//imageOut[Row*width+Col*channels+i] = 100;
		}
		// imageOut[(Row*width+Col)*3+BLUE] = (int)imageIn[(Row*width+Col)*3+BLUE] - 0.2;
		// imageOut[(Row*width+Col)*3+GREEN] = (int)imageIn[(Row*width+Col)*3+GREEN] - 0.2;
		// imageOut[(Row*width+Col)*3+RED] = (int)imageIn[(Row*width+Col)*3+RED] - 0.2;
	}
}

__global__ void over_exposed(unsigned char* imageIn, unsigned char* imageOut, int width, int height, int channels) {
	int Row = blockIdx.y*blockDim.y + threadIdx.y;
        int Col = blockIdx.x*blockDim.x + threadIdx.x;

        if((Row < height) && (Col < width)) {
                for(int i=0; i<channels; i++) {
                        imageOut[Row*width+Col*channels+i] = clamp(imageIn[Row*width+Col*channels+i] + 51);
                        //imageOut[Row*width+Col*channels+i] = 100;
                }
                // imageOut[(Row*width+Col)*3+BLUE] = (int)imageIn[(Row*width+Col)*3+BLUE] - 0.2;
                // imageOut[(Row*width+Col)*3+GREEN] = (int)imageIn[(Row*width+Col)*3+GREEN] - 0.2;
                // imageOut[(Row*width+Col)*3+RED] = (int)imageIn[(Row*width+Col)*3+RED] - 0.2;
        }

}

void showImage(Mat &image, const char *window) {
	namedWindow(window, CV_WINDOW_NORMAL);
	imshow(window, image);
}

int main(int argc, char** argv) {
	cudaError_t err = cudaSuccess;
	char* image_name = argv[1];
	clock_t start_cuda, end_cuda;
	double time_used_cuda;
	int width, height, sizeImage;
	Mat image, image_out_under, image_out_over;
	unsigned char *h_ImageData, *d_ImageData, *d_ImageOut_Under, *d_ImageOut_Over, *h_ImageOut_Under, *h_ImageOut_Over;
	Size imageSize; 

	//printf("Image name: %s\n", image_name);
	image = imread(image_name, 0);
	if(argc !=2 || !image.data){
	        printf("No image Data \n");
        	return -1;
	}

	if(image.empty()) {
		printf("Couldn't find or open the image...\n");
		return -1;
	}

//	printf("Type: %d", image.depth());

	imageSize = image.size();
	width = imageSize.width;
	height = imageSize.height;
	sizeImage = sizeof(unsigned char)*width*height*image.channels();
	// sizeImageGrey = sizeof(unsigned char)*width*height;
	
	h_ImageData = (unsigned char *) malloc (sizeImage);
	h_ImageData = image.data;
	h_ImageOut_Under = (unsigned char *) malloc (sizeImage);
	h_ImageOut_Over = (unsigned char *) malloc (sizeImage);

	err = cudaMalloc((void **)&d_ImageData, sizeImage);
	if(err != cudaSuccess){
        	printf("Error reservando memoria para d_ImageData\n");
	 	exit(-1);
	}
	err = cudaMalloc((void **)&d_ImageOut_Under, sizeImage);
	if(err != cudaSuccess){
        	printf("Error reservando memoria para d_ImageOut_Under\n");
	 	exit(-1);
	}

	err = cudaMalloc((void **)&d_ImageOut_Over, sizeImage);
	if(err != cudaSuccess){
        	printf("Error reservando memoria para d_ImageOut_Over\n");
	 	exit(-1);
	}


	err = cudaMemcpy(d_ImageData, h_ImageData, sizeImage, cudaMemcpyHostToDevice);
	if(err != cudaSuccess){
        	printf("Error copiando los datos de h_ImageData a d_ImageData\n");
	 	exit(-1);
	}
	
	int blockSize = 32;
	dim3 dimBlock(blockSize, blockSize, 1);
	dim3 dimGrid(ceil(width/float(blockSize)), ceil(height/float(blockSize)), 1);
	under_exposed<<<dimGrid, dimBlock>>>(d_ImageData, d_ImageOut_Under, width, height, image.channels());
	cudaDeviceSynchronize();
	err = cudaMemcpy(h_ImageOut_Under, d_ImageOut_Under, sizeImage, cudaMemcpyDeviceToHost);
	if(err != cudaSuccess){
        	printf("Error copiando los datos de d_ImageOut_Under a h_ImageOut_Under\n");
	 	exit(-1);
	}

	over_exposed<<<dimGrid, dimBlock>>>(d_ImageData, d_ImageOut_Over, width, height, image.channels());
	cudaDeviceSynchronize();
	err = cudaMemcpy(h_ImageOut_Over, d_ImageOut_Over, sizeImage, cudaMemcpyDeviceToHost);
	if(err != cudaSuccess){
        	printf("Error copiando los datos de d_ImageOut_Over a h_ImageOut_Under\n");
	 	exit(-1);
	}

	image_out_under.create(height, width, CV_8UC1);
	image_out_under.data = h_ImageOut_Under;
	imwrite("image_out_under.jpg", image_out_under);

	image_out_over.create(height, width, CV_8UC1);
	image_out_over.data = h_ImageOut_Over;
	imwrite("image_out_over.jpg", image_out_over);

	//printf("Done\n\n");
	//showImage(image, "Image In");
	//showImage(image_out_cuda, "Image out CUDA");
	//showImage(image_out_opencv, "Image out OpenCV");
	//waitKey(0);
	free(h_ImageOut_Under); free(h_ImageOut_Over); cudaFree(d_ImageData); cudaFree(d_ImageOut_Under); cudaFree(d_ImageOut_Over);
	return 0;
}
