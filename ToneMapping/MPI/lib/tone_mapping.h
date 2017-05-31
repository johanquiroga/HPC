#ifndef HPC_TONE_MAPPING_H
#define HPC_TONE_MAPPING_H

// Forward declarations
extern "C" {
    float gamma_tonemap(float *h_ImageData, float *h_ImageOut, int width, int height, int channels, float f_stop, float gamma, int blockSize,
                        int sizeImage);
	float adaptive_log_tonemap(float *h_ImageData, float *h_ImageOut, int width, int height, int channels, float b, float ld_max, int blockSize,
	                           int sizeImage);
	float log_tonemap(float *h_ImageData, float *h_ImageOut, int width, int height, int channels, float k, float q, int blockSize,
	                  int sizeImage);
    void my_abort(int err);
}

#endif //HPC_TONE_MAPPING_H
