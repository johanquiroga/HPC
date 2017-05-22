#ifndef HPC_TONE_MAPPING_H
#define HPC_TONE_MAPPING_H

// Forward declarations
extern "C" {
    void tonemap(float *h_ImageData, float *h_ImageOut, int width, int height, float f_stop, float gamma, int blockSize, int sizeImage);
    void my_abort(int err);
}

#endif //HPC_TONE_MAPPING_H
