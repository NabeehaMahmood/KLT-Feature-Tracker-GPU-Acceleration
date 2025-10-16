#ifndef UTILS_H
#define UTILS_H

#include <cuda_runtime.h>
#include <vector>
#include <string> // added

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while (0)

// Struct for feature points
struct FeaturePoint {
    float x, y;
    float quality; // For sorting
    int   val;     // 0 tracked; negative => lost (match V1 semantics)
};

// Function to load image as grayscale unsigned char array (using pnmio)
unsigned char* loadImage(const std::string& path, int& width, int& height);

// Function to save tracked points to file or display
void savePoints(const std::vector<FeaturePoint>& points, const std::string& filename);

#endif
