#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include "utils.h"

// Kernel to compute gradients Ix and Iy
__global__ void computeGradients(const float* img, float* Ix, float* Iy, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    if (x > 0 && x < width-1 && y > 0 && y < height-1) {
        Ix[idx] = (img[(y)*width + (x+1)] - img[(y)*width + (x-1)]) / 2.0f;
        Iy[idx] = (img[(y+1)*width + x] - img[(y-1)*width + x]) / 2.0f;
    } else {
        Ix[idx] = 0.0f;
        Iy[idx] = 0.0f;
    }
}

// Kernel to compute structure tensor and eigenvalues for feature detection
__global__ void computeFeatures(float* Ix, float* Iy, float* eigenvalues, int width, int height, int windowSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    float sumIx2 = 0, sumIy2 = 0, sumIxIy = 0;
    int half = windowSize / 2;

    // Sum over window (naive loop)
    for (int dy = -half; dy <= half; ++dy) {
        for (int dx = -half; dx <= half; ++dx) {
            int nx = x + dx, ny = y + dy;
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                int nidx = ny * width + nx;
                sumIx2 += Ix[nidx] * Ix[nidx];
                sumIy2 += Iy[nidx] * Iy[nidx];
                sumIxIy += Ix[nidx] * Iy[nidx];
            }
        }
    }

    // Compute eigenvalues (trace and det for min eigenvalue approx)
    float trace = sumIx2 + sumIy2;
    float det = sumIx2 * sumIy2 - sumIxIy * sumIxIy;
    eigenvalues[idx] = (trace - sqrtf(fmaxf(trace*trace - 4.0f*det, 0.0f))) / 2.0f; // Min eigenvalue
}

// Kernel for Lucas-Kanade tracking (solve for displacement per feature)
// Use gradients of image2 (current frame) for forward-additive LK
__global__ void trackFeatures(const float* img1, const float* img2, const float* Ix2, const float* Iy2, FeaturePoint* features, int numFeatures, int width, int height, int windowSize, int maxIter) {
    int fid = blockIdx.x * blockDim.x + threadIdx.x;
    if (fid >= numFeatures) return;

    FeaturePoint fp = features[fid];
    float u = 0, v = 0; // Displacement
    int half = windowSize / 2;
    int status = 0; // 0 ok, -4 low texture, -5 insufficient support/out-of-bounds

    for (int iter = 0; iter < maxIter; ++iter) {
        float sumIxIx = 0, sumIyIy = 0, sumIxIy = 0, sumIxIt = 0, sumIyIt = 0;
        int valid = 0;

        for (int dy = -half; dy <= half; ++dy) {
            for (int dx = -half; dx <= half; ++dx) {
                int x2 = (int)(fp.x + u + dx);
                int y2 = (int)(fp.y + v + dy);
                int x1 = (int)(fp.x + dx);
                int y1 = (int)(fp.y + dy);
                if (x2 >= 0 && x2 < width && y2 >= 0 && y2 < height &&
                    x1 >= 0 && x1 < width && y1 >= 0 && y1 < height) {
                    int idx2 = y2 * width + x2;
                    int idx1 = y1 * width + x1;
                    float It = img2[idx2] - img1[idx1]; // Temporal diff
                    float gx = Ix2[idx2], gy = Iy2[idx2];
                    sumIxIx += gx * gx;
                    sumIyIy += gy * gy;
                    sumIxIy += gx * gy;
                    sumIxIt += gx * It;
                    sumIyIt += gy * It;
                    ++valid;
                }
            }
        }

        // Insufficient support: stop and mark as lost (-5)
        if (valid < 10) { status = -5; break; }

        float det = sumIxIx * sumIyIy - sumIxIy * sumIxIy;
        if (fabsf(det) <= 1e-6f) { status = -4; break; }

        float du = ( sumIyIy * (-sumIxIt) - sumIxIy * (-sumIyIt)) / det;
        float dv = ( sumIxIx * (-sumIyIt) - sumIxIy * (-sumIxIt)) / det;
        u += du;
        v += dv;

        if (fabsf(du) < 0.1f && fabsf(dv) < 0.1f) { status = 0; break; }
    }

    // Commit result
    if (status == 0) {
        float nx = fp.x + u;
        float ny = fp.y + v;
        if (nx < 0 || ny < 0 || nx >= width || ny >= height) {
            // Off-image
            features[fid].x = -1.0f;
            features[fid].y = -1.0f;
            features[fid].val = -5;
        } else {
            features[fid].x = nx;
            features[fid].y = ny;
            features[fid].val = 0;
        }
    } else {
        features[fid].x = -1.0f;
        features[fid].y = -1.0f;
        features[fid].val = status; // -4 or -5
    }
}
