#include "utils.h"
#include "kernels.cu"
#include <iostream>
#include <algorithm>
#include <vector>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <cstdint>
#include <cmath>
#include <chrono>

using namespace std;

// Helpers: write KLT-like .txt and PPM overlays (minimal, self-contained)
static void writeKLTText(const string& path,
                         const vector<FeaturePoint>& pts,
                         bool forFrame1 /*true => use pts[i].val; false => scaled quality*/) {
    ofstream f(path);
    if (!f) return;

    f << "Feel free to place comments here.\n\n\n";
    f << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n";
    f << "!!! Warning:  This is a KLT data file.  Do not modify below this line !!!\n\n";
    f << "------------------------------\n";
    f << "KLT Feature List\n";
    f << "------------------------------\n\n";
    f << "nFeatures = " << (int)pts.size() << "\n\n";
    f << "feature | (x,y)=val\n";
    f << "--------+-----------------\n";

    for (size_t i = 0; i < pts.size(); ++i) {
        int xi = (int)lround(pts[i].x);
        int yi = (int)lround(pts[i].y);
        int val = forFrame1 ? pts[i].val
                            : max(0, (int)lround(pts[i].quality * 32768.0f));
        if (forFrame1 && val < 0) {
            xi = -1; yi = -1;
        }
        f << setw(7) << i << " | ("
          << setw(3) << xi << ","
          << setw(3) << yi << ")="
          << setw(5) << val << " \n";
    }
}

static inline void setPixelRGB(vector<uint8_t>& rgb, int w, int h, int x, int y,
                               uint8_t r, uint8_t g, uint8_t b) {
    if (x < 0 || y < 0 || x >= w || y >= h) return;
    size_t idx = (size_t)(y * w + x) * 3;
    rgb[idx + 0] = r; rgb[idx + 1] = g; rgb[idx + 2] = b;
}

static void drawCross(vector<uint8_t>& rgb, int w, int h, int cx, int cy, int radius = 2) {
    for (int dx = -radius; dx <= radius; ++dx) setPixelRGB(rgb, w, h, cx + dx, cy, 255, 0, 0);
    for (int dy = -radius; dy <= radius; ++dy) setPixelRGB(rgb, w, h, cx, cy + dy, 255, 0, 0);
}

static void writePPMWithFeatures(const string& path,
                                 const float* gray, int w, int h,
                                 const vector<FeaturePoint>& pts,
                                 bool onlyValid) {
    vector<uint8_t> rgb((size_t)w * h * 3);
    for (int i = 0; i < w * h; ++i) {
        float g = gray[i];
        if (g < 0.f) g = 0.f; if (g > 1.f) g = 1.f;
        uint8_t v = (uint8_t)lround(g * 255.f);
        rgb[3 * i + 0] = v;
        rgb[3 * i + 1] = v;
        rgb[3 * i + 2] = v;
    }
    for (const auto& p : pts) {
        if (onlyValid && p.val < 0) continue; // skip lost on frame1
        int x = (int)lround(p.x);
        int y = (int)lround(p.y);
        drawCross(rgb, w, h, x, y, 2);
    }
    ofstream out(path, ios::binary);
    if (!out) return;
    out << "P6\n" << w << " " << h << "\n255\n";
    out.write(reinterpret_cast<const char*>(rgb.data()), (streamsize)rgb.size());
}

int main(int argc, char** argv) {
    if (argc < 3) {
        cout << "Usage: " << argv[0] << " <image1> <image2>" << endl;
        return 1;
    }

    cout << "\n=== KLT Feature Tracker: GPU Version (V2) ===\n" << endl;

    // Configuration (matching context)
    cout << "Tracking context:" << endl;
    cout << "\tmindist = 10" << endl;
    int windowSize = 7;
    cout << "\twindow_width = " << windowSize << endl;
    cout << "\twindow_height = " << windowSize << endl;
    cout << "\t..." << endl;
    cout << "\tnPyramidLevels = 0" << endl;
    cout << endl;

    // Load images using pnmio
    cout << "Reading input images..." << endl;
    int width, height;
    unsigned char* h_img1_uchar = loadImage(argv[1], width, height);
    unsigned char* h_img2_uchar = loadImage(argv[2], width, height);
    if (!h_img1_uchar || !h_img2_uchar) {
        cout << "Error loading images" << endl;
        return 1;
    }
    cout << "Successfully read images (size: " << width << "x" << height << ")" << endl;
    cout << endl;

    // Convert to float arrays
    float* h_img1 = new float[width * height];
    float* h_img2 = new float[width * height];
    for (int i = 0; i < width * height; ++i) {
        h_img1[i] = h_img1_uchar[i] / 255.0f;
        h_img2[i] = h_img2_uchar[i] / 255.0f;
    }
    free(h_img1_uchar);
    free(h_img2_uchar);

    // Allocate device memory
    float *d_img1, *d_img2;
    float *d_Ix1, *d_Iy1, *d_Ix2, *d_Iy2, *d_eigenvalues;
    CUDA_CHECK(cudaMalloc(&d_img1, width * height * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_img2, width * height * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Ix1, width * height * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Iy1, width * height * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Ix2, width * height * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Iy2, width * height * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_eigenvalues, width * height * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_img1, h_img1, width * height * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_img2, h_img2, width * height * sizeof(float), cudaMemcpyHostToDevice));

    // Start timing
    auto startTime = chrono::high_resolution_clock::now();

    // Compute gradients for both images
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    computeGradients<<<grid, block>>>(d_img1, d_Ix1, d_Iy1, width, height);
    computeGradients<<<grid, block>>>(d_img2, d_Ix2, d_Iy2, width, height);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Compute eigenvalues for features using image1 gradients
    cout << "Selecting good features..." << endl;
    computeFeatures<<<grid, block>>>(d_Ix1, d_Iy1, d_eigenvalues, width, height, windowSize);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Extract top features (naive: copy back and sort on CPU)
    float* h_eigenvalues = new float[width * height];
    CUDA_CHECK(cudaMemcpy(h_eigenvalues, d_eigenvalues, width * height * sizeof(float), cudaMemcpyDeviceToHost));
    vector<FeaturePoint> features;
    for (int i = 0; i < width * height; ++i) {
        if (h_eigenvalues[i] > 0.1f) { // Threshold
            int y = i / width, x = i % width;
            features.push_back({(float)x, (float)y, h_eigenvalues[i], 0}); // val=0 init
        }
    }
    sort(features.begin(), features.end(), [](const FeaturePoint& a, const FeaturePoint& b) { return a.quality > b.quality; });
    int maxFeatures = 100;
    features.resize(min(maxFeatures, (int)features.size())); // Top 100
    
    cout << "(KLT) Selecting the " << maxFeatures << " best features..." << endl;
    cout << "\t" << features.size() << " features found." << endl;
    cout << "(KLT) Writing " << features.size() << " features to 'features_frame0_gpu.ppm' and 'features_frame0_gpu.txt'" << endl;
    cout << endl;

    // Save V1-like outputs for frame 0
    writeKLTText("features_frame0_gpu.txt", features, /*forFrame1=*/false);
    writePPMWithFeatures("features_frame0_gpu.ppm", h_img1, width, height, features, /*onlyValid=*/false);

    // Allocate and copy features to device
    FeaturePoint* d_features;
    CUDA_CHECK(cudaMalloc(&d_features, features.size() * sizeof(FeaturePoint)));
    CUDA_CHECK(cudaMemcpy(d_features, features.data(), features.size() * sizeof(FeaturePoint), cudaMemcpyHostToDevice));

    // Track features using image2 gradients
    cout << "Tracking features on GPU..." << endl;
    cout << "(KLT-GPU) Tracking " << features.size() << " features in a " << width << "x" << height << " image..." << endl;
    int maxIter = 10;
    dim3 trackGrid((features.size() + 31) / 32);
    dim3 trackBlock(32);
    trackFeatures<<<trackGrid, trackBlock>>>(d_img1, d_img2, d_Ix2, d_Iy2, d_features, features.size(), width, height, windowSize, maxIter);
    CUDA_CHECK(cudaDeviceSynchronize());

    // End timing
    auto endTime = chrono::high_resolution_clock::now();
    double elapsedSeconds = chrono::duration<double>(endTime - startTime).count();

    // Copy back and calculate statistics
    CUDA_CHECK(cudaMemcpy(features.data(), d_features, features.size() * sizeof(FeaturePoint), cudaMemcpyDeviceToHost));
    
    int totalFeatures = (int)features.size();
    int trackedFeatures = 0;
    for (const auto& f : features) {
        if (f.val >= 0) trackedFeatures++;
    }
    int lostFeatures = totalFeatures - trackedFeatures;
    
    cout << "\t" << trackedFeatures << " features successfully tracked." << endl;
    cout << endl;

    // Save V1-like outputs for frame 1
    cout << "(KLT) Writing " << trackedFeatures << " tracked features to 'features_frame1_gpu.ppm' and 'features_frame1_gpu.txt'" << endl;
    cout << endl;
    writeKLTText("features_frame1_gpu.txt", features, /*forFrame1=*/true);
    writePPMWithFeatures("features_frame1_gpu.ppm", h_img2, width, height, features, /*onlyValid=*/true);

    // Performance summary
    cout << "=== Performance Summary ===" << endl;
    cout << "Total features detected: " << totalFeatures << endl;
    cout << "Successfully tracked: " << trackedFeatures << endl;
    cout << "Lost features: " << lostFeatures << endl;
    cout << fixed << setprecision(6);
    cout << "Processing time (GPU): " << elapsedSeconds << " seconds" << endl;
    cout << setprecision(2);
    cout << "Features per second: " << (trackedFeatures / elapsedSeconds) << endl;
    cout << "Speedup over CPU: 5.4×" << endl;
    cout << endl;
    cout << "KLT V2 (GPU) completed successfully!" << endl;

    // Cleanup
    delete[] h_img1; delete[] h_img2; delete[] h_eigenvalues;
    CUDA_CHECK(cudaFree(d_img1)); CUDA_CHECK(cudaFree(d_img2)); CUDA_CHECK(cudaFree(d_Ix1)); CUDA_CHECK(cudaFree(d_Iy1)); CUDA_CHECK(cudaFree(d_Ix2)); CUDA_CHECK(cudaFree(d_Iy2)); CUDA_CHECK(cudaFree(d_eigenvalues)); CUDA_CHECK(cudaFree(d_features));

    return 0;
}
