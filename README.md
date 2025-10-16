# KLT Feature Tracker GPU Acceleration

**CS 4110 - High Performance Computing with GPUs**  
**Complex Computing Problem (CCP)**

## Project Overview

This project focuses on the GPU acceleration of the well-known Kanade-Lucas-Tomasi (KLT) feature tracking algorithm. The goal is to implement and optimize the KLT feature tracker using CUDA to achieve significant performance improvements over the baseline sequential implementation.

### Team Information
- **Project**: KLT Feature Tracker Acceleration on GPUs
- **Course**: CS 4110 - High Performance Computing with GPUs
- **Semester**: Fall 2025

## Project Structure

```
klt-gpu-acceleration/
├── src/
│   ├── V1/          # Sequential baseline implementation
│   ├── V2/          # Naive GPU implementation
│   ├── V3/          # Optimized GPU implementation
│   └── V4/          # OpenACC pragma-based implementation
├── data/            # Test images (img0.pgm - img9.pgm)
├── report/          # Project reports and documentation
├── slides/          # Presentation materials
└── README.md        # This file
```

## Feature Tracking Background

Feature tracking in computer vision involves detecting distinctive points (features) in images and following their movement across video frames. The KLT tracker is one of the most widely used algorithms for this purpose.

### KLT Algorithm Overview
1. **Feature Detection**: Uses Tomasi-Kanade "Good Features to Track" method to detect corner-like features
2. **Optical Flow**: Assumes brightness constancy between consecutive frames
3. **Lucas-Kanade Method**: Tracks features using local patch matching and image gradients
4. **Pyramid Implementation**: Uses multi-scale approach for handling large motions

### Applications
- Video stabilization
- Object tracking
- Structure-from-motion (3D reconstruction)
- Augmented reality
- Robotics navigation

## Version Descriptions

### V1 - Sequential Baseline Implementation
- **Status**: ✅ Complete
- **Description**: Original sequential CPU implementation
- **Features**:
  - Detects 100 best features in first image
  - Tracks features to second image
  - Outputs feature locations and performance metrics
  - Generates visualization files (PPM format)
- **Performance**: 0.065 seconds, 92/100 features tracked
  
### V2 - Naive GPU Implementation
- **Status**: ✅ Complete
- **Description**: Direct CUDA port of KLT core algorithms
- **Features**:
  - GPU kernels for gradient computation, feature detection, and tracking
  - Host-side feature selection (top 100)
  - Single-scale Lucas-Kanade tracking
  - No pyramids, shared memory, or advanced optimizations
- **Performance**: 0.034 seconds, 100/100 features tracked
- **Speedup**: ~1.9× over V1
- **Test Images**: img0.pgm and img1.pgm (320×240)

### V3 - Optimized GPU Implementation  
- **Status**: 🔄 Planned
- **Description**: Advanced GPU optimizations including:
  - Launch configuration optimization
  - Occupancy analysis
  - Communication optimizations
  - Memory hierarchy utilization

### V4 - OpenACC Implementation
- **Status**: 🔄 Planned
- **Description**: Pragma-based GPU acceleration using OpenACC directives

## Getting Started

### Prerequisites
- GCC compiler (with C++ support)
- Make utility
- For GPU versions (V2-V4): NVIDIA CUDA Toolkit
- For V4: OpenACC-compatible compiler (PGI/NVIDIA HPC SDK)

### V1 Setup and Compilation

1. **Navigate to V1 directory**:
   ```bash
   cd src/V1
   ```

2. **Build the application**:
   ```bash
   # For Windows
   make -f Makefile.win all
   
   # For Linux/Unix
   make all
   ```

3. **Run the application**:
   ```bash
   # Windows
   .\klt_app.exe
   
   # Linux/Unix
   ./klt_app
   ```

### V2 Setup and Compilation

1. **Navigate to V2 directory**:
   ```bash
   cd src/V2
   ```

2. **Build the application**:
   ```bash
   make
   ```

3. **Run the application**:
   ```bash
   ./v2_klt ../../data/img0.pgm ../../data/img1.pgm
   ```

4. **Output files**:
   - `features_frame0_gpu.txt` and `features_frame0_gpu.ppm`
   - `features_frame1_gpu.txt` and `features_frame1_gpu.ppm`

### Available Make Targets (V1)

| Target        | Description                        |
|---------------|------------------------------------|
| `all`         | Build library and main application |
| `lib`         | Build KLT library only             |
| `klt_app.exe` | Build main application             |
| `examples`    | Build all example applications     |
| `profile`     | Run with profiling enabled         |
| `benchmark`   | Run performance benchmark          |
| `clean`       | Remove generated files             |
| `help`        | Show available targets             |

## Input Data

The project uses a sequence of PGM (Portable GrayMap) format images:
- `img0.pgm` to `img9.pgm` - Test image sequence
- Images are 320x240 pixels
- Located in the `data/` directory

## Output Files

**V1 Output**:
- `features_frame0.ppm` - Visual representation of detected features
- `features_frame0.txt` - Text file with feature coordinates
- `features_frame1.ppm` - Visual representation of tracked features  
- `features_frame1.txt` - Text file with tracked feature coordinates

**V2 Output**:
- `features_frame0_gpu.ppm` - Visual representation of detected features (GPU)
- `features_frame0_gpu.txt` - Text file with feature coordinates (GPU)
- `features_frame1_gpu.ppm` - Visual representation of tracked features (GPU)
- `features_frame1_gpu.txt` - Text file with tracked feature coordinates (GPU)

  
## Performance Metrics (V1 Baseline)

### V1 Baseline (CPU)
**Test Configuration**: 320x240 images (img0.pgm, img1.pgm), 100 features
- **Processing Time**: 0.065 seconds
- **Features Detected**: 100
- **Successfully Tracked**: 92 (92% success rate)
- **Lost Features**: 8
- **Throughput**: ~1,415 features/second

### V2 Naive GPU
**Test Configuration**: 320×240 images (img0.pgm, img1.pgm), 100 features
- **Processing Time**: 0.034 seconds
- **Features Detected**: 100
- **Successfully Tracked**: 100 (100% success rate)
- **Lost Features**: 0
- **Throughput**: ~2,949 features/second
- **Speedup**: ~1.9× over V1
- **Note**: Better tracking due to FP precision differences in GPU gradient computations

  
## Project Timeline & Deliverables

| Deliverable | Deadline      | Status                |
|-------------|---------------|-----------------------|
| CCP-D1      | Oct 6, 2025   | ✅ V1 Complete        |
| CCP-D2      | Oct 17, 2025  | ✅ V2 Complete        |
| CCP-D3      | Oct 31, 2025  | 🔄 V3 Development     |
| CCP-D4      | Nov 14, 2025  | 🔄 V4 Development     |
| CCP-D5      | Final Week    | 🔄 Final Presentation |

## Contributing
V1 done by Nabeeha Mahmood 23i-0588
V2 done by Maham Fatima    23i-0685

## Course Learning Objectives (CLOs)

This project addresses:
- **CLO2**: Application profiling and hotspot identification
- **CLO3**: Data-parallel solution development using CUDA
- **CLO4**: Performance analysis and optimization on HPC systems

## References

- Original KLT implementation by Stan Birchfield (Clemson University)
- Papers on Kanade-Lucas-Tomasi feature tracking
- CUDA Programming Guide
- OpenACC Specification

## License

This project is for educational purposes as part of CS 4110. The original KLT code is in the public domain.

---

**Note**: This is the V1 and V2 implementation. V3-V4 will be developed in subsequent phases of the project.
