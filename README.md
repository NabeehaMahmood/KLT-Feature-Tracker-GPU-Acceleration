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

### V2 - Naive GPU Implementation
- **Status**: 🔄 Planned
- **Description**: Direct port of CPU code to GPU using basic CUDA

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

### Available Make Targets (V1)

| Target | Description |
|--------|-------------|
| `all` | Build library and main application |
| `lib` | Build KLT library only |
| `klt_app.exe` | Build main application |
| `examples` | Build all example applications |
| `profile` | Run with profiling enabled |
| `benchmark` | Run performance benchmark |
| `clean` | Remove generated files |
| `help` | Show available targets |

## Input Data

The project uses a sequence of PGM (Portable GrayMap) format images:
- `img0.pgm` to `img9.pgm` - Test image sequence
- Images are 320x240 pixels
- Located in the `data/` directory

## Output Files

V1 generates the following output files:
- `features_frame0.ppm` - Visual representation of detected features
- `features_frame0.txt` - Text file with feature coordinates
- `features_frame1.ppm` - Visual representation of tracked features  
- `features_frame1.txt` - Text file with tracked feature coordinates

## Performance Metrics (V1 Baseline)

**Test Configuration**: 320x240 images, 100 features
- **Processing Time**: ~1.15 seconds
- **Features Detected**: 100
- **Successfully Tracked**: 92 (92% success rate)
- **Lost Features**: 8
- **Throughput**: ~87 features/second

## Project Timeline & Deliverables

| Deliverable | Deadline | Status |
|-------------|----------|--------|
| CCP-D1 | Oct 6, 2025 | ✅ V1 Complete |
| CCP-D2 | Oct 17, 2025 | 🔄 V2 Development |
| CCP-D3 | Oct 31, 2025 | 🔄 V3 Development |
| CCP-D4 | Nov 14, 2025 | 🔄 V4 Development |
| CCP-D5 | Final Week | 🔄 Final Presentation |

## Course Learning Objectives (CLOs)

This project addresses:
- **CLO2**: Application profiling and hotspot identification
- **CLO3**: Data-parallel solution development using CUDA
- **CLO4**: Performance analysis and optimization on HPC systems

## Contributing

Each team member should:
1. Make regular commits to document individual contributions
2. Keep the GitHub repository private
3. Clearly document which parts were individual contributions
4. Follow the established directory structure

## References

- Original KLT implementation by Stan Birchfield (Clemson University)
- Papers on Kanade-Lucas-Tomasi feature tracking
- CUDA Programming Guide
- OpenACC Specification

## License

This project is for educational purposes as part of CS 4110. The original KLT code is in the public domain.

---

**Note**: This is the V1 implementation. GPU-accelerated versions (V2-V4) will be developed in subsequent phases of the project.