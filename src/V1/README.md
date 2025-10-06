# V1 — KLT Feature Tracker (Sequential Baseline)

Purpose
- Baseline CPU-only implementation of the Kanade–Lucas–Tomasi (KLT) feature tracker.
- Used for correctness, visualization, and profiling before GPU acceleration (V2+).

Quick facts
- Image size used in examples: 320×240
- Default features requested: 100
- Pyramid levels: 2 (configurable in tracking context)
- Outputs: features_frame0.ppm/.txt, features_frame1.ppm/.txt

Build (Linux / macOS)
- From project root:
  - cd src/V1
  - make all
- Typical target names: `all`, `lib`, `klt_app` (see top-level Makefile if present)

Build (Windows)
- Use provided Makefile.win or Visual Studio project if available:
  - make -f Makefile.win all

Run
- Ensure `data/img0.pgm` and `data/img1.pgm` exist relative to the executable (../../data/ by default).
- Execute the binary:
  - ./klt_app    (or the produced executable name)
- Example console outputs feature detection and tracking summaries and writes PPM/TXT files.

Inputs
- PGM images (gray-scale). Example sequence: img0.pgm, img1.pgm, ...
- Default paths assume `data/` directory at repository root.

Outputs
- features_frame0.ppm, features_frame0.txt — detected features in frame 0
- features_frame1.ppm, features_frame1.txt — tracked features in frame 1
- Console prints: tracking context, per-feature coordinates/status, simple performance summary

Quick validation
- Run V1 and compare produced TXT coordinates to reference outputs (if provided).
- Visually inspect PPM overlays to confirm tracked feature consistency.
- Use gprof / perf or add micro-timers to measure per-function cost (useful before porting).

Notes for GPU porting (next steps)
- Prioritize:
  1. KLTSelectGoodFeatures() — per-pixel cornerness computation
  2. KLTTrackFeatures() — per-feature iterative tracking
  3. Pyramid/convolution routines — separable filters
- Keep images/pyramids resident on device and minimize host↔device transfers.
- Add unit tests to assert coordinate consistency within sub-pixel tolerances.

Troubleshooting
- "Could not read img0.pgm": verify data path and file permissions.
- If features per second shows "N/A": timing interval was too small; run multiple frames or add repeated runs for stable timings.

Contacts / Authors
- NABEEHA MAHMOOD — 23i-0588  
- MAHAM FATIMA     — 23i-0685

License / Attribution
- Educational project for CS 4110. Original KLT code attribution: Stan Birchfield (public domain reference).

<!-- End of V1 README -->
