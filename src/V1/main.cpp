/**********************************************************************
 * KLT Feature Tracker V1 - Main Application
 * 
 * This is the baseline sequential implementation of the KLT feature tracker.
 * It finds the 100 best features in an image and tracks them to the next frame.
 **********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

extern "C" {
    #include "pnmio.h"
    #include "klt.h"
}

int main(int argc, char* argv[])
{
    printf("KLT Feature Tracker V1 - Sequential Implementation\n");
    printf("==================================================\n\n");

    // Timer variables for performance measurement
    clock_t start_time, end_time;
    double cpu_time_used;

    unsigned char *img1, *img2;
    KLT_TrackingContext tc;
    KLT_FeatureList fl;
    int nFeatures = 100;
    int ncols, nrows;
    int i;

    start_time = clock();

    // Create tracking context and feature list
    tc = KLTCreateTrackingContext();
    if (!tc) {
        printf("Error: Failed to create tracking context\n");
        return -1;
    }
    
    printf("Created tracking context\n");
    KLTPrintTrackingContext(tc);
    
    fl = KLTCreateFeatureList(nFeatures);
    if (!fl) {
        printf("Error: Failed to create feature list\n");
        return -1;
    }

    // Read input images
    printf("\nReading input images...\n");
    img1 = pgmReadFile("../../data/img0.pgm", NULL, &ncols, &nrows);
    if (!img1) {
        printf("Error: Could not read img0.pgm from data directory\n");
        return -1;
    }
    
    img2 = pgmReadFile("../../data/img1.pgm", NULL, &ncols, &nrows);
    if (!img2) {
        printf("Error: Could not read img1.pgm from data directory\n");
        free(img1);
        return -1;
    }
    
    printf("Successfully read images (size: %dx%d)\n", ncols, nrows);

    // Select good features in the first image
    printf("\nSelecting good features...\n");
    KLTSelectGoodFeatures(tc, img1, ncols, nrows, fl);

    printf("Features detected in first image:\n");
    for (i = 0; i < fl->nFeatures; i++) {
        printf("Feature #%d: (%.2f, %.2f) with value %d\n",
               i, fl->feature[i]->x, fl->feature[i]->y,
               fl->feature[i]->val);
    }

    // Write features to PPM file for visualization
    KLTWriteFeatureListToPPM(fl, img1, ncols, nrows, "features_frame0.ppm");
    KLTWriteFeatureList(fl, "features_frame0.txt", "%3d");
    printf("Wrote features to features_frame0.ppm and features_frame0.txt\n");

    // Track features to the second image
    printf("\nTracking features to second image...\n");
    KLTTrackFeatures(tc, img1, img2, ncols, nrows, fl);

    printf("Features tracked in second image:\n");
    int tracked_count = 0;
    for (i = 0; i < fl->nFeatures; i++) {
        if (fl->feature[i]->val >= 0) {
            printf("Feature #%d: (%.2f, %.2f) with value %d\n",
                   i, fl->feature[i]->x, fl->feature[i]->y,
                   fl->feature[i]->val);
            tracked_count++;
        } else {
            printf("Feature #%d: LOST (val=%d)\n", i, fl->feature[i]->val);
        }
    }

    // Write tracked features
    KLTWriteFeatureListToPPM(fl, img2, ncols, nrows, "features_frame1.ppm");
    KLTWriteFeatureList(fl, "features_frame1.txt", "%3d");
    printf("Wrote tracked features to features_frame1.ppm and features_frame1.txt\n");

    // Performance summary
    end_time = clock();
    cpu_time_used = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
    
    printf("\n=== Performance Summary ===\n");
    printf("Total features detected: %d\n", fl->nFeatures);
    printf("Successfully tracked: %d\n", tracked_count);
    printf("Lost features: %d\n", fl->nFeatures - tracked_count);
    printf("Processing time: %.6f seconds\n", cpu_time_used);
    printf("Features per second: %.2f\n", fl->nFeatures / cpu_time_used);

    // Cleanup
    free(img1);
    free(img2);
    KLTFreeFeatureList(fl);
    KLTFreeTrackingContext(tc);

    printf("\nKLT V1 processing completed successfully!\n");
    return 0;
}

