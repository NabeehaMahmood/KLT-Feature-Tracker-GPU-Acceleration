#include "utils.h"
// Include pnmio for image loading
extern "C" {
#include "pnmio.h"
}
#include <fstream>

unsigned char* loadImage(const std::string& path, int& width, int& height) {
    return pgmReadFile(const_cast<char*>(path.c_str()), NULL, &width, &height);
}

void savePoints(const std::vector<FeaturePoint>& points, const std::string& filename) {
    std::ofstream file(filename);
    for (const auto& p : points) {
        file << p.x << " " << p.y << std::endl;
    }
}
