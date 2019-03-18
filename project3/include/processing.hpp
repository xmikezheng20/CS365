/*
    processing.hpp
    Mike Zheng and Heidi He
    CS365 project 3
    3/6/19

    a library of 2d image processing functions

*/

#include <cstdio>
#include <cstring>
#include "opencv2/opencv.hpp"

// threshold for high saturation dark regions
cv::Mat threshold(cv::Mat src);

// apply morphological operations
cv::Mat morphOps(cv::Mat src);

// visualize the connected components labeled image
cv::Mat visConnectedComponents(cv::Mat labeled, int numLabels, std::vector<int> skipLabels);

// create a binary image of the specified region
cv::Mat extractRegion(cv::Mat labeled, int regionId);

// find contour of region, discard small region, extract features
// return 0 if region is valid, 1 if region is discarded
int extractFeature(cv::Mat region, int regionId,
    std::vector<std::vector<cv::Point>> &contours,
    std::vector<cv::Vec4i> &hierarchy, std::vector<double> &feature);

// visualize contour and features
cv::Mat visFeature(cv::Mat labeled, int numLabels, std::vector<int> skipLabels,
    std::vector<std::vector<cv::Point>> &contoursVector,
    std::vector<cv::Vec4i> &hierarchyVector,
    std::vector<std::vector<double>> feature, std::vector<std::vector<std::string>> &catsVector,
    int state);
