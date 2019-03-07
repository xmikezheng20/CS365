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
