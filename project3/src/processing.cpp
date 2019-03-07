/*
    processing.cpp
    Mike Zheng and Heidi He
    CS365 project 3
    3/6/19

    a library of 2d image processing functions

*/

#include <cstdio>
#include <cstring>
#include "opencv2/opencv.hpp"

#include "processing.hpp"

// threshold for high saturation dark regions
cv::Mat threshold(cv::Mat src) {
    printf("thresholding\n");
    cv::Mat hsv, saturation, intensity, dst;

    // get saturation
    cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);
    std::vector<cv::Mat> splited;
    cv::split(src, splited);

    cv::threshold(splited[1], saturation, 70, 255, 0);
    cv::threshold(splited[2], intensity, 55, 255, 0);

    // combine saturation and intensity
    cv::bitwise_and(saturation, intensity, dst);

    // cv::namedWindow("test", 1);
    // cv::imshow("test", dst);
    // cv::waitKey(0);

    return dst;
}
