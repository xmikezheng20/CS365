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
    printf("Thresholding\n");
    cv::Mat hsv, saturation, intensity, dst;

    // get saturation
    cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);
    std::vector<cv::Mat> splited;
    cv::split(src, splited);

    cv::threshold(splited[1], saturation, 70, 255, 1);
    cv::threshold(splited[2], intensity, 85, 255, 1);

    // combine saturation and intensity
    cv::bitwise_or(saturation, intensity, dst);

    // cv::namedWindow("test", 1);
    // cv::imshow("test", dst);
    // cv::waitKey(0);

    return dst;
}

// apply morphological operations
// close*3, open*2
cv::Mat morphOps(cv::Mat src) {
    printf("Applying morphological operations\n");
    cv::Mat dst;
    int morph_elem = 0; // 0: Rect - 1: Cross - 2: Ellipse
    int morph_size_L = 3;
    int morph_size_S = 2;
    cv::Mat elementL = cv::getStructuringElement( morph_elem,
        cv::Size( 2*morph_size_L + 1, 2*morph_size_L+1 ),
        cv::Point( morph_size_L, morph_size_L ) );

    cv::Mat elementS = cv::getStructuringElement( morph_elem,
        cv::Size( 2*morph_size_S + 1, 2*morph_size_S+1 ),
        cv::Point( morph_size_S, morph_size_S ) );

    cv::morphologyEx(src, dst, cv::MORPH_CLOSE, elementL, cv::Point(-1,-1), 3);
    cv::morphologyEx(dst, dst, cv::MORPH_OPEN, elementS, cv::Point(-1,-1), 2);

    // cv::namedWindow("test", 1);
    // cv::imshow("test", dst);
    // cv::waitKey(0);

    return dst;

}
