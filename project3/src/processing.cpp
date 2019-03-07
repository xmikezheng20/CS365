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

// visualize the connected components labeled image
cv::Mat visConnectedComponents(cv::Mat labeled, int numLabels) {
    printf("Visualizing connected components labels\n");
    std::vector<cv::Vec3b> colors(numLabels);
    colors[0] = cv::Vec3b(0, 0, 0);//background
    colors[1] = cv::Vec3b(0, 0, 255);//label 1:red
    colors[2] = cv::Vec3b(0, 255, 0);//label 2:green
    colors[3] = cv::Vec3b(255, 0, 0);//label 3:blue
    for(int label = 4; label < numLabels; ++label){
        colors[label] = cv::Vec3b( (rand()&255), (rand()&255), (rand()&255) );
    }
    // visualize the labels
    cv::Mat dst;
    dst = cv::Mat(labeled.size(), CV_8UC3);
    for(int r = 0; r < dst.rows; ++r){
        for(int c = 0; c < dst.cols; ++c){
            int label = labeled.at<int>(r, c);
            cv::Vec3b &pixel = dst.at<cv::Vec3b>(r, c);
            pixel = colors[label];
         }
     }

    // cv::namedWindow("test", 1);
    // cv::imshow("test", dst);
    // cv::waitKey(0);

    return dst;
}

// create a binary image of the specified region
cv::Mat extractRegion(cv::Mat labeled, int regionId) {
    printf("Extracting region %d\n", regionId);
    cv::Mat region;
    region = cv::Mat(labeled.size(), CV_8UC1);
    for (int r = 0; r<region.rows; r++) {
        for (int c= 0; c<region.cols; c++) {
            int label = labeled.at<int>(r, c);
            uchar &pixel = region.at<uchar>(r, c);
            if (label == regionId) {
                pixel = 255;
            } else {
                pixel = 0;
            }
        }
    }

    // cv::namedWindow("test", 1);
    // cv::imshow("test", region);
    // cv::waitKey(0);
    return region;
}

// find contour of region, discard small region, extract features
// return 0 if region is valid, 1 if region is discarded
int extractFeature(cv::Mat region, int regionId, double **feature) {
    printf("Extracting feature from region %d\n", regionId);


    

    return 0;
}
