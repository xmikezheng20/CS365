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
#include <math.h>

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
    int morph_size_L = 3; // 7*7
    int morph_size_S = 2; // 5*5
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
cv::Mat visConnectedComponents(cv::Mat labeled, int numLabels, std::vector<int> skipLabels) {
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
            if(!(std::find(skipLabels.begin(), skipLabels.end(), label) != skipLabels.end())) {
                cv::Vec3b &pixel = dst.at<cv::Vec3b>(r, c);
                pixel = colors[label];
            }
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
int extractFeature(cv::Mat region, int regionId,
    std::vector<std::vector<cv::Point>> &contours,
    std::vector<cv::Vec4i> &hierarchy, std::vector<double> &feature) {
    printf("Extracting feature from region %d\n", regionId);
    cv::findContours( region, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0) );

    double area = fabs(cv::contourArea(cv::Mat(contours[0])));
    // stop if area is too small
    if (area < 500) {
        return 1;
    }

    // https://docs.opencv.org/4.0.1/d1/d32/tutorial_py_contour_properties.html
    // use minimal bounding box (rotated rect instead)

    // minimal bounding box
    cv::Point2f vtx[4];
    cv::RotatedRect box = cv::minAreaRect(contours[0]);
    box.points(vtx);

    // width/height
    double aspectRatio;
    double l1square = (vtx[0].x-vtx[1].x)*(vtx[0].x-vtx[1].x)
                        + (vtx[0].y-vtx[1].y)*(vtx[0].y-vtx[1].y);
    double l2square = (vtx[1].x-vtx[2].x)*(vtx[1].x-vtx[2].x)
                        + (vtx[1].y-vtx[2].y)*(vtx[1].y-vtx[2].y);

    //larger value as width
    if (l1square>l2square) {
        aspectRatio = sqrt(l1square/l2square);
    } else {
        aspectRatio = sqrt(l2square/l1square);
    }

    feature.push_back(aspectRatio);

    // extent (object area/bounding box area)
    double extent = (double)area/sqrt(l1square*l2square);

    feature.push_back(extent);

    // Solidity (object area/convex hull area)
    std::vector<cv::Point> hull;
    cv::convexHull(contours[0], hull);
    double hull_area = fabs(cv::contourArea(cv::Mat(hull)));
    double solidity = (double)area/hull_area;

    feature.push_back(solidity);

    // hu Moments
    cv::Moments moments = cv::moments(contours[0]);
    double hu[7];
    cv::HuMoments(moments, hu);
    // for (int i=0; i<7; i++) {
    //     printf("Hu moments %d is %.4f\n", i, hu[i]);
    // }
    // use first three hu moments
    for (int i=0; i<3; i++) {
        feature.push_back(hu[i]);
    }

    printf("Aspect ratio %.2f, extent %.2f, solidity %.2f, hu0: %.2f, hu1 %.2f, hu2: %.2f\n", aspectRatio, extent, solidity, hu[0], hu[1], hu[2]);

    return 0;
}

// visualize contour and features
cv::Mat visFeature(cv::Mat labeled, int numLabels, std::vector<int> skipLabels,
    std::vector<std::vector<cv::Point>> &contoursVector,
    std::vector<cv::Vec4i> &hierarchyVector,
    std::vector<std::vector<double>> feature, std::vector<std::vector<std::string>> &catsVector,
    int state) {
    printf("Visualizing contours and features\n");

    // visualize the labels
    cv::Mat dst;
    dst = visConnectedComponents(labeled, numLabels, skipLabels);

    /// Draw contours
    cv::RNG rng(12345);
    cv::Mat drawing = cv::Mat::zeros( labeled.size(), CV_8UC3 );
    for( int i = 0; i< contoursVector.size(); i++ )
        {
         cv::Scalar color = cv::Scalar(255, 255, 255);
         cv::drawContours( drawing, contoursVector, i, color, 2, 8, hierarchyVector, 0, cv::Point() );
        }

    cv::Mat merged;
    cv::bitwise_or(dst, drawing, merged);

    // write state
    if (state == 0) {
        cv::putText(merged, "Training mode", cv::Point((int)merged.size().width/2-60, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 1);
    }
    else if (state == 1) {
        cv::putText(merged, "Testing mode", cv::Point((int)merged.size().width/2-60, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 1);
    }

    // draw bounding rectangle
    // Find the minimum area enclosing bounding box
    for (int i=0; i<contoursVector.size(); i++) {
        cv::Point2f vtx[4];
        cv::RotatedRect box = cv::minAreaRect(contoursVector[i]);
        box.points(vtx);
        // Draw the bounding box
        for(int j = 0; j < 4; j++ ){
            cv::line(merged, vtx[j], vtx[(j+1)%4], cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
        }
        // fit convex hull
        // std::vector<cv::Point> hull;
        // cv::convexHull(contoursVector[i], hull);
        // for(int j = 0; j < hull.size(); j++ ){
        //     cv::line(merged, hull[j], hull[(j+1)%(hull.size())], cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
        // }

        // compute center
        cv::Moments moments = cv::moments(contoursVector[i]);
        int cX = int(moments.m10 / moments.m00);
        int cY = int(moments.m01 / moments.m00);

        // put centroid
        cv::circle(merged, cv::Point(cX, cY), 5, cv::Scalar(255, 255, 255), -1);
        // // fit ellipse
        // cv::Point2f ellipsevtx[4];
        // cv::RotatedRect ellipsebox = cv::fitEllipse(contoursVector[i]);
        // ellipsebox.points(ellipsevtx);
        // cv::ellipse(merged, ellipsebox, cv::Scalar(255, 255, 255));
        //
        // // major/minor axis
        // cv::line(merged, cv::Point((ellipsevtx[0].x+ellipsevtx[1].x)/2,(ellipsevtx[0].y+ellipsevtx[1].y)/2), cv::Point((ellipsevtx[2].x+ellipsevtx[3].x)/2,(ellipsevtx[2].y+ellipsevtx[3].y)/2), cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        // cv::line(merged, cv::Point((ellipsevtx[0].x+ellipsevtx[3].x)/2,(ellipsevtx[0].y+ellipsevtx[3].y)/2), cv::Point((ellipsevtx[1].x+ellipsevtx[2].x)/2,(ellipsevtx[1].y+ellipsevtx[2].y)/2), cv::Scalar(255, 255, 255), 1, cv::LINE_AA);

        // put category text
        char catsStr[256];
        if (catsVector[0].size()>0) {
            sprintf(catsStr, "Euclidean: %s", catsVector[0][i].c_str());
            cv::putText(merged, catsStr, cv::Point(cX-60, cY-20), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
            sprintf(catsStr, "Naive Bayes: %s", catsVector[1][i].c_str());
            cv::putText(merged, catsStr, cv::Point(cX-60, cY-40), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
            sprintf(catsStr, "KNN: %s", catsVector[2][i].c_str());
            cv::putText(merged, catsStr, cv::Point(cX-60, cY-60), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);

        }
        // put feature text
        char featureStr[256];
        sprintf(featureStr, "Aspect Ratio: %.2f", feature[i][0]);
        cv::putText(merged, featureStr, cv::Point(cX-20, cY), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255));
        sprintf(featureStr, "Extent: %.2f", feature[i][1]);
        cv::putText(merged, featureStr, cv::Point(cX-20, cY+15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255));
        sprintf(featureStr, "Solidity: %.2f", feature[i][2]);
        cv::putText(merged, featureStr, cv::Point(cX-20, cY+30), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255));
    }

    return merged;

}
