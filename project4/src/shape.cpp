/*
	shape.cpp
	Mike Zheng and Heidi He
	CS365 project 4
	4/9/19

    a library of functions that draw different shapes

*/

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <opencv2/opencv.hpp>

#include"shape.hpp"


// draw xyz axes
void drawAxes(cv::Mat& frame, std::pair<cv::Mat, cv::Mat> curCam, cv::Mat rvec, cv::Mat tvec) {

    // draw xyz axes
    std::vector<cv::Point3f> obj_pts;
    std::vector<cv::Point2f> img_pts;
    obj_pts.push_back(cv::Point3f(0,0,0));
    obj_pts.push_back(cv::Point3f(1,0,0));
    obj_pts.push_back(cv::Point3f(0,1,0));
    obj_pts.push_back(cv::Point3f(0,0,1));
    cv::projectPoints(obj_pts, rvec, tvec, curCam.first, curCam.second, img_pts);

    cv::arrowedLine(frame, img_pts[0], img_pts[1], cv::Scalar(255,0,0), 2); // x: blue
    cv::arrowedLine(frame, img_pts[0], img_pts[2], cv::Scalar(0,255,0), 2); // y: green
    cv::arrowedLine(frame, img_pts[0], img_pts[3], cv::Scalar(0,0,255), 2); // z: red

}

// draw a cube at location pos with size
void drawCube(cv::Mat& frame, std::pair<cv::Mat, cv::Mat> curCam, cv::Mat rvec, cv::Mat tvec, cv::Point3f pos, float size) {
    std::vector<cv::Point3f> obj_pts;
    std::vector<cv::Point2f> img_pts;
    float x = pos.x;
    float y = pos.y;
    float z = pos.z;

    // make points
    obj_pts.push_back(cv::Point3f(x,y,z));
    obj_pts.push_back(cv::Point3f(x+size,y,z));
    obj_pts.push_back(cv::Point3f(x,y-size,z));
    obj_pts.push_back(cv::Point3f(x+size,y-size,z));
    obj_pts.push_back(cv::Point3f(x,y,z+size));
    obj_pts.push_back(cv::Point3f(x+size,y,z+size));
    obj_pts.push_back(cv::Point3f(x,y-size,z+size));
    obj_pts.push_back(cv::Point3f(x+size,y-size,z+size));

    cv::projectPoints(obj_pts, rvec, tvec, curCam.first, curCam.second, img_pts);

    // draw lines
    cv::line(frame, img_pts[0], img_pts[1], cv::Scalar(0,0,255), 3);
    cv::line(frame, img_pts[0], img_pts[2], cv::Scalar(0,0,255), 3);
    cv::line(frame, img_pts[3], img_pts[1], cv::Scalar(0,0,255), 3);
    cv::line(frame, img_pts[3], img_pts[2], cv::Scalar(0,0,255), 3);
    cv::line(frame, img_pts[4], img_pts[5], cv::Scalar(0,0,255), 3);
    cv::line(frame, img_pts[4], img_pts[6], cv::Scalar(0,0,255), 3);
    cv::line(frame, img_pts[7], img_pts[5], cv::Scalar(0,0,255), 3);
    cv::line(frame, img_pts[7], img_pts[6], cv::Scalar(0,0,255), 3);
    cv::line(frame, img_pts[0], img_pts[4], cv::Scalar(0,0,255), 3);
    cv::line(frame, img_pts[1], img_pts[5], cv::Scalar(0,0,255), 3);
    cv::line(frame, img_pts[2], img_pts[6], cv::Scalar(0,0,255), 3);
    cv::line(frame, img_pts[3], img_pts[7], cv::Scalar(0,0,255), 3);
}

// draw a pyramid at location pos with size
void drawPyramid(cv::Mat& frame, std::pair<cv::Mat, cv::Mat> curCam, cv::Mat rvec, cv::Mat tvec, cv::Point3f pos, float size) {
    std::vector<cv::Point3f> obj_pts;
    std::vector<cv::Point2f> img_pts;
    float x = pos.x;
    float y = pos.y;
    float z = pos.z;

    // make points
    obj_pts.push_back(cv::Point3f(x,y,z));
    obj_pts.push_back(cv::Point3f(x+size,y,z));
    obj_pts.push_back(cv::Point3f(x,y-size,z));
    obj_pts.push_back(cv::Point3f(x+size,y-size,z));
    obj_pts.push_back(cv::Point3f(x+size/2,y-size/2,z+size));

    cv::projectPoints(obj_pts, rvec, tvec, curCam.first, curCam.second, img_pts);

    // draw lines
    cv::line(frame, img_pts[0], img_pts[1], cv::Scalar(255,0,0), 3);
    cv::line(frame, img_pts[0], img_pts[2], cv::Scalar(255,0,0), 3);
    cv::line(frame, img_pts[3], img_pts[1], cv::Scalar(255,0,0), 3);
    cv::line(frame, img_pts[3], img_pts[2], cv::Scalar(255,0,0), 3);
    cv::line(frame, img_pts[0], img_pts[4], cv::Scalar(255,0,0), 3);
    cv::line(frame, img_pts[1], img_pts[4], cv::Scalar(255,0,0), 3);
    cv::line(frame, img_pts[2], img_pts[4], cv::Scalar(255,0,0), 3);
    cv::line(frame, img_pts[3], img_pts[4], cv::Scalar(255,0,0), 3);
}


// change the checkerboard target to a gray background
// by drawing a rectangle over the checkerboard
void mask_target(cv::Mat& frame, std::pair<cv::Mat, cv::Mat> curCam, cv::Mat rvec, cv::Mat tvec, cv::Size patternsize) {

    std::vector<cv::Point3f> obj_pts;
    std::vector<cv::Point2f> img_pts;
    obj_pts.push_back(cv::Point3f(-1,1,0));
    obj_pts.push_back(cv::Point3f(-1,-patternsize.height,0));
    obj_pts.push_back(cv::Point3f(patternsize.width+1,-patternsize.height,0));
    obj_pts.push_back(cv::Point3f(patternsize.width+1,1,0));

    cv::projectPoints(obj_pts, rvec, tvec, curCam.first, curCam.second, img_pts);

    cv::Point vertices[4];
    for (int i=0;i<4;i++) {
        vertices[i] = img_pts[i];
    }

    cv::fillConvexPoly(frame, vertices, 4, cv::Scalar(100,100,100));

}
