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

    cv::arrowedLine(frame, img_pts[0], img_pts[1], cv::Scalar(99, 114, 137), 2); // x: blue
    cv::arrowedLine(frame, img_pts[0], img_pts[2], cv::Scalar(99, 137, 111), 2); // y: green
    cv::arrowedLine(frame, img_pts[0], img_pts[3], cv::Scalar(107, 99, 137), 2); // z: red

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
    cv::line(frame, img_pts[0], img_pts[1], cv::Scalar(63, 89, 219), 3);//63, 89, 219
    cv::line(frame, img_pts[0], img_pts[2], cv::Scalar(63, 89, 219), 3);
    cv::line(frame, img_pts[3], img_pts[1], cv::Scalar(63, 89, 219), 3);
    cv::line(frame, img_pts[3], img_pts[2], cv::Scalar(63, 89, 219), 3);
    cv::line(frame, img_pts[4], img_pts[5], cv::Scalar(63, 89, 219), 3);
    cv::line(frame, img_pts[4], img_pts[6], cv::Scalar(63, 89, 219), 3);
    cv::line(frame, img_pts[7], img_pts[5], cv::Scalar(63, 89, 219), 3);
    cv::line(frame, img_pts[7], img_pts[6], cv::Scalar(63, 89, 219), 3);
    cv::line(frame, img_pts[0], img_pts[4], cv::Scalar(63, 89, 219), 3);
    cv::line(frame, img_pts[1], img_pts[5], cv::Scalar(63, 89, 219), 3);
    cv::line(frame, img_pts[2], img_pts[6], cv::Scalar(63, 89, 219), 3);
    cv::line(frame, img_pts[3], img_pts[7], cv::Scalar(63, 89, 219), 3);
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
    cv::line(frame, img_pts[0], img_pts[1], cv::Scalar(106, 185, 232), 3); //106, 185, 232
    cv::line(frame, img_pts[0], img_pts[2], cv::Scalar(106, 185, 232), 3);
    cv::line(frame, img_pts[3], img_pts[1], cv::Scalar(106, 185, 232), 3);
    cv::line(frame, img_pts[3], img_pts[2], cv::Scalar(106, 185, 232), 3);
    cv::line(frame, img_pts[0], img_pts[4], cv::Scalar(106, 185, 232), 3);
    cv::line(frame, img_pts[1], img_pts[4], cv::Scalar(106, 185, 232), 3);
    cv::line(frame, img_pts[2], img_pts[4], cv::Scalar(106, 185, 232), 3);
    cv::line(frame, img_pts[3], img_pts[4], cv::Scalar(106, 185, 232), 3);
}


//draw circle
void drawCircle(cv::Mat& frame, std::pair<cv::Mat, cv::Mat> curCam, cv::Mat rvec, cv::Mat tvec, cv::Point3f pos, float size) {
    std::vector<cv::Point3f> obj_pts;
    std::vector<cv::Point2f> img_pts;
    float x = pos.x;
    float y = pos.y;
    float z = pos.z;

    obj_pts.push_back(cv::Point3f(x,y,z)); //0

    cv::projectPoints(obj_pts, rvec, tvec, curCam.first, curCam.second, img_pts);

    cv::circle(frame, img_pts[0], size,cv::Scalar(107, 99, 137),8); //107, 99, 137
    // printf("circle drawn\n");
}

// draw a cone/ diamond at location pos with size
void drawDiamond(cv::Mat& frame, std::pair<cv::Mat, cv::Mat> curCam, cv::Mat rvec, cv::Mat tvec, cv::Point3f pos, float size) {
    std::vector<cv::Point3f> obj_pts;
    std::vector<cv::Point2f> img_pts;
    float x = pos.x;
    float y = pos.y;
    float z = pos.z;

    float tan22 = 0.41421356;
    float convexDistance = (size/2) * tan22;
    // make points
    //0.41421356
    obj_pts.push_back(cv::Point3f(x,y,z)); //0
    obj_pts.push_back(cv::Point3f(x+size/2,y+convexDistance,z)); //1
    obj_pts.push_back(cv::Point3f(x+size,y,z)); //2
    obj_pts.push_back(cv::Point3f(x+size+convexDistance,y - size/2,z)); //3
    obj_pts.push_back(cv::Point3f(x+size,y-size,z)); //4
    obj_pts.push_back(cv::Point3f(x+size/2,y-size-convexDistance,z));//5
    obj_pts.push_back(cv::Point3f(x,y-size,z));//6
    obj_pts.push_back(cv::Point3f(x-convexDistance,y-size/2,z)); //7
    //top piont
    obj_pts.push_back(cv::Point3f(x+size/2,y-size/2,z+size));//8

    cv::projectPoints(obj_pts, rvec, tvec, curCam.first, curCam.second, img_pts);
    //draw bottom
    cv::Point vertices[8];
    for (int i=0;i<8;i++) {
        vertices[i] = img_pts[i];
    }

    cv::fillConvexPoly(frame, vertices, 8, cv::Scalar(145,126,183));

    // draw lines
    cv::line(frame, img_pts[0], img_pts[1], cv::Scalar(178, 105, 108), 3); // 145,126,183
    cv::line(frame, img_pts[1], img_pts[2], cv::Scalar(178, 105, 108), 3);
    cv::line(frame, img_pts[2], img_pts[3], cv::Scalar(178, 105, 108), 3);
    cv::line(frame, img_pts[3], img_pts[4], cv::Scalar(178, 105, 108), 3);
    cv::line(frame, img_pts[4], img_pts[5], cv::Scalar(178, 105, 108), 3);
    cv::line(frame, img_pts[5], img_pts[6], cv::Scalar(178, 105, 108), 3);
    cv::line(frame, img_pts[6], img_pts[7], cv::Scalar(178, 105, 108), 3);
    cv::line(frame, img_pts[7], img_pts[0], cv::Scalar(178, 105, 108), 3);
    //to top piont
    cv::line(frame, img_pts[0], img_pts[8], cv::Scalar(178, 105, 108), 3);
    cv::line(frame, img_pts[1], img_pts[8], cv::Scalar(178, 105, 108), 3);
    cv::line(frame, img_pts[2], img_pts[8], cv::Scalar(178, 105, 108), 3);
    cv::line(frame, img_pts[3], img_pts[8], cv::Scalar(178, 105, 108), 3);
    cv::line(frame, img_pts[4], img_pts[8], cv::Scalar(178, 105, 108), 3);
    cv::line(frame, img_pts[5], img_pts[8], cv::Scalar(178, 105, 108), 3);
    cv::line(frame, img_pts[6], img_pts[8], cv::Scalar(178, 105, 108), 3);
    cv::line(frame, img_pts[7], img_pts[8], cv::Scalar(178, 105, 108), 3);
}


// draw a heart at location pos with size
void drawHeart(cv::Mat& frame, std::pair<cv::Mat, cv::Mat> curCam, cv::Mat rvec, cv::Mat tvec, cv::Point3f pos, float size) {
    std::vector<cv::Point3f> obj_pts;
    std::vector<cv::Point2f> img_pts;
    float x = pos.x;
    float y = pos.y;
    float z = pos.z;

    //bottom
    obj_pts.push_back(cv::Point3f(x,y,z)); //0
    //the left half
    obj_pts.push_back(cv::Point3f(x-size/3,y+size/3,z)); //1 - go left
    obj_pts.push_back(cv::Point3f(x-2*size/3,y+size/3,z)); //2 - go left
    obj_pts.push_back(cv::Point3f(x-size,y,z)); //3 - go left
    obj_pts.push_back(cv::Point3f(x-size,y-size,z)); //4 - go left
    obj_pts.push_back(cv::Point3f(x,y-size*2,z)); //5 - go left
    //the right half
    obj_pts.push_back(cv::Point3f(x+size,y-size,z)); //6
    obj_pts.push_back(cv::Point3f(x+size,y,z)); //7
    obj_pts.push_back(cv::Point3f(x+2*size/3,y+size/3,z)); //8
    obj_pts.push_back(cv::Point3f(x+size/3,y+size/3,z)); //9

    //draw top
    z = z+3;
    obj_pts.push_back(cv::Point3f(x,y,z)); //0
    //the left half
    obj_pts.push_back(cv::Point3f(x-size/3,y+size/3,z)); //1 - go left
    obj_pts.push_back(cv::Point3f(x-2*size/3,y+size/3,z)); //2 - go left
    obj_pts.push_back(cv::Point3f(x-size,y,z)); //3 - go left
    obj_pts.push_back(cv::Point3f(x-size,y-size,z)); //4 - go left
    obj_pts.push_back(cv::Point3f(x,y-size*2,z)); //5 - go left
    //the right half
    obj_pts.push_back(cv::Point3f(x+size,y-size,z)); //6
    obj_pts.push_back(cv::Point3f(x+size,y,z)); //7
    obj_pts.push_back(cv::Point3f(x+2*size/3,y+size/3,z)); //8
    obj_pts.push_back(cv::Point3f(x+size/3,y+size/3,z)); //9


    cv::projectPoints(obj_pts, rvec, tvec, curCam.first, curCam.second, img_pts);

    //draw bottom
    cv::Point verticesBottom[10];
    cv::Point verticesTop[10];
    for (int i=0;i<20;i++) {
        if(i<10){
            verticesBottom[i] = img_pts[i];
        }else{
            verticesTop[i-10] = img_pts[i];
        }
    }
    //draw lines - bottom
    // watermelon: 106, 154, 232
    for (int i=0;i<10;i++) {
        if(i==9){
            cv::line(frame, img_pts[i], img_pts[i-9], cv::Scalar(106, 154, 232), 5);
        }else{
            cv::line(frame, img_pts[i], img_pts[i+1], cv::Scalar(106, 154, 232), 5);
        }
    }

    //draw lines - top
    for (int i=0;i<10;i++) {
        if(i==9){
            cv::line(frame, img_pts[i+10], img_pts[i+10-9], cv::Scalar(106, 154, 232), 5);
        }else{
            cv::line(frame, img_pts[i+10], img_pts[i+10+1], cv::Scalar(106, 154, 232), 5);
        }
    }


    //draw parellel lines
    for (int i=0;i<10;i++) {
        cv::line(frame, img_pts[i], img_pts[i+10], cv::Scalar(106, 154, 232), 2);
    }

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

    cv::fillConvexPoly(frame, vertices, 4, cv::Scalar(183,175,126 )); //183,175,126

}
