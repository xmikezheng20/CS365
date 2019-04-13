/*
	shape.hpp
	Mike Zheng and Heidi He
	CS365 project 4
	4/9/19

    a library of functions that draw different shapes

*/

#include <cstdio>
#include <opencv2/opencv.hpp>



void drawAxes(cv::Mat& frame, std::pair<cv::Mat, cv::Mat> curCam, cv::Mat rvec, cv::Mat tvec);

void drawCube(cv::Mat& frame, std::pair<cv::Mat, cv::Mat> curCam, cv::Mat rvec, cv::Mat tvec, cv::Point3f pos, float size);

void drawPyramid(cv::Mat& frame, std::pair<cv::Mat, cv::Mat> curCam, cv::Mat rvec, cv::Mat tvec, cv::Point3f pos, float size);

void drawDiamond(cv::Mat& frame, std::pair<cv::Mat, cv::Mat> curCam, cv::Mat rvec, cv::Mat tvec, cv::Point3f pos, float size);

//draw circle
void drawCircle(cv::Mat& frame, std::pair<cv::Mat, cv::Mat> curCam, cv::Mat rvec, cv::Mat tvec, cv::Point3f pos, float size);

//draw heart
void drawHeart(cv::Mat& frame, std::pair<cv::Mat, cv::Mat> curCam, cv::Mat rvec, cv::Mat tvec, cv::Point3f pos, float size);


void mask_target(cv::Mat& frame, std::pair<cv::Mat, cv::Mat> curCam, cv::Mat rvec, cv::Mat tvec, cv::Size patternsize);
