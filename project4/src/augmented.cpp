/*
	calib.cpp
	Mike Zheng and Heidi He
	CS365 project 4
	4/8/19

    pose object

    to compile:
        make augmented
    to run:
        ../bin/augmented
*/

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>

#include "calib.hpp"


/*given object points, image point, camera matrix,
calculate camera position with rotation and translation vector*/
// std::pair<std::vector<cv::Mat>, std::vector<cv::Mat>> detect_pose(
camera *detect_pose(std::vector<std::vector<cv::Point3f>> &point_list,
                    std::vector<std::vector<cv::Point2f>> &corner_list,
                    camera *thisCamera){


    // std::cout<<"point_list size is :"<<point_list.size()<<std::endl;
    // std::cout<<"camera Matrix is :"<< thisCamera->cameraMat<<std::endl;
    // std::cout<<"distortion coefficients is :"<< thisCamera->distCoeffs<<std::endl;

    std::vector<cv::Mat>  rvec, tvec;

    for(int i=0; i<point_list.size(); i++){
        cv::Mat curRvec;
        cv::Mat curTvec;
        cv::solvePnP(point_list[i], corner_list[i], thisCamera->cameraMat, thisCamera->distCoeffs, curRvec, curTvec);

        // std::cout<<"rotation vector is: "<< curRvec <<std::endl;
        // std::cout<<"translation vector is: "<< curRvec <<std::endl;

        rvec.push_back(curRvec);
        tvec.push_back(curTvec);
    }

    // for(int i=0; i<rvec.size(); i++){
    //     std::cout<<"rotation vector is: "<< rvec[i] <<std::endl;
    //     std::cout<<"translation vector is: "<< tvec[i] <<std::endl;
    // }
    thisCamera->rvec = rvec;
    thisCamera->tvec = tvec;

    return thisCamera;

}




int main(int argc, char *argv[]) {

    struct camera *curCamera = new camera();
    bool calibrated;

    // usage
    if( argc < 1 ) {
        printf("Usage: %s \n", argv[0]);
        exit(-1);
    }

    cv::VideoCapture *capdev;
    int quit = 0;

    // open the video device
    capdev = new cv::VideoCapture(0); //default 0 for using webcam
    if( !capdev->isOpened() ) {
        printf("Unable to open video device\n");
        return(-1);
    }

    cv::Size refS( (int) capdev->get(cv::CAP_PROP_FRAME_WIDTH ),
               (int) capdev->get(cv::CAP_PROP_FRAME_HEIGHT));

    printf("Expected size: %d %d\n", refS.width, refS.height);

    cv::namedWindow("Video", 1);
	cv::Mat frame;
    int frameid = 0;

    std::vector<std::vector<cv::Point3f>> point_list;
    std::vector<std::vector<cv::Point2f>> corner_list;

    // start video capture
	for(;!quit;) {
		*capdev >> frame; // get a new frame from the camera, treat as a stream

		if( frame.empty() ) {
		  printf("frame is empty\n");
		  break;
		}
        // std::pair<cv::Mat, cv::Mat> cameraInfo;
        // detect target and extract corners
        cv::Mat grey;
        // convert to gray scale
        cv::cvtColor(frame, grey, cv::COLOR_BGR2GRAY);
        // detect corners
        cv::Size patternsize(9,6); //interior number of corners
        std::vector<cv::Point2f> corner_set; //this will be filled by the detected corners

        //CALIB_CB_FAST_CHECK saves a lot of time on images
        //that do not contain any chessboard corners
        bool patternfound = cv::findChessboardCorners(grey, patternsize, corner_set,
                cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE
                + cv::CALIB_CB_FAST_CHECK);

        // std::cout<<patternfound<<std::endl;

        //if found checkboard
        if(patternfound) {
            cv::cornerSubPix(grey, corner_set, cv::Size(11, 11), cv::Size(-1, -1),
                cv::TermCriteria(cv::TermCriteria::EPS+cv::TermCriteria::COUNT, 30, 0.1));

            // printf("The first corner is (%.2f,%.2f)\n", corner_set[0].x,corner_set[0].y);
        }

        cv::drawChessboardCorners(frame, patternsize, cv::Mat(corner_set), patternfound);


        //if checkboard exist: grab the locations of the corners,
        //and then get the board's pose (rotation and translation).
        if(patternfound && calibrated){
            // std::cout<<"camera Matrix is :"<< curCamera->cameraMat<<std::endl;

            detect_pose(point_list, corner_list, curCamera);

        }

		cv::imshow("Video", frame);
        
        //keyboard interaction --------------------
		int key = cv::waitKey(10);

        switch(key) {
            case 'q':
                quit = 1;
                std::cout<<"q pressed"<<std::endl;
                break;
            case 'r':
                // r to reset the frames that have been captured
                printf("Resetting\n");
                frameid = 0;
                point_list.clear();
                corner_list.clear();
                break;
            case 'c':
                //auto calibrate camera
                saveframe(frame, frameid, corner_set, patternsize, point_list, corner_list);
                saveframe(frame, frameid, corner_set, patternsize, point_list, corner_list);
                saveframe(frame, frameid, corner_set, patternsize, point_list, corner_list);
                saveframe(frame, frameid, corner_set, patternsize, point_list, corner_list);
                saveframe(frame, frameid, corner_set, patternsize, point_list, corner_list);
                if (frameid>4) {
                    *curCamera = calibrate(point_list, corner_list, refS);
                    printf("Calibrated after pressing 't'\n");
                    calibrated = 1;
                    std::cout<<"camera Matrix is :\n"<< curCamera->cameraMat<<std::endl;
                    std::cout<<"camera distortion coefficients is :\n"<< curCamera->distCoeffs<<std::endl;

                } else {
                    printf("Need at least 5 images; currently %d\n", frameid);
                }

            default:
                // std::cout<<"no keyboard triggered"<<std::endl;
                break;
        }

    }

    // terminate the video capture
    printf("Terminating\n");
    delete capdev;


    return(0);
}
