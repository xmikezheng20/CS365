/*
	calib.cpp
	Mike Zheng and Heidi He
	CS365 project 4
	4/1/19

    camera calibration

    to compile:
        make calib
    to run:
        ../bin/calib
*/


#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>

// save the frame (update point_list, corner_list)
void saveframe(cv::Mat &frame, int &frameid, std::vector<cv::Point2f> &corner_set,
    cv::Size patternsize,
    std::vector<std::vector<cv::Point3f>> &point_list,
    std::vector<std::vector<cv::Point2f>> &corner_list) {

    corner_list.push_back(corner_set);
    std::vector<cv::Point3f> point_set;

    for (int i=0; i<patternsize.height; i++) {
        for (int j=0; j<patternsize.width; j++) {
            point_set.push_back(cv::Point3f(j, -i, 0));
        }
    }

    point_list.push_back(point_set);

    // // check point set and corner set
    // for (int i=0; i<patternsize.height; i++) {
    //     for (int j=0; j<patternsize.width; j++) {
    //         printf("point idx %d: screen: (%.2f, %.2f); world: (%.2f, %.2f, %.2f)\n", i*patternsize.width+j,
    //             corner_set[i*patternsize.width+j].x, corner_set[i*patternsize.width+j].y,
    //             point_set[i*patternsize.width+j].x, point_set[i*patternsize.width+j].y, point_set[i*patternsize.width+j].z);
    //     }
    // }

    // save frame as an image to data directory
    char buffer[256];
    std::vector<int> pars;
    sprintf(buffer, "../data/calib.%03d.png", frameid++);
    cv::imwrite(buffer, frame, pars);
    printf("Image written: %s\n", buffer);

}

// calibrate camera and write out results
void calibrate(std::vector<std::vector<cv::Point3f>> &point_list,
    std::vector<std::vector<cv::Point2f>> &corner_list, cv::Size &refS) {

    double error;
    double cameraMatInit[9] = {1,0,(double)refS.width/2, 0,1,(double)refS.height/2, 0,0,1};
    cv::Mat cameraMat = cv::Mat(3,3,CV_64FC1, cameraMatInit);
    cv::Mat distCoeffs;
    std::vector<cv::Mat> rvecs, tvecs;
    error = cv::calibrateCamera(point_list, corner_list, refS,
        cameraMat, distCoeffs, rvecs, tvecs,
        cv::CALIB_FIX_ASPECT_RATIO);
    std::cout<<"camera matrix:\n"<<cameraMat<<std::endl;
    std::cout<<"error:\n"<<error<<std::endl;
    // write to file
    std::ofstream file;
    file.open ("../data/params.txt");
    file << "CameraMatrix\n"<<cameraMat<<"\ndistCoeffs\n"
        <<distCoeffs<<"\nrvecs\n";
    for (int i=0;i<rvecs.size();i++){
        file<<rvecs[i]<<"\n";
    }
    file<<"tvecs\n";
    for (int i=0;i<tvecs.size();i++){
        file<<tvecs[i]<<"\n";
    }
    file<<"error\n"<<error;
    file.close();

}

int main(int argc, char *argv[]) {

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
    bool patternfound = false;

    std::vector<std::vector<cv::Point3f>> point_list;
    std::vector<std::vector<cv::Point2f>> corner_list;

    // start video capture
	for(;!quit;) {
		*capdev >> frame; // get a new frame from the camera, treat as a stream

		if( frame.empty() ) {
		  printf("frame is empty\n");
		  break;
		}

        // detect target and extract corners
        cv::Mat grey;

        // convert to gray scale
        cv::cvtColor(frame, grey, cv::COLOR_BGR2GRAY);

        // detect corners
        cv::Size patternsize(9,6); //interior number of corners
        std::vector<cv::Point2f> corner_set; //this will be filled by the detected corners

        //CALIB_CB_FAST_CHECK saves a lot of time on images
        //that do not contain any chessboard corners
        patternfound = cv::findChessboardCorners(grey, patternsize, corner_set,
                cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE
                + cv::CALIB_CB_FAST_CHECK);

        // std::cout<<patternfound<<std::endl;

        if(patternfound) {
            cv::cornerSubPix(grey, corner_set, cv::Size(11, 11), cv::Size(-1, -1),
                cv::TermCriteria(cv::TermCriteria::EPS+cv::TermCriteria::COUNT, 30, 0.1));

            // printf("The first corner is (%.2f,%.2f)\n", corner_set[0].x,corner_set[0].y);
        }

        cv::drawChessboardCorners(frame, patternsize, cv::Mat(corner_set), patternfound);


		cv::imshow("Video", frame);

		int key = cv::waitKey(10);

        switch(key) {
        case 'q':
            quit = 1;
            break;
        case 's':
            if (patternfound){
                // s to save frame for calibration (only if pattern is detected)
                saveframe(frame, frameid, corner_set, patternsize, point_list, corner_list);
            }
            break;
        case 'c':
            // c to calibrate if has 5 or more calibration images
            // calibrate the camera and write out camera matrix, distortion coefficients,
            // rotation, translation, and error
            if (frameid>4) {
                calibrate(point_list, corner_list, refS);
            } else {
                printf("Need at least 5 images; currently %d\n", frameid);
            }

            break;
        case 'r':
            // r to reset the frames that have been captured
            printf("Resetting\n");
            frameid = 0;
            point_list.clear();
            corner_list.clear();
            break;

        default:
            break;
        }

    }

    // terminate the video capture
    printf("Terminating\n");
    delete capdev;


    return(0);
}
