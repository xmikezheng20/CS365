/*
	ar_aruco.cpp
	Mike Zheng and Heidi He
	CS365 project 4
	4/13/19

    pose object

    to compile:
        make ar_aruco
    to run:
        ../bin/ar_aruco ../data/params.txt
*/

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <iostream>
#include <fstream>


#include "shape.hpp"

// read camera parameters from file
std::pair<cv::Mat, cv::Mat> read_params(char* filename){
    std::string line;
    std::ifstream myfile(filename);

    cv::Mat cameraMat, distCoeffs;

    if (myfile.is_open()){

        // while ( getline (myfile,line) ){
        //       std::cout<< line << '\n';
        // }
        double cameraMatVals[9];
        double distCoeffsVals[5];

        // handle first line
        getline(myfile, line);
        // handle second through fourth line -> camera Matrix
        std::string buf;
        for (int i=0;i<3;i++) {
            getline(myfile, line);
            std::stringstream tmpstream(line);
            for (int j=0;j<3;j++) {
                getline(tmpstream, buf, ',');
                if (buf.front() == '['){
                    // std::cout<<stod(buf.substr(1))<<std::endl;
                    cameraMatVals[i*3+j]=stod(buf.substr(1));
                } else {
                    // std::cout<<stod(buf)<<std::endl;
                    cameraMatVals[i*3+j]=stod(buf);
                }
            }
        }
        cameraMat = cv::Mat(3,3,CV_64FC1, cameraMatVals);
        // std::cout<<"cameraMat is :"<<cameraMat<<std::endl;

        //handle fifth line -> skip
        getline(myfile, line);
        //handle distortion coefficients
        getline(myfile, line);
        std::stringstream tmpstream(line);
        for (int j=0;j<5;j++) {
            getline(tmpstream, buf, ',');
            // std::cout<<buf<<std::endl;
            if (buf.front() == '['){
                // std::cout<<stod(buf.substr(1))<<std::endl;
                distCoeffsVals[j]=stod(buf.substr(1));
            } else {
                // std::cout<<stod(buf)<<std::endl;
                distCoeffsVals[j]=stod(buf);
            }
        }
        distCoeffs = cv::Mat(1,5,CV_64FC1, distCoeffsVals);
        // std::cout<<"distCoeffs is :"<<distCoeffs<<std::endl;

        myfile.close();
    }
    else{
        std::cout<< "Unable to open file";
        exit(-1);
    }
    cameraMat = cameraMat.clone();
    distCoeffs = distCoeffs.clone();

    // std::cout<<cameraMat<<std::endl;
    // std::cout<<distCoeffs<<std::endl;

    std::pair<cv::Mat, cv::Mat> curCam = std::make_pair(cameraMat, distCoeffs);

    return curCam;

}

int main(int argc, char *argv[]) {

    // usage
    if( argc < 2) {
        printf("Usage: %s <params.txt>\n", argv[0]);
        exit(-1);
    }

    char filename[256];
    strcpy(filename, argv[1]);


    std::pair<cv::Mat, cv::Mat> curCam;
    curCam = read_params(filename);
    // std::cout<<curCam.first<<std::endl;
    // std::cout<<curCam.second<<std::endl;

    cv::VideoCapture *capdev;
    int quit = 0;

    // open the video device
    capdev = new cv::VideoCapture(1); //default 0 for using webcam
    if( !capdev->isOpened() ) {
        printf("Unable to open video device\n");
        return(-1);
    }

    cv::Size refS( (int) capdev->get(cv::CAP_PROP_FRAME_WIDTH ),
               (int) capdev->get(cv::CAP_PROP_FRAME_HEIGHT));

    printf("Expected size: %d %d\n", refS.width, refS.height);

    cv::namedWindow("Video", 1);
    cv::Mat frame;

    bool patternfound = false;

    cv::Mat cameraMatrix, distCoeffs;
    cameraMatrix = curCam.first;
    distCoeffs = curCam.second;

    // set to handle 9*6 checkerboard
    cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
    cv::Ptr<cv::aruco::GridBoard> board = cv::aruco::GridBoard::create(5, 7, 0.04, 0.01, dictionary);

    // start video capture
	for(;!quit;) {
		*capdev >> frame; // get a new frame from the camera, treat as a stream

		if( frame.empty() ) {
		  printf("frame is empty\n");
		  break;
		}

        cv::Mat image, imageCopy;
        capdev->retrieve(image);
        image.copyTo(imageCopy);
        std::vector<int> ids;
        std::vector<std::vector<cv::Point2f> > corners;
        cv::aruco::detectMarkers(image, dictionary, corners, ids);
        // if at least one marker detected
        if (ids.size() > 0) {
            cv::aruco::drawDetectedMarkers(imageCopy, corners, ids);
            cv::Mat rvec, tvec;
            int valid = estimatePoseBoard(corners, ids, board, cameraMatrix, distCoeffs, rvec, tvec);
            // if at least one board marker detected
            if(valid > 0){
                drawCube(imageCopy, curCam, rvec, tvec, cv::Point3f(0.05,0.06,0), 0.1);
                drawDiamond(imageCopy, curCam, rvec, tvec, cv::Point3f(0.1,0.3,0), 0.2);

                cv::aruco::drawAxis(imageCopy, cameraMatrix, distCoeffs, rvec, tvec, 0.1);

            }
        }

        // cv::imshow("Video", frame);
		cv::imshow("Video", imageCopy); //

		int key = cv::waitKey(10);

        switch(key) {
        case 'q':
            quit = 1;
            break;
        default:
            break;
        }

    }



    // terminate the video capture
    printf("Terminating\n");
    cv::destroyWindow("Video");
    delete capdev;



    return (0);
}
