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

int main(int argc, char *argv[]) {

    // usage
    if( argc < 1 ) {
        printf("Usage: %s \n", argv[0]);
        exit(-1);
    }

    cv::VideoCapture *capdev;
    int quit = 0;

    // open the video device
    capdev = new cv::VideoCapture(2); //default 0 for using webcam
    if( !capdev->isOpened() ) {
        printf("Unable to open video device\n");
        return(-1);
    }

    cv::Size refS( (int) capdev->get(cv::CAP_PROP_FRAME_WIDTH ),
               (int) capdev->get(cv::CAP_PROP_FRAME_HEIGHT));

    printf("Expected size: %d %d\n", refS.width, refS.height);

    cv::namedWindow("Video", 1);
	cv::Mat frame;

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
        bool patternfound = cv::findChessboardCorners(grey, patternsize, corner_set,
                cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE
                + cv::CALIB_CB_FAST_CHECK);

        // std::cout<<patternfound<<std::endl;

        if(patternfound) {
            cv::cornerSubPix(grey, corner_set, cv::Size(11, 11), cv::Size(-1, -1),
                cv::TermCriteria(cv::TermCriteria::EPS+cv::TermCriteria::COUNT, 30, 0.1));

            // printf("The first corner is (%.2f,%.2f)\n", corner_set[0].x,corner_set[0].y);
        }

        cv::drawChessboardCorners(frame, patternsize, cv::Mat(corner_set), patternfound);

        //




		cv::imshow("Video", frame);

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
    delete capdev;


    return(0);
}
