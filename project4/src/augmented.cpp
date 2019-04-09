/*
	calib.cpp
	Mike Zheng and Heidi He
	CS365 project 4
	4/8/19

    pose object

    to compile:
        make augmented
    to run:
        ../bin/augmented ../data/params.txt
*/

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>

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

    bool patternfound = false;

    // set to handle 9*6 checkerboard
    cv::Size patternsize(9,6); //interior number of corners
    std::vector<cv::Point3f> point_set;
    for (int i=0; i<patternsize.height; i++) {
        for (int j=0; j<patternsize.width; j++) {
            point_set.push_back(cv::Point3f(j, -i, 0));
        }
    }


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

            // // solve for pose
            cv::Mat rvec, tvec;
            bool solveSuccess = cv::solvePnP(point_set, corner_set,
                curCam.first, curCam.second, rvec, tvec);

            if (solveSuccess) {
                // std::cout<<"tvec "<<tvec<<std::endl;
                // std::cout<<"rvec "<<rvec<<std::endl;

                // draw xyz axes
                std::vector<cv::Point3f> obj_pts;
                std::vector<cv::Point2f> img_pts;
                obj_pts.push_back(cv::Point3f(0,0,0));
                obj_pts.push_back(cv::Point3f(1,0,0));
                obj_pts.push_back(cv::Point3f(0,1,0));
                obj_pts.push_back(cv::Point3f(0,0,1));
                cv::projectPoints(obj_pts, rvec, tvec, curCam.first, curCam.second, img_pts);

                cv::line(frame, img_pts[0], img_pts[1], cv::Scalar(255,0,0), 3); // x: blue
                cv::line(frame, img_pts[0], img_pts[2], cv::Scalar(0,255,0), 3); // y: green
                cv::line(frame, img_pts[0], img_pts[3], cv::Scalar(0,0,255), 3); // z: red

            }

        }

        cv::drawChessboardCorners(frame, patternsize, cv::Mat(corner_set), patternfound);


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
    cv::destroyWindow("Video");
    delete capdev;



    return (0);
}
