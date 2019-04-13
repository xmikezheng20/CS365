/*
	calib.cpp
	Mike Zheng and Heidi He
	CS365 project 4
	4/10/19

    to compile:
        make ar_img
    to run:
        ../bin/ar_img ../data/ ../data/params.txt

    ar of static image
    image must be taken by the same camera
*/

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>

#include"shape.hpp"

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

// read images from a directory
std::vector<std::string> readDir(char *dirname) {

    DIR *dirp;
	struct dirent *dp;
    std::vector<std::string> imgNames;

	printf("Accessing directory %s\n", dirname);

	// open the directory
	dirp = opendir( dirname );
	if( dirp == NULL ) {
		printf("Cannot open directory %s\n", dirname);
		exit(-1);
	}

	// loop over the contents of the directory, looking for images
	while( (dp = readdir(dirp)) != NULL ) {
		if( strstr(dp->d_name, ".jpg") ||
				strstr(dp->d_name, ".png") ||
				strstr(dp->d_name, ".ppm") ||
				strstr(dp->d_name, ".tif") ) {

			// printf("image file: %s\n", dp->d_name);
            imgNames.push_back(std::string(dirname)+std::string(dp->d_name));

		}
	}

	// close the directory
	closedir(dirp);

    return imgNames;
}



int main(int argc, char *argv[]) {

    // usage
    if( argc < 3) {
        printf("Usage: %s <image directory> <params.txt>\n", argv[0]);
        exit(-1);
    }

    char filename[256], imgdir[256], imgname[256];
    strcpy(imgdir, argv[1]);
    strcpy(filename, argv[2]);

    // read the img names
    std::vector<std::string> imgNames;
    imgNames = readDir(imgdir);


    std::pair<cv::Mat, cv::Mat> curCam;
    curCam = read_params(filename);
    // std::cout<<curCam.first<<std::endl;
    // std::cout<<curCam.second<<std::endl;

    cv::namedWindow("Original", 1);
    cv::namedWindow("AR", 1);
    cv::moveWindow("AR", 640, 0);

    cv::Mat src, ar;


    for (int i=0; i<imgNames.size(); i++) {

        // read the image
        src = cv::imread(imgNames[i]);//

        // test if the read was successful
        if(src.data == NULL) {
            printf("Unable to read image %s\n", imgNames[i].c_str());
            exit(-1);
        }

        // ar
        printf("Processing image %s\n", imgNames[i].c_str());

        // set to handle 9*6 checkerboard
        cv::Size patternsize(9,6); //interior number of corners
        std::vector<cv::Point3f> point_set;
        for (int i=0; i<patternsize.height; i++) {
            for (int j=0; j<patternsize.width; j++) {
                point_set.push_back(cv::Point3f(j, -i, 0));
            }
        }

        cv::Mat grey;
        bool patternfound;

        // convert to gray scale
        cv::cvtColor(src, grey, cv::COLOR_BGR2GRAY);

        // detect corners
        std::vector<cv::Point2f> corner_set; //this will be filled by the detected corners

        //CALIB_CB_FAST_CHECK saves a lot of time on images
        //that do not contain any chessboard corners
        patternfound = cv::findChessboardCorners(grey, patternsize, corner_set,
                cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE
                + cv::CALIB_CB_FAST_CHECK);

        if(patternfound) {
            cv::cornerSubPix(grey, corner_set, cv::Size(11, 11), cv::Size(-1, -1),
                cv::TermCriteria(cv::TermCriteria::EPS+cv::TermCriteria::COUNT, 30, 0.1));

            // printf("The first corner is (%.2f,%.2f)\n", corner_set[0].x,corner_set[0].y);

            // // solve for pose
            cv::Mat rvec, tvec;
            bool solveSuccess = cv::solvePnP(point_set, corner_set,
                curCam.first, curCam.second, rvec, tvec);

            if (solveSuccess) {
                ar = src.clone();

                mask_target(ar, curCam, rvec, tvec, patternsize);

                // std::cout<<"tvec "<<tvec<<std::endl;
                // std::cout<<"rvec "<<rvec<<std::endl;
                drawAxes(ar, curCam, rvec, tvec);
                drawCube(ar, curCam, rvec, tvec, cv::Point3f(5,-2,0), 3);
                // drawPyramid(ar, curCam, rvec, tvec, cv::Point3f(5,-2,3), 3);
                printf("drawing reandom circles\n");
                for(int i=0; i<10; i++){
                    drawCircle(ar, curCam, rvec, tvec, cv::Point3f(i,-5,0),10*i);
                }
                drawHeart(ar, curCam, rvec, tvec, cv::Point3f(3,-1,0), 2);
                // drawCircle(ar, curCam, rvec, tvec, cv::Point3f(5,-5,0),10);
                drawDiamond(ar, curCam, rvec, tvec, cv::Point3f(5,-2,3), 3);
            }
        }

        // show the image in a window
        cv::imshow("Original", src);
        cv::imshow("AR", ar);


        // wait for a key press (indefinitely)
        cv::waitKey(0);
    }

    // get rid of the window
    cv::destroyWindow("Original");
    cv::destroyWindow("AR");

    // terminate the program
    printf("Terminating\n");


    return (0);
}
