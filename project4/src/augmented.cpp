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

struct Camera {
    cv::Mat cameraMat;
    cv::Mat distCoeffs;
};

// struct Pose {
//     cv::Mat tvec;
//     cv::Mat rvec;
// }


/*given object points, image point, camera matrix,
calculate camera position with rotation and translation vector*/
// std::pair<std::vector<cv::Mat>, std::vector<cv::Mat>> detect_pose(
Pose detect_pose(std::vector<std::vector<cv::Point3f>> &point_list,
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
