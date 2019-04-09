/*
	calib.hpp
    calibrate the camera

	Mike Zheng and Heidi He
	CS365 project 4
	4/8/19
*/


struct camera{
  cv::Mat cameraMat;
  cv::Mat distCoeffs;
  std::vector<cv::Mat>  rvec;
  std::vector<cv::Mat>  tvec;
};


// save the frame (update point_list, corner_list)
void saveframe(cv::Mat &frame, int &frameid, std::vector<cv::Point2f> &corner_set,
    cv::Size patternsize,
    std::vector<std::vector<cv::Point3f>> &point_list,
    std::vector<std::vector<cv::Point2f>> &corner_list);

// calibrate camera and write out results
camera calibrate(std::vector<std::vector<cv::Point3f>> &point_list,
    std::vector<std::vector<cv::Point2f>> &corner_list, cv::Size &refS);
