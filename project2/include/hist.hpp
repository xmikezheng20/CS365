/*
  hist.hpp
  Mike Zheng and Heidi He
  CS365 project 2
  2/26/19

  a library of histogram functions

*/

#include <cstdio>
#include <cstring>
#include "opencv2/opencv.hpp"

// baseline histogram matching
// uses HS-histogram on the full image
cv::Mat hist_whole_hs(char *path);
