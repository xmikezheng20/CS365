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


// texture histogram of the whole image
// texture: apply multiple texture filters, aggregate 7*7 box to get energy,
// calculates energy histograms
std::vector<cv::Mat> hist_whole_texture_laws_subset(char *path);
