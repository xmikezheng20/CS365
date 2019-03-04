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

/* baseline histogram matching*/
/* uses HS-histogram on the full image*/
cv::Mat hist_whole_hs(char *path);

/*create whole hue-saturation histogram for an img*/
cv::Mat hist_whole_hs_img(cv::Mat src);

/*create multi whole hue-saturation histogram and return two*/
std::pair<cv::Mat,cv::Mat> multi_hist_whole_hs(char *path);

/*draw histogram given src, histogram, hue bins, saturation bins*/
void draw_hist_whole_hs(cv::Mat src, cv::Mat hist, int hbins, int sbins);

/* texture histogram of the whole image
 texture: apply multiple texture filters, aggregate 7*7 box to get energy,
 calculates energy histograms*/
std::vector<cv::Mat> hist_whole_texture_laws_subset(char *path);

/*create whole rgb + saturation histogram for a given path
* returns a list of 1-d histogram*/
std::vector<cv::Mat> hist_whole_rgbs(char *path);
