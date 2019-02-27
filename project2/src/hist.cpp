/*
  hist.cpp
  Mike Zheng and Heidi He
  CS365 project 2
  2/26/19

  a library of histogram functions

*/

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "opencv2/opencv.hpp"

#include "hist.hpp"

/*create whole hue-saturation histogram*/
cv::Mat hist_whole_hs(char *path) {
  // printf("Calculating hs histogram of %s\n", path);
  cv::Mat src, hsv;

  // read the image
  src = cv::imread(path);
  if(src.data == NULL) {
    printf("Unable to read query image %s\n", path);
    exit(-1);
  }

  // cv::imshow(path, src);
  // cv::waitKey(0);

  // convert to hsv
  cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);

  // quantize the hue to 30 levels
  // saturation to 32 levels
  int hbins = 30, sbins = 32;
  int histSize[] = {hbins, sbins};
  // hue varies from 0 to 179
  float hranges[] = {0, 180};
  // saturation ranges from 0 to 255
  float sranges[] = {0, 256};
  const float* ranges[] = {hranges, sranges};

  cv::Mat hist;
  // channels 0 and 1
  int channels[] = {0,1};
  cv::calcHist( &hsv, 1, channels, cv::Mat(), hist, 2, histSize, ranges, true, false);
  // normalize the histogram
  cv::normalize( hist, hist, 0, 1, cv::NORM_MINMAX, -1, cv::Mat() );

  //draw histogram, could be commented out
  draw_hist(src, hist, hbins, sbins);

  return hist;
}

/*draw histogram given src, hue bins, saturation bins*/
void draw_hist(cv::Mat src, cv::Mat hist, int hbins, int sbins){
  double maxVal = 0;
  cv::minMaxLoc(hist, 0, &maxVal, 0, 0);
  
  // draw histogram
  int scale = 10;
  cv::Mat histImg = cv::Mat::zeros(sbins*scale, hbins*10, CV_8UC3);
  for( int h = 0; h < hbins; h++ )
      for( int s = 0; s < sbins; s++ )
      {
          float binVal = hist.at<float>(h, s);
          int intensity = cvRound(binVal*255/maxVal);
          cv::rectangle( histImg, cv::Point(h*scale, s*scale),
                      cv::Point( (h+1)*scale - 1, (s+1)*scale - 1),
                      cv::Scalar::all(intensity),
                      -1 );
      }
  cv::namedWindow( "Source", 1 );
  cv::imshow( "Source", src );
  cv::namedWindow( "H-S Histogram", 1 );
  cv::imshow( "H-S Histogram", histImg );
  cv::waitKey(0);
}
