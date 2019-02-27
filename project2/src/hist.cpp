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

/*create whole hue-saturation histogram for a given path*/
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
  //draw_hist(src, hist, hbins, sbins);

  return hist;
}

/*create whole hue-saturation histogram for an img*/
cv::Mat hist_whole_hs_img(cv::Mat src) {
  cv::Mat hsv;
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
  //draw_hist(src, hist, hbins, sbins);

  return hist;
}

/*create multi whole hue-saturation histogram and return two*/
std::pair<cv::Mat,cv::Mat> multi_hist_whole_hs(char *path){

  cv::Mat src, hsv;
  //divide the query image into blocks, 
  //compute histogram in each block and sum together as a whole histogram
  src = cv::imread(path);
  if(src.data == NULL) {
    printf("Unable to read query image %s\n", path);
    exit(-1);
  }
  // printf("query image size: %d rows x %d columns\n", (int)src.size().height, (int)src.size().width);

  //first take the vertical center 1/2 from the query image
  cv::Mat queryBlockCenter;
  src(cv::Rect(((int)src.size().width)/3,
              ((int)src.size().height)/3,
              ((int)src.size().width)/3,
              ((int)src.size().height)/3)).copyTo(queryBlockCenter);
  cv::Mat queryHistCenter = hist_whole_hs_img(queryBlockCenter); // first histogram for multi histogram input

  //then take the edges and corner of the image: 1/4 of the image in all directions
  //top
  cv::Mat queryBlockEdge1;
  src(cv::Rect(0, 0, (int)src.size().width, ((int)src.size().height)/4)).copyTo(queryBlockEdge1);
  cv::Mat queryHistEdge1 = hist_whole_hs_img(queryBlockEdge1);
  //left
  cv::Mat queryBlockEdge2;
  src(cv::Rect(0, 0, ((int)src.size().width)/4, ((int)src.size().height))).copyTo(queryBlockEdge2);
  cv::Mat queryHistEdge2 = hist_whole_hs_img(queryBlockEdge2);
  //bottom
  cv::Mat queryBlockEdge3;
  src(cv::Rect(0, ((int)src.size().height)*3/4, (int)src.size().width, ((int)src.size().height)/4)).copyTo(queryBlockEdge3);
  cv::Mat queryHistEdge3 = hist_whole_hs_img(queryBlockEdge3);
  //right
  cv::Mat queryBlockEdge4;
  src(cv::Rect((int)src.size().width*3/4, 0, (int)src.size().width/4, ((int)src.size().height))).copyTo(queryBlockEdge4);
  cv::Mat queryHistEdge4 = hist_whole_hs_img(queryBlockEdge4);

  cv::Mat queryHistEdge = queryHistEdge1 + queryHistEdge2 + queryHistEdge3 + queryHistEdge4;


  return std::make_pair(queryHistCenter, queryHistEdge);
}

/*draw histogram given src, histogram, hue bins, saturation bins*/
void draw_hist_whole_hs(cv::Mat src, cv::Mat hist, int hbins, int sbins){
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
