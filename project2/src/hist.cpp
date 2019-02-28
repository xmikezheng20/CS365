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


  // double maxVal = 0;
  // cv::minMaxLoc(hist, 0, &maxVal, 0, 0);
  //
  // // draw histogram
  // int scale = 10;
  // cv::Mat histImg = cv::Mat::zeros(sbins*scale, hbins*10, CV_8UC3);
  // for( int h = 0; h < hbins; h++ )
  //     for( int s = 0; s < sbins; s++ )
  //     {
  //         float binVal = hist.at<float>(h, s);
  //         int intensity = cvRound(binVal*255/maxVal);
  //         cv::rectangle( histImg, cv::Point(h*scale, s*scale),
  //                     cv::Point( (h+1)*scale - 1, (s+1)*scale - 1),
  //                     cv::Scalar::all(intensity),
  //                     -1 );
  //     }
  // cv::namedWindow( "Source", 1 );
  // cv::imshow( "Source", src );
  // cv::namedWindow( "H-S Histogram", 1 );
  // cv::imshow( "H-S Histogram", histImg );
  // cv::waitKey(0);

  return hist;
}





// color and texture histogram of the whole image
// color: HS-histogram
// texture: apply multiple texture filters, aggregate 7*7 box to get energy,
// calculates energy histograms
std::vector<cv::Mat> hist_whole_texture_laws_subset(char *path) {
  // printf("Calculating color texture histogram of %s\n", path);
  std::vector<cv::Mat> hists;

  cv::Mat src, src_gray, filtered, l5l5Response, filtered_abs, energy, hist;

  int histSize = 50;
  float range[] = { 0, 256 }; //the upper boundary is exclusive
  const float* histRange = { range };

  // read the image
  src = cv::imread(path);
  if(src.data == NULL) {
    printf("Unable to read query image %s\n", path);
    exit(-1);
  }
  // convert to grayscale
  cv::cvtColor( src, src_gray, cv::COLOR_BGR2GRAY );
  // src_gray.convertTo(src_gray, CV_32F);

  // laws filters
  // // l5l5 to reduce illumination
  // float l5l5_data[25] = {1, 4, 6, 4, 1,
  //                        4, 16, 24, 16, 4,
  //                        6, 24, 36, 24, 6,
  //                        4, 16, 24, 16, 4,
  //                        1, 4, 6, 4, 1};
  // cv::Mat l5l5 = cv::Mat(5, 5, CV_32F, l5l5_data);
  // // l5l5 /= 256;
  //
  // cv::filter2D(src_gray, l5l5Response, -1, l5l5, cv::Point(-1, -1), 0,
  //              cv::BORDER_DEFAULT);


  // e5l5
  float e5l5_data[25] = {-1, -4, -6, -4, -1,
                         -2, -8, -12, -8, -2,
                         0, 0, 0, 0, 0,
                         2, 8, 12, 8, 2,
                         1, 4, 6, 4, 1};
  cv::Mat e5l5 = cv::Mat(5, 5, CV_32F, e5l5_data);
  cv::filter2D(src_gray, filtered, -1, e5l5, cv::Point(-1, -1), 0,
               cv::BORDER_DEFAULT);

  // // normalize by l5l5 response
  // cv::divide(filtered, l5l5Response, filtered);
  //
  // average absolute values in 7*7 block to get energy
  filtered_abs = cv::abs(filtered);
  cv::blur(filtered_abs, energy, cv::Size(7, 7));
  //
  // calculate histogram
  cv::calcHist( &energy, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, true, false);
  cv::normalize( hist, hist, 0, 1, cv::NORM_MINMAX, -1, cv::Mat() );

  hists.push_back(hist.clone());


  // // l5e5
  float l5e5_data[25] = {-1, -2, 0, 2, 1,
                        -4, -8, 0, 8, 4,
                        -6, -12, 0, 12, 6,
                        -4, -8, 0, 8, 4,
                        -1, -2, 0, 2, 1};
  cv::Mat l5e5 = cv::Mat(5, 5, CV_32F, l5e5_data);

  cv::filter2D(src_gray, filtered, -1, l5e5, cv::Point(-1, -1), 0,
              cv::BORDER_DEFAULT);

  // normalize by l5l5 response
  // cv::divide(filtered, l5l5Response, filtered);

  // average absolute values in 7*7 block to get energy
  filtered_abs = cv::abs(filtered);
  cv::blur(filtered_abs, energy, cv::Size(7, 7));

  // calculate histogram
  cv::calcHist( &energy, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, true, false);
  cv::normalize( hist, hist, 0, 1, cv::NORM_MINMAX, -1, cv::Mat() );

  hists.push_back(hist.clone());
  //
  // std::cout << "hist0 is " << hists[0] <<std::endl;
  // std::cout << "hist1 is " << hists[1] <<std::endl;
  //
  // cv::namedWindow( "Source_gray", 1 );
  // cv::imshow( "Source_gray", src_gray );
  // cv::namedWindow( "Filtered", 1 );
  // cv::imshow( "Filtered", energy );
  // cv::waitKey(0);


  return hists;

}
