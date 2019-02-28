/*
  img.hpp
  Mike Zheng and Heidi He
  CS365 project 2
  2/16/19

  img class holds information about an image in the image database
  may compare with query image and store the similarity score

*/

#include <cstdio>
#include <cstring>
#include "opencv2/opencv.hpp"

class Img
{
private:
  char *path;
  int status; // 1: done; 0 not done.
  double similarity;

public:
  // constructor
  Img(char *newPath);

  // getters and setters
  char *getPath();
  void setPath(char *newPath);

  int getStatus();
  void setStatus(int newStatus);

  double getSimilarity();
  void setSimilarity(double newSimilarity);

  // print
  void printImgInfo();

  // cbir methods
  void baselineMatching(cv::Mat queryBlock, int halfBlockSize);
  void baselineHistogram(cv::Mat queryHist);

  void colorTextureHistogram(std::vector<cv::Mat> queryHists);

  // destructor
  ~Img();

};
