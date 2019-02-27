/*
  img.cpp
  Mike Zheng and Heidi He
  CS365 project 2
  2/16/19

  img class holds information about an image in the image database
  may compare with query image and store the similarity score

*/

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "opencv2/opencv.hpp"

#include "img.hpp"
#include "hist.hpp"

// constructor
Img::Img(char *newPath) {
  this->path = newPath;
  this->status = 0;
  this->similarity = 0;
  // printf("The path is %s\n",this->path);
}

// getters and setters
char *Img::getPath() {
  return this->path;
}
void Img::setPath(char *newPath) {
  this->path = newPath;
}

int Img::getStatus() {
  return this->status;
}
void Img::setStatus(int newStatus) {
  this->status = newStatus;
}

int Img::getSimilarity() {
  return this->similarity;
}
void Img::setSimilarity(int newSimilarity) {
  this->similarity = newSimilarity;
}

// print
void Img::printImgInfo() {
  printf("Image: %s\nStatus: %d\nSimilarity: %d\n\n", this->path, this->status, this->similarity);
}

// cbir methods
void Img::baselineMatching(cv::Mat queryBlock, int halfBlockSize) {
  printf("Baseline Matching with %s\n", this->path);

  cv::Mat dataImg = cv::imread(this->path);
  if(dataImg.data == NULL) {
    printf("Unable to read data image %s\n", this->path);
    exit(-1);
  }
  // printf("data image size: %d rows x %d columns\n", (int)dataImg.size().height, (int)dataImg.size().width);
  int dataMidLeft = ((int)dataImg.size().width)/2-halfBlockSize;
  int dataMidUp = ((int)dataImg.size().height)/2-halfBlockSize;
  // printf("data image mid point %d,%d\n", dataMidLeft+2, dataMidUp+2);

  // calculate sum of square
  int ssd = 0;
  cv::Vec3b *queryPixel;
  cv::Vec3b *dataPixel;
  for (int i=0; i<halfBlockSize*2+1; i++) {
    for (int j=0; j<halfBlockSize*2+1; j++) {
      // printf("comparing query (%d,%d) vs. data (%d,%d)\n", queryMidLeft+i,queryMidUp+j,dataMidLeft+i,dataMidUp+j);
      queryPixel = &(queryBlock.at<cv::Vec3b>(j, i));
      dataPixel = &(dataImg.at<cv::Vec3b>(dataMidUp+j, dataMidLeft+i));
      ssd += (queryPixel->val[0]-dataPixel->val[0])*(queryPixel->val[0]-dataPixel->val[0])
            +(queryPixel->val[1]-dataPixel->val[1])*(queryPixel->val[1]-dataPixel->val[1])
            +(queryPixel->val[2]-dataPixel->val[2])*(queryPixel->val[2]-dataPixel->val[2]);
    }
  }
  // printf("ssd is %d\n",ssd);
  this->similarity = -ssd;
}

void Img::baselineHistogram(cv::Mat queryHist) {
  printf("Baseline Histogram Matching with %s\n", this->path);
  cv::Mat targetHist;
  targetHist = hist_whole_hs(this->path);
  // use intersection distance
  this->similarity = cv::compareHist(queryHist, targetHist, cv::HISTCMP_INTERSECT);

}

// destructor
Img::~Img() {
  free(this->path);
}
