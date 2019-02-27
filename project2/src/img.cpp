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
  this->status = 0; // 0 -> image unchecked; 1 -> image checked
  this->similarity = 0; //the larger similarity is, the closer the images are
  // printf("The path is %s\n",this->path);
}

/* getters and setters*/
/* get image path (address) */
char *Img::getPath() {
  return this->path;
}

/* set image path */
void Img::setPath(char *newPath) {
  this->path = newPath;
}

/* get status: 0 -> image unchecked; 1 -> image checked*/
int Img::getStatus() {
  return this->status;
}

/*get image status, to see if it is checked*/
void Img::setStatus(int newStatus) {
  this->status = newStatus;
}

/* return similarity, the larger similarity is, the closer the images are*/
int Img::getSimilarity() {
  return this->similarity;
}

/* set similarity*/
void Img::setSimilarity(int newSimilarity) {
  this->similarity = newSimilarity;
}

// print image info
void Img::printImgInfo() {
  printf("Image: %s\nStatus: %d\nSimilarity: %d\n\n", this->path, this->status, this->similarity);
}


/*baseline histrogram that takes in a query image and an image database, 
                matches the query image to each database image using a distance metric, 
                then sorts the database images according to their similarity to the query images. */
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

/* uses a whole-image histogram to determine the similarity 
between the query image and the DB images. */ 
void Img::baselineHistogram(cv::Mat queryHist) {
  printf("Baseline Histogram Matching with %s\n", this->path);
  cv::Mat targetHist;
  targetHist = hist_whole_hs(this->path);
  // use intersection distance
  this->similarity = cv::compareHist(queryHist, targetHist, cv::HISTCMP_INTERSECT);

}

/*multi histogram matching*/
void Img::multiHistogram(cv::Mat queryHist1, cv::Mat queryHist2){
  printf("Multiple Histogram Matching with %s\n", this->path);
  cv::Mat targetHist1 = multi_hist_whole_hs(this->path).first;
  cv::Mat targetHist2 = multi_hist_whole_hs(this->path).second;
  //use chi-square comparison
  int similarity1 = cv::compareHist(queryHist1, targetHist1, CV_COMP_CHISQR);
  int similarity2 = cv::compareHist(queryHist2, targetHist2, CV_COMP_CHISQR);

  this->similarity = similarity1*2 + similarity2; //so far, arbituary weight

}

// destructor
Img::~Img(){
  free(this->path);
}
