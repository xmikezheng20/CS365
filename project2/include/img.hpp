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

  /* cbir methods: */
  /*baseline histrogram that takes in a query image and an image database,
                matches the query image to each database image using a distance metric,
                then sorts the database images according to their similarity to the query images. */
  void baselineMatching(cv::Mat queryBlock, int halfBlockSize);
  /* uses a whole-image histogram to determine the similarity
  between the query image and the DB images. */
  void baselineHistogram(cv::Mat queryHist);
  /*multi histogram matching*/
  void multiHistogram(cv::Mat queryHist1, cv::Mat queryHist2);

  void colorTextureHistogram(cv::Mat queryColorHist, std::vector<cv::Mat> queryHists);

  void colorSobelHistogram(cv::Mat queryColorHist, cv::Mat querySobelHist);

  void earthMoverDistance(cv::Mat queryHist);

  // color + fourier texture histograms
  void colorFourierHistogram(cv::Mat queryColorHist, cv::Mat queryFourierHist);

  // destructor
  ~Img();

};
