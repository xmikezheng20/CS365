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

class Img
{
private:
  char *path;
  int status; // 1: done; 0 not done.
  int similarity;

public:
  // constructor
  Img(char *newPath);

  // getters and setters
  char *getPath();
  void setPath(char *newPath);

  int getStatus();
  void setStatus(int newStatus);

  int getSimilarity();
  void setSimilarity(int newSimilarity);

  // print
  void printImgInfo();

  // cbir methods
  void baselineMatching(char *query);

};
