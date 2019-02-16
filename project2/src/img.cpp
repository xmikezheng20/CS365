/*
  img.cpp
  Mike Zheng and Heidi He
  CS365 project 2
  2/16/19

  img class holds information about an image in the image database
  may compare with query image and store the similarity score

*/

#include <cstdio>
#include <cstring>

#include "img.hpp"

// constructor
Img::Img(char *newPath) {
  this->path = newPath;
  this->status = 0;
  this->similarity = 0;
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
void Img::baselineMatching(char *query) {
  return;
}
