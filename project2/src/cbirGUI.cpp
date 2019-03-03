/*
  cbirGUI.cpp
  Mike Zheng and Heidi He
  CS365 project 2
  2/28/19

  Content-based image retrieval
  GUI

  to compile: make cbirGUI
  to run: ../bin/cbirgui ../data/MacbethChart.jpg ../../../olympus

  display multiple image based on:
  https://github.com/opencv/opencv/wiki/DisplayManyImages
*/

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include "opencv2/opencv.hpp"

#include "img.hpp"
#include "hist.hpp"

char **readDB(char *dir, int *num);
void readDB_rec(char *dir, char ***fileArr, int *max, int *numFile);
int imgComparator(const void* p1, const void* p2);
void display(char *query, Img **dispImgArr);
Img **update(char *query, int numFile, Img **imgArr, Img **dispImgArr, int method );

int main(int argc, char *argv[]) {

  char query[256];
  char database[256];
  int numResult = 20;
  int method;

  // usage
  if(argc < 3) {
    printf("Usage: %s <query> <database>\n", argv[0]);
    exit(-1);
  }

  // read arguments
  strcpy(query, argv[1]);
  strcpy(database, argv[2]);

  printf("CBIR: Querying %s \nfrom database %s\n\n", query, database);

  // recursively read all files from database directory
  char **fileArr;
  int numFile;
  fileArr = readDB(database, &numFile);
  // printf("There are %d files\n", numFile);

  // create the corresponding Img objects
  Img **imgArr = (Img **)malloc(sizeof(Img *)*numFile);
  for (int i = 0; i<numFile; i++) {
    // printf("%s\n", fileArr[i]);
    imgArr[i] = new Img(fileArr[i]);
  }
  free(fileArr);

  // create a window
	cv::namedWindow(query, 1);

  // an array of 20 best images to display
  Img **dispImgArr = (Img **)malloc(sizeof(Img *)*20);

  // trackbar params
  int method_slider = 0;
  int method_max = 3;
  cv::createTrackbar( "Method", query, &method_slider, method_max);

  // display the query and the matches
  dispImgArr = update(query, numFile, imgArr, dispImgArr, method_slider);
  display(query, dispImgArr);

  int key = cv::waitKey(0);
  while (1) {
    if (key == 65 || key == 97) {
      dispImgArr = update(query, numFile, imgArr, dispImgArr, method_slider);
      display(query, dispImgArr);
      key = cv::waitKey(0);
    }
    else {
      break;
    }
  }
  cv::destroyWindow(query);
  free(dispImgArr);

  return (0);
}

// apply the new cbir operation based on the trackbar values return the new 20
// best match array
Img **update(char *query, int numFile, Img **imgArr, Img **dispImgArr, int method ) {
  printf("Applying new cbir operation with method: %d\n", method);
  // for (int i = 0; i<numFile; i++) {
  //   imgArr[i]->printImgInfo();
  // }

  switch(method) {
    // case 0: run baseline matching - task1
    case(0):
      {
        // get the block of the query image
        cv::Mat queryImg, queryBlock;
        int halfBlockSize, queryMidLeft, queryMidUp;
        halfBlockSize = 2;
        queryImg = cv::imread(query);
        if(queryImg.data == NULL) {
          printf("Unable to read query image %s\n", query);
          exit(-1);
        }
        // printf("query image size: %d rows x %d columns\n", (int)queryImg.size().height, (int)queryImg.size().width);
        queryMidLeft = ((int)queryImg.size().width)/2-halfBlockSize;
        queryMidUp = ((int)queryImg.size().height)/2-halfBlockSize;
        // printf("query image mid point %d,%d\n", queryMidLeft+2, queryMidUp+2);
        queryImg(cv::Rect(queryMidLeft,queryMidUp,2*halfBlockSize+1,2*halfBlockSize+1)).copyTo(queryBlock);

        //run baseline matching on the db
        for (int i = 0; i<numFile; i++) {
          imgArr[i]->baselineMatching(queryBlock, halfBlockSize);
        }
        break;
      }

    // case 1: calculate whole image hs histogram of the query image - task2
    case(1):
      {
        cv::Mat queryHist = hist_whole_hs(query);
        // run baseline histogram matching
        for (int i = 0; i<numFile; i++) {
          imgArr[i]->baselineHistogram(queryHist);
        }
        break;
      }

    // case 2: calculate multi histogram matching - task3
    case(2):{
      cv::Mat queryHist1 = multi_hist_whole_hs(query).first;
      cv::Mat queryHist2 = multi_hist_whole_hs(query).second;
      //loop through all images and run multi histogram matching
      for (int i = 0; i<numFile; i++) {
        imgArr[i]->multiHistogram(queryHist1, queryHist2);
      }
      break;

    }
    case(3):
      // calculate whole image histogram based on color and texture
      {
        std::vector<cv::Mat> queryTextureHists;
        queryTextureHists = hist_whole_texture_laws_subset(query);
        cv::Mat queryHSHist = hist_whole_hs(query);

        // run color texture histogram matching
        for (int i = 0; i<numFile; i++) {
          imgArr[i]->colorTextureHistogram(queryHSHist, queryTextureHists);
        }

        break;

      }

    default:
      printf("Invalid method\n");
      exit(-1);
  }

  // sort the resulting images based on similarity score
  qsort((void *)imgArr, numFile, sizeof(Img *), imgComparator);
  // for (int i = 0; i<std::min(numFile, numResult); i++) {
  //   imgArr[i]->printImgInfo();
  // }

  for (int i = 0; i<20; i++) {
    dispImgArr[i] = imgArr[i];
  }

  return dispImgArr;
}

// display the query image and the 20 best match images
void display(char *query, Img **dispImgArr) {
  // read query image
  cv::Mat queryImg;
  queryImg = cv::imread(query);
  if(queryImg.data == NULL) {
    printf("Unable to read image %s\n", query);
    exit(-1);
  }

  // Create a new 3 channel image
  cv::Mat dispImg = cv::Mat::zeros(cv::Size(1920, 1080), CV_8UC3);

  // scale and copy the query img to dispImg
  int x,y, max, posx, posy;
  int xcoords[] = {900, 1120, 1340, 1560};
  int ycoords[] = {20, 230, 440, 650, 860};
  float scale;
  cv::Mat temp;
  cv::Mat matchImg;
  // Find the width and height of the image
  x = queryImg.cols;
  y = queryImg.rows;
  // Find the scaling factor to resize the image
  if ((float)y/x>1.3) {
    scale = (float) ( (float) y / 1040 );
  } else {
    scale = (float) ( (float) x / 800 );
  }

  // Set the image ROI to display the current image
  // Resize the input image and copy the it to the Single Big Image
  cv::Rect ROI(20, (int)((1080-( y/scale ))/2), (int)( x/scale ), (int)( y/scale ));
  cv::resize(queryImg,temp, cv::Size(ROI.width, ROI.height));
  temp.copyTo(dispImg(ROI));

  // scale and copy the match img to dispImg
  for (int i = 0; i<20; i++) {
    // dispImgArr[i]->printImgInfo();
    matchImg = cv::imread(dispImgArr[i]->getPath());
    if(matchImg.data == NULL) {
      printf("Unable to read image %s\n", dispImgArr[i]->getPath());
      exit(-1);
    }
    x = matchImg.cols;
    y = matchImg.rows;
    // Find whether height or width is greater in order to resize the image
    max = (x > y)? x: y;
    // printf("max is %d\n", max);
    // Find the scaling factor to resize the image
    scale = (float) ( (float) max / 200 );
    // Used to Align the images
    posx = i % 4;
    posy = i / 4;
    // Set the image ROI to display the current image
    // Resize the input image and copy the it to the Single Big Image
    cv::Rect ROI(xcoords[posx], ycoords[posy], (int)( x/scale ), (int)( y/scale ));
    cv::resize(matchImg,temp, cv::Size(ROI.width, ROI.height));
    temp.copyTo(dispImg(ROI));

  }

  cv::imshow(query, dispImg);
}


/* get all file names of a given directory*/
char **readDB(char *dir, int *num) {
  int max = 16;
  int numFile = 0;
  char **fileArr = (char**)malloc(sizeof(char *)*max);
  readDB_rec(dir, &fileArr, &max, &numFile);
  *num = numFile;
  return fileArr;
}


/* helper function for readDB*/
void readDB_rec(char *dir, char ***fileArr, int *max, int *numFile) {
  DIR *dirp;
  struct dirent *dp;
  // printf("Accessing directory %s\n", dir);

  // open the directory
	dirp = opendir( dir );
	if( dirp == NULL ) {
		printf("Cannot open directory %s\n", dir);
		exit(-1);
	}
  // loop over the contents of the directory
	while( (dp = readdir(dirp)) != NULL ) {
    if (dp->d_name[0] != '.') {
      // printf("The array is %d/%d\n", *numFile, *max);
      char *path = (char *)malloc(256);
      strcpy(path, "");
      strcat(path, dir);
      //directory naming
      if (path[strlen(path)-1] != 47) {
        strcat(path, "/");
      }
      strcat(path, dp->d_name);
      // printf("path is now\n%s\n", path);
      if (dp->d_type == DT_DIR) {
        // printf("%s is a directory\n", path);
        readDB_rec(path, fileArr, max, numFile);
      }
      else if (dp->d_type == DT_REG) {
        // printf("%s is a file\n", path);
        // look for images
        if( strstr(dp->d_name, ".jpg") ||
            strstr(dp->d_name, ".JPG") ||
				    strstr(dp->d_name, ".png") ||
				    strstr(dp->d_name, ".ppm") ||
				    strstr(dp->d_name, ".tif") ) {
          // double the file array if necessary
          if (*numFile == *max) {
            // printf("Doubling the array\n");
            *max *= 2;
            // printf("New max is %d\n", *max);
            *fileArr = (char **)realloc(*fileArr, sizeof(char *)*(*max));
          }
  		    (*fileArr)[*numFile] = path;
          (*numFile)++;
        }
      }
    }
	}
	// close the directory
  closedir(dirp);
}

int imgComparator(const void* p1, const void* p2) {
  double img1Similarity = (*(Img **)p1)->getSimilarity();
  double img2Similarity = (*(Img **)p2)->getSimilarity();
  // printf("comparing %.2f and %.2f\n", img1Similarity, img2Similarity);
  if (img2Similarity-img1Similarity>0) {return 1;}
  else if (img1Similarity-img2Similarity>0) {return -1;}
  else {return 0;}
}
