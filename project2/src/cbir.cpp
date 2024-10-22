/*
  cbir.cpp
  Mike Zheng and Heidi He
  CS365 project 2
  2/16/19

  Content-based image retrieval

  to compile: make cbir
  to run: ../bin/cbir ../data/MacbethChart.jpg ../../../olympus 5 0
  to run: ../bin/cbir ../data/_DSC1159.jpg ../../../olympus 5 1
  to run: ../bin/cbir ../data/_DSC1159.jpg ../../../test 5 3

  task3: _DSC0435.jpg


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

int main(int argc, char *argv[]) {

    char query[256];
    char database[256];
    int numResult;
    int method;

    // usage
    if(argc < 5) {
    	printf("Usage: %s <query> <database> <number of results> <method>\n", argv[0]);
    	exit(-1);
    }

    // read arguments
    strcpy(query, argv[1]);
    strcpy(database, argv[2]);
    numResult = atoi(argv[3]);
    method = atoi(argv[4]);

    printf("CBIR: Querying %s \nfrom database %s\nusing method %d.\nDisplaying top %d results.\n\n", query, database, method, numResult);

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

    // for (int i = 0; i<numFile; i++) {
    //   imgArr[i]->printImgInfo();
    // }

    // run cbir
    cv::Mat queryHist;
    cv::Mat queryImg, queryBlock;
    int halfBlockSize, queryMidLeft, queryMidUp;

    cv::Vec3b *queryPixel;
    cv::Vec3b *queryPixel2;

    switch(method) {
    // case 0: run baseline matching - task1
    case(0):
      // get the block of the query image
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

    // case 1: calculate whole image hs histogram of the query image - task2
    case(1):
      queryHist = hist_whole_hs(query);
      // run baseline histogram matching
      for (int i = 0; i<numFile; i++) {
        imgArr[i]->baselineHistogram(queryHist);
      }
      break;

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
    // case 3 : sobel filters and color
    case(3):
        {
            cv::Mat queryTextureHist;
            queryTextureHist = hist_whole_texture_sobel(query);
            cv::Mat queryHSHist = hist_whole_hs(query);

            // run color texture histogram matching
            for (int i = 0; i<numFile; i++) {
              imgArr[i]->colorSobelHistogram(queryHSHist, queryTextureHist);
            }

            break;
        }
      //earth Mover's Distance
     case(4): {
          cv::Mat queryHist = hist_whole_hs(query);
          // run baseline histogram matching
          for (int i = 0; i<numFile; i++) {
            imgArr[i]->earthMoverDistance(queryHist);
          }
          break;
      }

      case(5):
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

        // case 6: calculate fourier texture and color
        case(6):
        {
            cv::Mat queryTextureHist;
            queryTextureHist = hist_whole_fourier(query);
            cv::Mat queryHSHist = hist_whole_hs(query);

            // run color texture histogram matching
            for (int i = 0; i<numFile; i++) {
              imgArr[i]->colorFourierHistogram(queryHSHist, queryTextureHist);
            }

            break;
        }


     case(7):{
         printf("getting whole rgbs histogram for query image \n");
         std::vector<cv::Mat> queryHists = hist_whole_rgbs(query);
         for (int i = 0; i<numFile; i++) {
             imgArr[i]->rgbsHistogram(queryHists);
        }
        break;
     }

    default:
      printf("Invalid method\n");
      exit(-1);
    }

    // sort the imgArr based on similarity score
    qsort((void *)imgArr, numFile, sizeof(Img *), imgComparator);
    for (int i = 0; i<std::min(numFile, numResult); i++) {
    imgArr[i]->printImgInfo();
    }

    //show result
    cv::Mat bestMatch;
    bestMatch = cv::imread(imgArr[0]->getPath());
    cv::imshow( "Best match 1", bestMatch);
    cv::waitKey(0);

    bestMatch = cv::imread(imgArr[1]->getPath());
    cv::imshow( "Best match 2", bestMatch);
    cv::waitKey(0);

    bestMatch = cv::imread(imgArr[2]->getPath());
    cv::imshow( "Best match 3", bestMatch);
    cv::waitKey(0);

    bestMatch = cv::imread(imgArr[3]->getPath());
    cv::imshow( "Best match 4", bestMatch);
    cv::waitKey(0);

    bestMatch = cv::imread(imgArr[4]->getPath());
    cv::imshow( "Best match 5", bestMatch);
    cv::waitKey(0);

    return(0);
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
