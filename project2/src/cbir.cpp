/*
  cbir.cpp
  Mike Zheng and Heidi He
  CS365 project 2
  2/16/19

  Content-based image retrieval

  to compile: make cbir
  to run: ../bin/cbir ../data/MacbethChart.jpg /Users/xiaoyuezheng/Documents/Colby/SP19/CS365/images 5 0

*/


#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include "opencv2/opencv.hpp"

#include "img.hpp"

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

  // run cbir baseline matching
  for (int i = 0; i<numFile; i++) {
    imgArr[i]->baselineMatching(query);
  }

  // sort the imgArr based on similarity score
  qsort((void *)imgArr, numFile, sizeof(Img *), imgComparator);
  for (int i = 0; i<std::min(numFile, numResult); i++) {
    imgArr[i]->printImgInfo();
  }


	return(0);
}

// get all file names of a given directory
char **readDB(char *dir, int *num) {
  int max = 16;
  int numFile = 0;
  char **fileArr = (char**)malloc(sizeof(char *)*max);
  readDB_rec(dir, &fileArr, &max, &numFile);
  *num = numFile;
  return fileArr;
}


// helper function for readDB
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
	// close the directory
  closedir(dirp);
}

int imgComparator(const void* p1, const void* p2) {
  int img1Similarity = (*(Img **)p1)->getSimilarity();
  int img2Similarity = (*(Img **)p2)->getSimilarity();
  // printf("comparing %d and %d\n", img1Similarity, img2Similarity);
  return img1Similarity-img2Similarity;
}
