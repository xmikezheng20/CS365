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

void readDB(char *dir);

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
  readDB(database);


	return(0);
}

// get all file names of a given directory
void readDB(char *dir) {

  DIR *dirp;
  struct dirent *dp;
  printf("Accessing directory %s\n", dir);

  // open the directory
	dirp = opendir( dir );
	if( dirp == NULL ) {
		printf("Cannot open directory %s\n", dir);
		exit(-1);
	}
  // loop over the contents of the directory
	while( (dp = readdir(dirp)) != NULL ) {
      if (dp->d_name[0] != '.') {
          char path[256] = "";
          strcat(path, dir);
          strcat(path, "/");
          strcat(path, dp->d_name);
          if (dp->d_type == DT_DIR) {
            printf("%s is a directory\n", path);
            readDB(path);
          }
          else if (dp->d_type == DT_REG) {
  			    printf("%s is a file\n", path);
          }
      }
	}
	// close the directory
  closedir(dirp);
}
