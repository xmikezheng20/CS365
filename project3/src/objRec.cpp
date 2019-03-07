/*
	objRec.cpp
	Mike Zheng and Heidi He
	CS365 project 3
	3/6/19

	to compile: make objRec
	to run:
		for video: ../bin/objRec ../data/objDB.csv 0
		for still images: ../bin/objRec ../data/objDB.csv 1 ../../../training/

	based on

	Bruce A. Maxwell
	S19
	Simple example of video capture and manipulation
	Based on OpenCV tutorials

	Compile command (macos)

	clang++ -o vid -I /opt/local/include vidDisplay.cpp -L /opt/local/lib -lopencv_core -lopencv_highgui -lopencv_video -lopencv_videoio

	use the makefiles provided

	make vid

*/
#include <cstdio>
#include <cstring>
#include <dirent.h>
#include <opencv2/opencv.hpp>

#include "processing.hpp"

char **readDB(char *dir, int *num);
void readDB_rec(char *dir, char ***fileArr, int *max, int *numFile);


int main(int argc, char *argv[]) {

	char objDB[256];
	int mode; // 0: video; 1: still images

	// usage
	if( argc < 3 ) {
		printf("Usage: %s <objDB> 0 or %s <objDB> 1 <imgs>\n", argv[0], argv[0]);
		exit(-1);
	}

	strcpy(objDB, argv[1]);
	mode = atoi(argv[2]);

	printf("Object database path: %s\n", objDB);

	if (mode == 0) {
		printf("Video capture\n");

		cv::VideoCapture *capdev;

		// open the video device
		capdev = new cv::VideoCapture(0);
		if( !capdev->isOpened() ) {
			printf("Unable to open video device\n");
			return(-1);
		}

		cv::Size refS( (int) capdev->get(cv::CAP_PROP_FRAME_WIDTH ),
				   (int) capdev->get(cv::CAP_PROP_FRAME_HEIGHT));

		printf("Expected size: %d %d\n", refS.width, refS.height);

		cv::namedWindow("Video", 1); // identifies a window?
		cv::namedWindow("Processed", 1);
		cv::moveWindow("Processed", 200, 0);
		cv::Mat frame;
		cv::Mat thresholded, morphed;

		for(;;) {
			*capdev >> frame; // get a new frame from the camera, treat as a stream

			if( frame.empty() ) {
			  printf("frame is empty\n");
			  break;
			}

			cv::imshow("Video", frame);

			// threshold the image
			thresholded = threshold(frame);
			// apply morphological operations
			morphed = morphOps(thresholded);
			cv::imshow("Processed", morphed);

			if(cv::waitKey(10) == 'q') {
				break;
			}

		}

		delete capdev;

	}
	else if (mode == 1) {
		printf("Still images\n");
		char imgDir[256];
		strcpy(imgDir, argv[3]);

		// recursively read all files from database directory
		char **fileArr;
		int numFile;
		fileArr = readDB(imgDir, &numFile);
		// printf("There are %d files\n", numFile);

		// display the images
		cv::namedWindow("Image", 1);
		cv::namedWindow("Processed", 1);
		cv::moveWindow("Processed", 700, 0);
		cv::Mat img;
		cv::Mat thresholded, morphed;
		int key;

		for (int i=0; i<numFile; i++) {

			img = cv::imread(fileArr[i]);
			if(img.data == NULL) {
			  printf("Unable to read query image %s\n", fileArr[i]);
			  exit(-1);
			}

			cv::imshow("Image", img);

			// threshold the image
			thresholded = threshold(img);
			// apply morphological operations
			morphed = morphOps(thresholded);
			cv::imshow("Processed", morphed);


			key = cv::waitKey(0);
			// quit when pressed q
			if (key == 81 or key == 113) {
				break;
			}

		}

		free(fileArr);
	}

	// terminate the video capture
	printf("Terminating\n");

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
