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
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <opencv2/opencv.hpp>

#include "processing.hpp"
#include "classifier.hpp"

char **readDB(char *dir, int *num);
void readDB_rec(char *dir, char ***fileArr, int *max, int *numFile);
void writeDB(char *filename, char *cat, std::vector<double> feature);
int createDB(char *filename);


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
	// if db not exist, create one
	int exist = createDB(objDB);

	// read training data
	std::vector<std::vector<double>> objDBData;
	std::vector<int> objDBCategory;
	std::map<std::string, int> objDBCategoryDict;

	// build the scaled euclidean classifier
	ScaledEuclidean euclideanClassifier = ScaledEuclidean();

	// build naive bayes classifier
	NaiveBayes naiveBayesClassifier = NaiveBayes();

	// build KNN classifier
	KNN knnClassifier = KNN();

	// state: training (0) vs testing (1)
	int state;
	if (exist == 0) {
		state = 0;
	}
	else {
		state = 1;
		readObjDB(objDB, objDBData, objDBCategory, objDBCategoryDict);
		euclideanClassifier.build(objDBData, objDBCategory, objDBCategoryDict);
		naiveBayesClassifier.build(objDBData, objDBCategory, objDBCategoryDict);
		knnClassifier.build(objDBData, objDBCategory, objDBCategoryDict, 7);
		printf("finished building\n");
	}


	if (mode == 0) {
		if(exist == 0){
			printf("Error: no database\n" );
			exit(0);
		}
		state = 1;
		printf("Video capture\n");

		cv::VideoCapture *capdev;

		// open the video device
		capdev = new cv::VideoCapture(1); //default 0 for using webcam
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
		cv::Mat thresholded, morphed, labeled, labeledVis;
		int numLabels;
		cv::Mat region;
		cv::Mat contoursVis;
		std::vector<std::vector<cv::Point>> contoursVector;
		std::vector<cv::Vec4i> hierarchyVector;
		std::vector<int> skipLabels;
		std::vector<std::vector<double>> completeFeatureArray;
		std::vector<int> classCatsArray;
		std::vector<std::string> euclideanCatsVector, naiveBayesCatsVector, knnCatsVector;
		std::vector<std::vector<std::string>> catsVector;
		std::vector<std::vector<double>> featureVector;

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
			// connected component analysis
			labeled = cv::Mat(morphed.size(), CV_32S);
			numLabels = cv::connectedComponents(morphed, labeled, 8);
			// visualize connected component analysis
			skipLabels.clear();
			labeledVis = visConnectedComponents(labeled, numLabels, skipLabels);
			// contour based metrics
			// separate each region, if region is too small, then discard it
			// otherwise, calculate features and visualize
			contoursVector.clear();
			hierarchyVector.clear();
			euclideanCatsVector.clear();
			naiveBayesCatsVector.clear();
			knnCatsVector.clear();
			catsVector.clear();
			featureVector.clear();
			if (numLabels>1) {
				for (int j=1; j<numLabels; j++) {
					// handle each region indivually to make index consistent
					region = extractRegion(labeled, j);
					// find contour of region, discard small region, extract features
					std::vector<double> feature;
					std::vector<std::vector<cv::Point>> contours;
  					std::vector<cv::Vec4i> hierarchy;
					int featureStatus = extractFeature(region, j, contours, hierarchy, feature);
					// status 0: valid region; status 1: discard
					if (featureStatus == 0) {
						printf("Feature successfully extracted\n");
						contoursVector.push_back(contours[0]);
						hierarchyVector.push_back(hierarchy[0]);
						featureVector.push_back(feature);
						completeFeatureArray.push_back(feature);

						if (state == 1) {
							// classify!
							int euclideanCat, naiveBayesCat, knnCat;
							euclideanCat = euclideanClassifier.classify(feature);
							naiveBayesCat = naiveBayesClassifier.classify(feature);
							knnCat = knnClassifier.classify(feature);

							printf("Feature vector %.2f, %.2f, %.2f\n",feature[0],feature[1], feature[2]);
							printf("Category idx: Euclidean: %d; Naive Bayes: %d; knn: %d\n", euclideanCat, naiveBayesCat, knnCat);
							for(std::map<std::string, int>::value_type& x : euclideanClassifier.getObjDBDict())
							{
								if (x.second == euclideanCat) {
									printf("Category: Euclidean: %s\n", x.first.c_str());
									euclideanCatsVector.push_back(x.first.c_str());
								}
							}
							for(std::map<std::string, int>::value_type& x : naiveBayesClassifier.getObjDBDict())
							{
								if (x.second == naiveBayesCat) {
									printf("Category: Naive Bayes: %s\n", x.first.c_str());
									naiveBayesCatsVector.push_back(x.first.c_str());
								}
							}
							for(std::map<std::string, int>::value_type& x : knnClassifier.getObjDBDict())
							{
								if (x.second == knnCat) {
									printf("Category: KNN: %s\n", x.first.c_str());
									knnCatsVector.push_back(x.first.c_str());
								}
							}
						}

					} else {
						skipLabels.push_back(j);
					}
				}
				catsVector.push_back(euclideanCatsVector);
				catsVector.push_back(naiveBayesCatsVector);
				catsVector.push_back(knnCatsVector);
				contoursVis = visFeature(labeled, numLabels, skipLabels, contoursVector, hierarchyVector, featureVector, catsVector, state);
			}


			cv::imshow("Processed", contoursVis);

			int key = cv::waitKey(25);
			//q/Q for exit
			if(key == 81 or key == 113) {
				break;
			}


		}

		delete capdev;
	}


	//picture mode
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
		cv::Mat thresholded, morphed, labeled, labeledVis;
		int numLabels;
		int key;
		cv::Mat region;
		cv::Mat contoursVis;
		std::vector<std::vector<cv::Point>> contoursVector;
	    std::vector<cv::Vec4i> hierarchyVector;
		std::vector<int> skipLabels;
		std::vector<std::vector<double>> completeFeatureArray;
		std::vector<int> euclideanClassCatsArray, naiveBayesClassCatsArray, knnClassCatsArray;
		std::vector<std::string> euclideanCatsVector, naiveBayesCatsVector, knnCatsVector;
		std::vector<std::vector<std::string>> catsVector;
		std::vector<std::vector<double>> featureVector;
		std::vector<int> trueCatsArray;

		for (int i=0; i<numFile; i++) {
			printf("state: %d\n", state);

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
			// connected component analysis
			labeled = cv::Mat(morphed.size(), CV_32S);
			numLabels = cv::connectedComponents(morphed, labeled, 8);
			// visualize connected component analysis
			skipLabels.clear();
			labeledVis = visConnectedComponents(labeled, numLabels, skipLabels);
			// contour based metrics
			// separate each region, if region is too small, then discard it
			// otherwise, calculate features and visualize
			contoursVector.clear();
			hierarchyVector.clear();
			euclideanCatsVector.clear();
			naiveBayesCatsVector.clear();
			knnCatsVector.clear();
			catsVector.clear();
			featureVector.clear();
			if (numLabels>1) {
				int k=0;
				for (int j=1; j<numLabels; j++) {
					// handle each region indivually to make index consistent
					region = extractRegion(labeled, j);
					// find contour of region, discard small region, extract features
					std::vector<double> feature;
					std::vector<std::vector<cv::Point>> contours;
  					std::vector<cv::Vec4i> hierarchy;
					int featureStatus = extractFeature(region, j, contours, hierarchy, feature);
					// status 0: valid region; status 1: discard
					if (featureStatus == 0) {
						k++;
						printf("Feature successfully extracted\n");
						contoursVector.push_back(contours[0]);
						hierarchyVector.push_back(hierarchy[0]);
						featureVector.push_back(feature);
						completeFeatureArray.push_back(feature);

						// write this feature vector to database along with the category
						char tmp[256];
						strcpy(tmp, fileArr[i]);
						char *p = strrchr(tmp, '/');
						p++;
						char *q = strchr(p, '.');
						*q = '\0';

						if (state == 0) {
							// write this feature vector to database along with the category
							writeDB(objDB, p, feature);
						}

						else if (state == 1) {
							// classify!
							int euclideanCat, naiveBayesCat, knnCat;
							euclideanCat = euclideanClassifier.classify(feature);
							naiveBayesCat = naiveBayesClassifier.classify(feature);
							knnCat = knnClassifier.classify(feature);

							// only save the first category as the category of the image
							if (k==1){
								euclideanClassCatsArray.push_back(euclideanCat);
								naiveBayesClassCatsArray.push_back(naiveBayesCat);
								knnClassCatsArray.push_back(knnCat);
								trueCatsArray.push_back(euclideanClassifier.getObjDBDict()[p]);
								// printf("%d %d %d\n", euclideanClassifier.getObjDBDict()[p], euclideanCat, naiveBayesCat);
							}

							printf("Feature vector %.2f, %.2f, %.2f, %.2f, %.2f, %.2f\n",feature[0],feature[1], feature[2], feature[3], feature[4], feature[5]);
							printf("Category idx: Euclidean: %d; Naive Bayes: %d; knn: %d\n", euclideanCat, naiveBayesCat, knnCat);
							for(std::map<std::string, int>::value_type& x : euclideanClassifier.getObjDBDict())
							{
								if (x.second == euclideanCat) {
									printf("Category: Euclidean: %s\n", x.first.c_str());
									euclideanCatsVector.push_back(x.first.c_str());
								}
							}
							for(std::map<std::string, int>::value_type& x : naiveBayesClassifier.getObjDBDict())
							{
								if (x.second == naiveBayesCat) {
									printf("Category: Naive Bayes: %s\n", x.first.c_str());
									naiveBayesCatsVector.push_back(x.first.c_str());
								}
							}
							for(std::map<std::string, int>::value_type& x : knnClassifier.getObjDBDict())
							{
								// printf("DEBUG 1\n");
								if (x.second == knnCat) {
									printf("Category: KNN: %s\n", x.first.c_str());
									knnCatsVector.push_back(x.first.c_str());
								}
							}
						}

					} else {
						skipLabels.push_back(j);
					}
				}
				catsVector.push_back(euclideanCatsVector);
				catsVector.push_back(naiveBayesCatsVector);
				catsVector.push_back(knnCatsVector);
				contoursVis = visFeature(labeled, numLabels, skipLabels, contoursVector, hierarchyVector, featureVector, catsVector, state);
			}



			//thresholded, morphed, labeledVis, final: contoursVis
			cv::imshow("Processed", contoursVis);

			key = cv::waitKey(0);
			// quit when pressed q
			if (key == 81 or key == 113) {
				break;
			}
			// b to enter build mode
			else if (key == 66 or key == 98) {
				state = 0;
			}
			// c to enter classify mode
			else if (key == 67 or key == 99) {
				if (state == 0) {
					printf("Building new classifier\n");
					readObjDB(objDB, objDBData, objDBCategory, objDBCategoryDict);
					euclideanClassifier.build(objDBData, objDBCategory, objDBCategoryDict);
					naiveBayesClassifier.build(objDBData, objDBCategory, objDBCategoryDict);
					knnClassifier.build(objDBData, objDBCategory, objDBCategoryDict, 7);
				}
				state = 1;
			}

		}

		if (mode == 1) {
			// print out the confusion matrix
			printf("Classifier type 0: Euclidean; type 1: KNN; type 2: NBC\n");
			if (trueCatsArray.size()>0) {
				std::vector<std::vector<int>> euclidean_conf_mat = euclideanClassifier.confusion_matrix(trueCatsArray, euclideanClassCatsArray);
				euclideanClassifier.print_confusion_matrix(euclidean_conf_mat);
				std::vector<std::vector<int>> knn_conf_mat = knnClassifier.confusion_matrix(trueCatsArray, knnClassCatsArray);
				knnClassifier.print_confusion_matrix(knn_conf_mat);
				std::vector<std::vector<int>> nbc_conf_mat = naiveBayesClassifier.confusion_matrix(trueCatsArray, naiveBayesClassCatsArray);
				naiveBayesClassifier.print_confusion_matrix(nbc_conf_mat);
			}
		}



		// // print out the feature vectors of the image set
		// printf("aspectRatio, extent, solidity, class\n");
		// for (int i=0; i<completeFeatureArray.size();i++) {
		// 	char *p = strrchr(fileArr[i], '/');
		// 	p++;
		// 	char *q = strchr(p, '.');
		// 	*q = '\0';
		// 	printf("%.4f, %.4f, %.4f, %s\n", completeFeatureArray[i][0], completeFeatureArray[i][1], completeFeatureArray[i][2], p);
		// }

		free(fileArr);
	}

	// terminate the video capture
	printf("Terminating\n");

	return(0);
}

// write the training feature and cats to the feature db
void writeDB(char *filename, char *cat, std::vector<double> feature) {
	FILE *fp;
    fp = fopen(filename, "a");
    if (fp==NULL) {
      printf("File not valid\n");
      exit(0);
    }
	fprintf(fp, "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%s\n",feature[0],feature[1],feature[2],feature[3],feature[4],feature[5],cat);

	fclose(fp);

}

// check if db exist, if not, create one
int createDB(char *filename) {
	FILE *fp;
	// read the file
    if (!(fp=fopen(filename, "r"))) {
      printf("Creating database\n");
	  fp=fopen(filename, "w");
	  fprintf(fp, "aspectRatio, extent, solidity, hu0, hu1, hu2, class\n");
	  fclose(fp);
	  return 0;
    }
	return 1;

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
