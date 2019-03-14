/*
	classifierTest.cpp
    A test file for the classifier class

    to compile: make classifierTest
    to run: ../bin/classifierTest <testIdx>

	Mike Zheng and Heidi He
	CS365 project 3
	3/11/19
*/

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "classifier.hpp"

// read objDB to std::vector<std::vector<double>>
std::vector<std::vector<double>> readObjDB(char *path);

// test Scaled Euclidean classifier
void test0();

int main(int argc, char *argv[]) {

    // usage
	if( argc < 2 ) {
		printf("Usage: %s <testidx>\n", argv[0]);
		exit(-1);
	}

    int testIdx = atoi(argv[1]);

    switch(testIdx) {
        case 0:
            test0();
            break;
        default:
            printf("Unknown test idx, exiting\n");
    }


    return (0);
}

// read objDB to std::vector<std::vector<double>>
void readObjDB(char *path, std::vector<std::vector<double>> &objDBData,
	std::vector<int> &objDBCategory, std::map<std::string, int> &objDBDict ) {
	printf("reading %s\n", path);

	// read the file
    FILE *fp;
    fp = fopen(path, "r");
    if (fp==NULL) {
      printf("File not valid\n");
      exit(0);
    }

	std::vector<double> feature;
	int numCategory = 0;
	// parse the file
	char buf[256];
	fgets(buf, 256, fp);
	while (fgets(buf, 256, fp)!=NULL) {
		feature.clear();
		int idx = 0;
		// printf("%s", buf);
		// split
		char *pch;
		pch = strtok(buf,",");
		while (pch != NULL)
		{
			// printf("%s\n",pch);
			// strip
			// https://www.unix.com/programming/21264-how-trim-white-space-around-string-c-program.html
			char ptr[strlen(pch)+1];
			int i,j=0;
			for(i=0;pch[i]!='\0';i++)
			{
				if (pch[i] != ' ' && pch[i] != '\t' && pch[i] != '\n')
				ptr[j++]=pch[i];
			}
			ptr[j]='\0';
			pch=ptr;
			// store the training sample
			if (idx<3) {
				feature.push_back(atof(pch));
			}else {
				// if not exist
				if (objDBDict.count(pch)==0) {
					objDBDict[pch] = numCategory++;
				}
				objDBCategory.push_back(objDBDict[pch]);
			}
			idx++;

			pch = strtok(NULL,",");
		}
		objDBData.push_back(feature);
	}

}

// test Scaled Euclidean classifier
void test0() {
    printf("Testing Scaled Euclidean classifier\n");
    ScaledEuclidean euclideanClassifier = ScaledEuclidean();

	// get the training set
	char filename[] = "../data/objDB.csv";
	std::vector<std::vector<double>> objDBData;
	std::vector<int> objDBCategory;
	std::map<std::string, int> objDBCategoryDict;

	readObjDB(filename, objDBData, objDBCategory, objDBCategoryDict);

	// check feature
	for (int i=0; i<objDBData.size(); i++) {
		printf("%.2f, %.2f, %.2f\n", objDBData[i][0],objDBData[i][1],objDBData[i][2]);
	}

	// check category
	for (int i=0; i<objDBCategory.size(); i++) {
		printf("%d\n", objDBCategory[i]);
	}

	// check dict
	for(std::map<std::string, int>::value_type& x : objDBCategoryDict)
	{
	    std::cout << x.first << "," << x.second << std::endl;
	}



}
