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

// test Scaled Euclidean classifier
void test0() {
    printf("Testing Scaled Euclidean classifier\n");

	// get the training set
	char filename[] = "../data/objDB.csv";
	std::vector<std::vector<double>> objDBData;
	std::vector<int> objDBCategory;
	std::map<std::string, int> objDBCategoryDict;

	readObjDB(filename, objDBData, objDBCategory, objDBCategoryDict);

	// build the scaled euclidean classifier
	ScaledEuclidean euclideanClassifier = ScaledEuclidean();
	euclideanClassifier.build(objDBData, objDBCategory, objDBCategoryDict);
	// // check feature
	// for (int i=0; i<objDBData.size(); i++) {
	// 	printf("%.2f, %.2f, %.2f\n", objDBData[i][0],objDBData[i][1],objDBData[i][2]);
	// }
	//
	// // check category
	// for (int i=0; i<objDBCategory.size(); i++) {
	// 	printf("%d\n", objDBCategory[i]);
	// }
	//
	// // check dict
	// for(std::map<std::string, int>::value_type& x : objDBCategoryDict)
	// {
	//     std::cout << x.first << "," << x.second << std::endl;
	// }

	// make a test object feature vector
	std::vector<double> newObj;
	int cat;
	newObj.push_back(3.8);
	newObj.push_back(0.3);
	newObj.push_back(0.5);

	cat = euclideanClassifier.classify(newObj);

	printf("Feature vector %.2f, %.2f, %.2f\n",newObj[0],newObj[1], newObj[2]);
	printf("Category idx: %d\n", cat);
	for(std::map<std::string, int>::value_type& x : euclideanClassifier.getObjDBDict())
	{
		if (x.second == cat) {
			printf("Category : %s\n", x.first.c_str());
		}
	}


}
