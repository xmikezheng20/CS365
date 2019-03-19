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
// test KNN classifier
void test1();
// test Naive Bayes Classifier
void test2();

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
		case 1:
            test1();
            break;
		case 2:
			test2();
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
	newObj.push_back(1.71);
	newObj.push_back(0.92);
	newObj.push_back(0.99);
	newObj.push_back(0.19);
	newObj.push_back(0.01);
	newObj.push_back(0.00);

	cat = euclideanClassifier.classify(newObj);

	printf("Feature vector %.2f, %.2f, %.2f, %.2f, %.2f, %.2f\n",newObj[0],newObj[1], newObj[2], newObj[3], newObj[4], newObj[5]);
	printf("Category idx: %d\n", cat);
	for(std::map<std::string, int>::value_type& x : euclideanClassifier.getObjDBDict())
	{
		if (x.second == cat) {
			printf("Category : %s\n", x.first.c_str());
		}
	}


}

// test naive bayes classifier
void test2() {
	printf("Testing Naive Bayes classifier\n");

	// get the training set
	char filename[] = "../data/objDB.csv";
	std::vector<std::vector<double>> objDBData;
	std::vector<int> objDBCategory;
	std::map<std::string, int> objDBCategoryDict;

	readObjDB(filename, objDBData, objDBCategory, objDBCategoryDict);

	// build the scaled euclidean classifier
	NaiveBayes naiveBayesClassifier = NaiveBayes();
	naiveBayesClassifier.build(objDBData, objDBCategory, objDBCategoryDict);

	// make a test object feature vector
	std::vector<double> newObj;
	int cat;
	newObj.push_back(1.71);
	newObj.push_back(0.92);
	newObj.push_back(0.99);
	newObj.push_back(0.19);
	newObj.push_back(0.01);
	newObj.push_back(0.00);

	cat = naiveBayesClassifier.classify(newObj);

	printf("Feature vector %.2f, %.2f, %.2f, %.2f, %.2f, %.2f\n",newObj[0],newObj[1], newObj[2], newObj[3], newObj[4], newObj[5]);
	printf("Category idx: %d\n", cat);
	for(std::map<std::string, int>::value_type& x : naiveBayesClassifier.getObjDBDict())
	{
		if (x.second == cat) {
			printf("Category : %s\n", x.first.c_str());
		}
	}

}

// test KNN classifier
void test1() {
	printf("Testing KNN classifier\n");

	// get the training set
	char filename[] = "../data/objDB.csv";
	std::vector<std::vector<double>> objDBData;
	std::vector<int> objDBCategory;
	std::map<std::string, int> objDBCategoryDict;

	readObjDB(filename, objDBData, objDBCategory, objDBCategoryDict);

	// build KNN classifier
	KNN knn = KNN();
	knn.build(objDBData, objDBCategory, objDBCategoryDict, 9);

	// make a test object feature vector
	std::vector<double> newObj;
	int cat;
	newObj.push_back(1.71);
	newObj.push_back(0.92);
	newObj.push_back(0.99);
	newObj.push_back(0.19);
	newObj.push_back(0.01);
	newObj.push_back(0.00);

	cat = knn.classify(newObj);

	printf("Feature vector %.2f, %.2f, %.2f, %.2f, %.2f, %.2f\n",newObj[0],newObj[1], newObj[2], newObj[3], newObj[4], newObj[5]);
	printf("Category idx: %d\n", cat);
	for(std::map<std::string, int>::value_type& x : knn.getObjDBDict())
	{
		if (x.second == cat) {
			printf("Category : %s\n", x.first.c_str());
		}
	}

}
