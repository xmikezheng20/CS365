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
#include <cstring>

#include "classifier.hpp"

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
    ScaledEuclidean euclideanClassifier = ScaledEuclidean();

}
