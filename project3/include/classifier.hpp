/*
	classifier.cpp
    This is a classifier class that has several child classes. including KNN classifier
    NaiveBayes classifier

	Mike Zheng and Heidi He
	CS365 project 3
	3/11/19
*/

#include <cstdio>
#include <cstring>
#include <dirent.h>
#include <opencv2/opencv.hpp>

#include "classifier.hpp"

/*parent class Classifier*/
class Classifier{
    protected int type; //the type of classifier

public:
    Classifier(int curType): type(curType){}
    /*getter and setter*/
    int getType();
    void setType(int newType);

    /*compute the confusion matrix*/
    std::vector<double> confusion_matrix(std::vector<double> targets,
                                        std::vector<double> outputs);

};

/* KNN classifier inherits Classifier */
class KNN: public Classifier{
public:
    KNN(int curType): Classifier(curType){}

};


/* NaiveBayes classifier inherits Classifier */
class NaiveBayes: public Classifier{
public:
    NaiveBayes(int curType): Classifier(curType) {}


};
