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

/*parent class Classifier*/
class Classifier{
protected:
    int type; //the type of classifier

public:
    // constructor
    Classifier(int type);

    // type setter and getter
    int getType();
    void setType(int newType);

    /*compute the confusion matrix*/
    /*Takes in two standard vectors of zero-index numeric categories and
        computes the confusion matrix. The rows represent true
        categories, and the columns represent the classifier output.*/
    std::vector<std::vector<int>> confusion_matrix(
        std::vector<int> truecats, std::vector<int> classcats);

};

/* classifier using the Euclidean distance */
class ScaledEuclidean: public Classifier {
public:
    // constructor
    ScaledEuclidean();

};

/* KNN classifier inherits Classifier */
class KNN: public Classifier{
public:
    // constructor
    KNN();

};


/* NaiveBayes classifier inherits Classifier */
class NaiveBayes: public Classifier{
public:
    // constructor
    NaiveBayes();

};
