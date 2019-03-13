/*
	classifier.cpp
    This is a classifier class that has several child classes. including KNN classifier
    NaiveBayes classifier

	Mike Zheng and Heidi He
	CS365 project 3
	3/11/19
*/


#include "classifier.hpp"

// classifier class

// constructor
Classifier::Classifier(int type) {
    printf("constructing\n");
    this->type = type;
}

// type setter and getter
int Classifier::getType() {
    return this->type;
}
void Classifier::setType(int newType) {
    this->type = newType;
}

/*compute the confusion matrix*/
std::vector<std::vector<int>> Classifier::confusion_matrix(
    std::vector<int> truecats, std::vector<int> classcats) {

    std::vector<std::vector<int>> conf_mat;






    return conf_mat;
}

// scaled euclidean classifier
// constructor
ScaledEuclidean::ScaledEuclidean():Classifier::Classifier(0){;}
