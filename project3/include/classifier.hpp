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
    std::vector<std::vector<double>> objDBData;
    std::vector<int> objDBCategory;
    std::map<std::string, int> objDBDict;

public:
    // constructor
    Classifier(int type);

    // type setter and getter
    int getType();
    void setType(int newType);

    std::map<std::string, int> getObjDBDict();

    /*compute the confusion matrix*/
    /*Takes in two standard vectors of zero-index numeric categories and
        computes the confusion matrix. The rows represent true
        categories, and the columns represent the classifier output.*/
    std::vector<std::vector<int>> confusion_matrix(
        std::vector<int> truecats, std::vector<int> classcats);

    void print_confusion_matrix(std::vector<std::vector<int>> conf_mat);

    // virtual int check_unknown_label(){};

};

/* classifier using the Euclidean distance */
class ScaledEuclidean: public Classifier {
private:
    std::vector<double> stdevs;
    int size, numFeature;
    double MINDIST;

public:
    // constructor
    ScaledEuclidean();

    // build classifier
    void build(std::vector<std::vector<double>> &objDBData,
    	std::vector<int> &objDBCategory, std::map<std::string, int> &objDBDict);

    // Classify
    int classify(std::vector<double> newObj);

};

/* KNN classifier inherits Classifier */
class KNN: public Classifier{
private:
    int size, numFeature, K;
    double MINDIST;
    std::vector<double> stdevs;

public:
    // constructor
    KNN();

    /*Builds the classifier give the data points in objectDBData and the categories*/
    void build(std::vector<std::vector<double>> &objDBData,
        std::vector<int> &objDBCategory, std::map<std::string, int> &objDBDict, int K = 5);

    /*classify through KNN, return int for category*/
    int classify(std::vector<double> curObj);

    /*calculates euclidean distance between two data points*/
    double euclidean_distance(std::vector<double> dataPoint1, std::vector<double> dataPoint2 );

    /*get object dictionary*/
    std::map<std::string, int> getObjDBDict() {
        return this->objDBDict;
    };

    /* check unknown label*/
    int check_unknown_label(std::vector<std::pair<double, int>> dcPairs);


};


/* NaiveBayes classifier inherits Classifier */
class NaiveBayes: public Classifier{
private:
    cv::Ptr<cv::ml::NormalBayesClassifier> nbc;
    int size, numFeature;
    std::vector<float> mins, maxs, ranges;
public:
    // constructor
    NaiveBayes();

    // build the classifier
    void build(std::vector<std::vector<double>> &objDBData,
        std::vector<int> &objDBCategory, std::map<std::string, int> &objDBDict);

    // Classify
    int classify(std::vector<double> newObj);


};



void readObjDB(char *path, std::vector<std::vector<double>> &objDBData,
	std::vector<int> &objDBCategory, std::map<std::string, int> &objDBDict );

// helper function that calculates standard deviation of a matrix columnwise
std::vector<double> stdev(std::vector<std::vector<double>> featurels);
