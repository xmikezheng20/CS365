/*
	classifier.cpp
    This is a classifier class that has several child classes. including KNN classifier
    NaiveBayes classifier

	Mike Zheng and Heidi He
	CS365 project 3
	3/11/19
*/


#include "classifier.hpp"

/* classifier class */

/*  constructor */
Classifier::Classifier(int type) {
    printf("constructing\n");
    this->type = type;
}

/* type setter and getter */
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

/* scaled euclidean classifier */
/* constructor */
ScaledEuclidean::ScaledEuclidean():Classifier::Classifier(0){;}


/* knn */
/* constructor */
KNN::KNN():Classifier::Classifier(1){;}

/*Builds the classifier give the data points in objectDBData and the categories*/
void KNN::build(std::vector<std::vector<double>> &objDBData,
    std::vector<int> &objDBCategory, std::map<std::string, int> &objDBDict, int K){

    this->objDBData = objDBData;
    this->objDBCategory = objDBCategory;
    this->objDBDict = objDBDict;
    this->K = K;

    this->size = this->objDBData.size();
    this->numFeature = this->objDBData[0].size();

    printf("built KNN classifier\n");
}

/*calculates euclidean distance between two data points*/
double KNN::euclidean_distance(std::vector<double> dataPoint1, std::vector<double> dataPoint2 ){
    // printf("get euclidean distance\n" );
    //error check to see if features are in same length
    if( dataPoint1.size() != this->numFeature){
        printf("Error! A does not have the same number of columns as num features\n");
        return 0;
    }
    double distance = 0;
    for(int i=0; i<this->numFeature; i++){
        // printf("dataPoint1,2 %f %f\n", dataPoint1[i], dataPoint2[i]);
        distance += (dataPoint1[i]-dataPoint2[i]) * (dataPoint1[i]-dataPoint2[i]);
    }

    // printf("finished euclidean distance \n");
    return sqrt(distance);
}


/*classify through KNN, return int for category*/
int KNN::classify(std::vector<double> curObj){
    printf("classifing through KNN\n" );
    std::vector<double> distances;
    std::vector<std::pair<double, int>> distCatPairs;

    printf("this size is %d\n", this->size);
    //get euclidian distance
    for(int i=0; i< this->size; i++){
        std::vector<double> curRow = this->objDBData[i];
        double curDist = euclidean_distance(curRow, curObj);
        distances.push_back(curDist);
        std::pair<double, int> curPair(curDist, i);
        distCatPairs.push_back(curPair);
    }

    // sort  distance - cat pair by distance
    std::sort(distCatPairs.begin(), distCatPairs.end());
    //get K nearest neighbors
    std::vector<int> neighbors;
    printf("K is %d\n", this->K);
    for(int i=0; i<this->K; i++){
        printf("distance is %f\n",distCatPairs[i].first );
        printf("index is %d\n",distCatPairs[i].second );
        neighbors.push_back(distCatPairs[i].second);
    }

    int cats[this->K];
    for(int i=0; i<this->K; i++){
        int curIndex = neighbors[i];
        // printf("neighbors indexs are %d\n", neighbors[i]);
        int curCat = this->objDBCategory[curIndex];
        cats[i] = curCat;
        // printf("cat is %d\n", cats[i]);

    }

    //calculate the most frequent class in the neighbors
    int previous = cats[0];
    int majority = cats[0];
    int count = 1;
    int maxCount = 1;
    for (int i = 1; i < this->K; i++) {
        // printf("cat is %d\n", cats[i]);
        if (cats[i] == previous)
            count++;
        else {
            if (count > maxCount) {
                majority = cats[i-1];
                maxCount = count;
            }
            previous = cats[i];
            count = 1;
        }
    }

    return majority;
}

int doubleComparator(double d1, double d2){
    if(d1>d2) return 1;
    else if(d1<d2) return -1;
    else return 0;

}
