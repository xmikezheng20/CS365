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

std::map<std::string, int> Classifier::getObjDBDict() {
    return this->objDBDict;
}

/*compute the confusion matrix*/
std::vector<std::vector<int>> Classifier::confusion_matrix(
    std::vector<int> truecats, std::vector<int> classcats) {
    // for (int i = 0;i<truecats.size();i++) {
    //     printf("%d %d\n", truecats[i], classcats[i]);
    // }

    //true cat & class cat array append -1
    // truecats.insert(truecats.end(), -1);
    // classcats.insert(classcats.end(), -1);

    //clear dictionary
    // this->objDBDict.erase("unknown");
    // printf("successfully erased unknown label\n");
    // printf("truecats size: %lu; classcat size: %lu\n", truecats.size(), classcats.size() );
    // initialize matrix
    std::vector<std::vector<int>> conf_mat;
    for (int i=0; i<this->objDBDict.size();i++){
        std::vector<int> tmp;
        for (int j=0; j<this->objDBDict.size();j++) {
            tmp.push_back(0);
        }
        conf_mat.push_back(tmp);
    }
    // printf("DEBUG 2\n");
    // fill matrix
    for (int i=0;i<truecats.size();i++) {
        conf_mat[truecats[i]][classcats[i]]++;
    }
    // printf("DEBUG 3\n");

    // for (int i=0;i<truecats.size();i++) {
    //     printf("%d %d\n", truecats[i], classcats[i]);
    // }

    return conf_mat;
}


// print out the confusion matrix
void Classifier::print_confusion_matrix(std::vector<std::vector<int>> conf_mat) {
    printf("Confusion matrix for classifier type %d: column-predict, row-true\n", this->type);
    // get the keys
    std::vector<std::string> keys;

    std::map<int, std::string> tmpmap;

    for(std::map<std::string, int>::value_type& x : this->objDBDict)
    {
        tmpmap[x.second] = x.first;
    }

    for(std::map<int, std::string>::value_type& x : tmpmap)
    {
        keys.push_back(x.second);
    }

    // first line
    printf("        ");
    for (int i=0; i<keys.size();i++) {
        printf("|%8.8s",keys[i].c_str());
    }
    printf("\n");
    for (int i=0; i<keys.size();i++) {
        printf("----------");
    }
    printf("\n");
    // other lines
    for (int i=0; i<keys.size()-1;i++) {
        printf("%8.8s", keys[i].c_str());
        for (int j=0;j<keys.size();j++) {
            printf("|%8d",conf_mat[i][j]);
        }
        printf("\n");
    }
}

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
    this->MINDIST = 0.7;

    // reshape the feature matrix
    std::vector<std::vector<double>> featurels;
    std::vector<double> tmp;

    // printf("Reshaping feature matrix\n");
    for (int i=0;i<this->numFeature;i++){
        for (int j=0;j<this->size;j++) {
            tmp.push_back(this->objDBData[j][i]);
        }
        featurels.push_back(tmp);
        tmp.clear();
    }

    // calculate standard deviations
    // printf("calculating stdev\n");
    this->stdevs = stdev(featurels);


    // printf("built KNN classifier\n");
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
        distance += (dataPoint1[i]-dataPoint2[i]) * (dataPoint1[i]-dataPoint2[i]) / this->stdevs[i]/ this->stdevs[i];
    }
    // printf("distance is %f\n", sqrt(distance));
    // printf("finished euclidean distance \n");
    return sqrt(distance);
}

int KNN::check_unknown_label(std::vector<std::pair<double, int>> dcPairs){
    // for (int i=0; i<K; i++) {
    //     printf("%.4f %d\n", dcPairs[i].first, dcPairs[i].second);
    // }
    // printf("%.4f\n", dcPairs[0].first);
    if(dcPairs[0].first < this->MINDIST){
        return 1;
    }
    //if unknown_label, return 0
    return 0;
}

/*classify through KNN, return int for category*/
int KNN::classify(std::vector<double> curObj){
    printf("Classify using KNN\n" );
    std::vector<double> distances;
    std::vector<std::pair<double, int>> distCatPairs;

    // printf("this size is %d\n", this->size);
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

    //check unknown label
    if (!check_unknown_label(distCatPairs)){
        printf("discovered unknown label\n");
        return this->objDBDict.size()-1;// category is max for unknown label
    }
    //get K nearest neighbors
    std::vector<int> neighbors;
    // printf("K is %d\n", this->K);
    for(int i=0; i<this->K; i++){
        // printf("distance is %f\n",distCatPairs[i].first );
        // printf("index is %d\n",distCatPairs[i].second );
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

    printf("KNN result %d/%d with min dist %.2f\n", count, K, distCatPairs[0].first);
    // another way to classify unknown
    if (count < int(K/2)+1) {
        return this->objDBDict.size()-1;
    }
    return majority;
}

int doubleComparator(double d1, double d2){
    if(d1>d2) return 1;
    else if(d1<d2) return -1;
    else return 0;
}



// scaled euclidean classifier
// constructor
ScaledEuclidean::ScaledEuclidean():Classifier::Classifier(0){;}

// build
void ScaledEuclidean::build(std::vector<std::vector<double>> &objDBData,
    std::vector<int> &objDBCategory, std::map<std::string, int> &objDBDict) {
        this->objDBData = objDBData;
        this->objDBCategory = objDBCategory;
        this->objDBDict = objDBDict;
        this->size = this->objDBData.size();
        this->numFeature = this->objDBData[0].size();

        this->MINDIST = 0.7;

        // reshape the feature matrix
        std::vector<std::vector<double>> featurels;
        std::vector<double> tmp;

        // printf("Reshaping feature matrix\n");
        for (int i=0;i<this->numFeature;i++){
            for (int j=0;j<this->size;j++) {
                tmp.push_back(this->objDBData[j][i]);
            }
            featurels.push_back(tmp);
            tmp.clear();
        }

        // calculate standard deviations
        // printf("calculating stdev\n");
        this->stdevs = stdev(featurels);
        // for (int i=0; i<this->numFeature; i++){
        //     printf("STDEV %d: %.2f\n", i, this->stdevs[i]);
        // }
        return;
    }

// helper function that calculates standard deviation of a matrix columnwise
std::vector<double> stdev(std::vector<std::vector<double>> featurels) {
    std::vector<double> stdevs;
    double stdev;
    std::vector<double> means;
    double sum;
    int numFeature = featurels.size();
    int size = featurels[0].size();

    // calculate mean
    for (int i=0;i<numFeature;i++) {
        sum = 0;
        for (int j=0;j<size;j++) {
            sum += featurels[i][j];
        }
        means.push_back(sum/size);
    }

    // for (int i=0; i<means.size(); i++) {
    //     printf("mean of feature %d: %.2f\n",i,means[i]);
    // }

    // calculate stdev
    for (int i=0;i<numFeature;i++) {
        sum = 0;
        for (int j=0;j<size;j++) {
            sum += (featurels[i][j]-means[i])*(featurels[i][j]-means[i]);
        }
        stdevs.push_back(sqrt(sum/(size-1)));
    }
    // for (int i=0; i<stdevs.size(); i++) {
    //     printf("stdev of feature %d: %.4f\n",i,stdevs[i]);
    // }

    return stdevs;

}

// Classify a feature vector
int ScaledEuclidean::classify(std::vector<double> newObj) {
    printf("Classify using Scaled Euclidean Classifier\n");
    double dist, scaledDiff;
    double min = 100000;
    int cat = -1;
    for (int i=0; i<this->size; i++) {
        dist = 0;
        for (int j=0;j<this->numFeature; j++) {
            scaledDiff = (newObj[j]-this->objDBData[i][j])/this->stdevs[j];
            dist += scaledDiff*scaledDiff;
        }
        if (dist < min) {
            min = dist;
            cat = this->objDBCategory[i];
        }
    }
    min = sqrt(min);
    //for unknown label check
    // printf("!!!min dist is %f\n", min);
    if(min>this->MINDIST){
        printf("Unknown obj\n");
        return this->objDBDict.size()-1;
    }

    printf("Min dist: %.2f\n", min);
    return cat;
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
			if (idx<6) {
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
        // printf("Feature length %lu\n", feature.size());


		objDBData.push_back(feature);
	}
    //insert the unknown label to dictionary
    objDBDict["unknown"] = objDBDict.size();



    // // objDBDict.insert( {'unknown',-1});
    // //print dictionary for debug
    // for(auto it = objDBDict.cbegin(); it != objDBDict.cend(); ++it){
    //     std::cout << it->first << " " << it->second << " " << "\n";
    // }
    // printf("DEBUG 5 - finished readObjDB\n");
}


// naive bayes classifier from OpenCV (normal bayes)
// constructor
NaiveBayes::NaiveBayes():Classifier::Classifier(2){;}

// build the classifier
void NaiveBayes::build(std::vector<std::vector<double>> &objDBData,
    std::vector<int> &objDBCategory, std::map<std::string, int> &objDBDict) {
    this->objDBData = objDBData;
    this->objDBCategory = objDBCategory;
    this->objDBDict = objDBDict;

    this->size = this->objDBData.size();
    this->numFeature = this->objDBData[0].size();

    for (int i=0; i<this->numFeature; i++) {
        this->mins.push_back(10000);
        this->maxs.push_back(0);
    }

    this->nbc = cv::ml::NormalBayesClassifier::create();

    // copy vector of vector to mat
    // min/max normalize
    cv::Mat trainingData = cv::Mat(this->size,this->numFeature,CV_32F);
    for (int i=0; i<this->size; i++) {
        for (int j=0; j<this->numFeature; j++) {
            float val = float(this->objDBData[i][j]);
            if (val<this->mins[j]) {
                this->mins[j] = val;
            } else if (val>this->maxs[j]) {
                this->maxs[j] = val;
            }
        }
    }

    // check min max
    for (int i=0; i<this->numFeature; i++) {
        // printf("Feature %d has min %.4f, max %.4f\n", i, this->mins[i], this->maxs[i]);
        this->ranges.push_back(this->maxs[i]-this->mins[i]);
    }

    // update mat
    for (int i=0; i<this->size; i++) {
        for (int j=0; j<this->numFeature; j++) {
            float val = float(this->objDBData[i][j]);
            trainingData.at<float>(i,j) = (val-this->mins[j])/this->ranges[j];
        }
    }

    // std::cout<<trainingData<<std::endl;

    cv::Mat trainingCats = cv::Mat(this->size, 1, CV_32SC1);
    for (int i=0; i<this->size; i++) {
        trainingCats.at<int>(i,0) = float(this->objDBCategory[i]);
    }

    this->nbc->train(trainingData,cv::ml::ROW_SAMPLE, trainingCats);

}

// Classify
int NaiveBayes::classify(std::vector<double> newObj) {
    printf("Classify using NBC\n");
    cv::Mat newObjMat = cv::Mat(1,this->numFeature,CV_32F);
    for (int i=0;i<this->numFeature;i++) {
        // min/max normalize
        newObjMat.at<float>(0,i) = (float(newObj[i])-this->mins[i])/this->ranges[i];
    }

    // std::cout<<newObjMat<<std::endl;

    cv::Mat outputs, outputProbs;

    this->nbc->predictProb(newObjMat, outputs, outputProbs);
    int cat = int(outputs.at<int>(0,0));
    // std::cout<<outputs<<std::endl;
    // std::cout<<outputProbs<<std::endl;

    return cat;
}
