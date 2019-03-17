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

// build
void ScaledEuclidean::build(std::vector<std::vector<double>> &objDBData,
    std::vector<int> &objDBCategory, std::map<std::string, int> &objDBDict) {
        this->objDBData = objDBData;
        this->objDBCategory = objDBCategory;
        this->objDBDict = objDBDict;

        this->size = this->objDBData.size();
        this->numFeature = this->objDBData[0].size();

        // reshape the feature matrix
        std::vector<std::vector<double>> featurels;
        std::vector<double> tmp;

        // calculate standard deviations
        for (int i=0;i<this->numFeature;i++){
            for (int j=0;j<this->size;j++) {
                tmp.push_back(this->objDBData[j][i]);
            }
            featurels.push_back(tmp);
            tmp.clear();
        }

        this->stdevs = stdev(featurels);
        return;
    }

// helper function that calculates standard deviation of a matrix columnwise
std::vector<double> ScaledEuclidean::stdev(std::vector<std::vector<double>> featurels) {
    std::vector<double> stdevs;
    double stdev;
    std::vector<double> means;
    double sum;

    // calculate mean
    for (int i=0;i<this->numFeature;i++) {
        sum = 0;
        for (int j=0;j<this->size;j++) {
            sum += featurels[i][j];
        }
        means.push_back(sum/this->size);
    }

    // for (int i=0; i<means.size(); i++) {
    //     printf("mean of feature %d: %.2f\n",i,means[i]);
    // }

    // calculate stdev
    for (int i=0;i<this->numFeature;i++) {
        sum = 0;
        for (int j=0;j<size;j++) {
            sum += (featurels[i][j]-means[i])*(featurels[i][j]-means[i]);
        }
        stdevs.push_back(sqrt(sum/(this->size-1)));
    }
    // for (int i=0; i<stdevs.size(); i++) {
    //     printf("stdev of feature %d: %.4f\n",i,stdevs[i]);
    // }

    return stdevs;

}

// Classify a feature vector
int ScaledEuclidean::classify(std::vector<double> newObj) {
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
