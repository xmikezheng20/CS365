/*
  img.cpp
  Mike Zheng and Heidi He
  CS365 project 2
  2/16/19

  img class holds information about an image in the image database
  may compare with query image and store the similarity score

*/

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "opencv2/opencv.hpp"

#include "img.hpp"
#include "hist.hpp"

// constructor
Img::Img(char *newPath) {
    this->path = newPath;
    this->status = 0; // 0 -> image unchecked; 1 -> image checked
    this->similarity = 0; //the larger similarity is, the closer the images are
    // printf("The path is %s\n",this->path);
}

/* getters and setters*/
/* get image path (address) */
char *Img::getPath() {
    return this->path;
}

/* set image path */
void Img::setPath(char *newPath) {
    this->path = newPath;
}

/* get status: 0 -> image unchecked; 1 -> image checked*/
int Img::getStatus() {
    return this->status;
}

/*get image status, to see if it is checked*/
void Img::setStatus(int newStatus) {
    this->status = newStatus;
}


/* return similarity, the larger similarity is, the closer the images are*/

double Img::getSimilarity() {
    return this->similarity;
}
/* set similarity*/
void Img::setSimilarity(double newSimilarity) {
    this->similarity = newSimilarity;
}

// print image info
void Img::printImgInfo() {
    printf("Image: %s\nStatus: %d\nSimilarity: %.4f\n\n", this->path, this->status, this->similarity);
}


/*baseline histrogram that takes in a query image and an image database,
                matches the query image to each database image using a distance metric,
                then sorts the database images according to their similarity to the query images. */
void Img::baselineMatching(cv::Mat queryBlock, int halfBlockSize) {
    printf("Baseline Matching with %s\n", this->path);

    cv::Mat dataImg = cv::imread(this->path);
    if(dataImg.data == NULL) {
    printf("Unable to read data image %s\n", this->path);
    exit(-1);
    }
    // printf("data image size: %d rows x %d columns\n", (int)dataImg.size().height, (int)dataImg.size().width);
    int dataMidLeft = ((int)dataImg.size().width)/2-halfBlockSize;
    int dataMidUp = ((int)dataImg.size().height)/2-halfBlockSize;
    // printf("data image mid point %d,%d\n", dataMidLeft+2, dataMidUp+2);

    // calculate sum of square
    int ssd = 0;
    cv::Vec3b *queryPixel;
    cv::Vec3b *dataPixel;
    for (int i=0; i<halfBlockSize*2+1; i++) {
        for (int j=0; j<halfBlockSize*2+1; j++) {
          // printf("comparing query (%d,%d) vs. data (%d,%d)\n", queryMidLeft+i,queryMidUp+j,dataMidLeft+i,dataMidUp+j);
          queryPixel = &(queryBlock.at<cv::Vec3b>(j, i));
          dataPixel = &(dataImg.at<cv::Vec3b>(dataMidUp+j, dataMidLeft+i));
          ssd += (queryPixel->val[0]-dataPixel->val[0])*(queryPixel->val[0]-dataPixel->val[0])
                +(queryPixel->val[1]-dataPixel->val[1])*(queryPixel->val[1]-dataPixel->val[1])
                +(queryPixel->val[2]-dataPixel->val[2])*(queryPixel->val[2]-dataPixel->val[2]);
        }
    }
    // printf("ssd is %d\n",ssd);
    this->similarity = (double)(-ssd);
}

/* uses a whole-image histogram to determine the similarity
between the query image and the DB images. */
void Img::baselineHistogram(cv::Mat queryHist) {
    printf("Baseline Histogram Matching with %s\n", this->path);
    cv::Mat targetHist;
    targetHist = hist_whole_hs(this->path);
    // use intersection distance

    this->similarity = cv::compareHist(queryHist, targetHist, cv::HISTCMP_INTERSECT);
    // this->similarity = cv::compareHist(queryHist, targetHist,  cv::HISTCMP_CORREL);
}

/*multi histogram matching*/
void Img::multiHistogram(cv::Mat queryHist1, cv::Mat queryHist2){
    printf("Multiple Histogram Matching with %s\n", this->path);
    cv::Mat targetHist1 = multi_hist_whole_hs(this->path).first;
    cv::Mat targetHist2 = multi_hist_whole_hs(this->path).second;
    //use chi-square comparison
    double similarity1 = cv::compareHist(queryHist1, targetHist1, cv::HISTCMP_INTERSECT);
    double similarity2 = cv::compareHist(queryHist2, targetHist2, cv::HISTCMP_INTERSECT); //CV_COMP_CHISQR
    // double similarity1 = cv::compareHist(queryHist1, targetHist1, cv::HISTCMP_CORREL);
    // double similarity2 = cv::compareHist(queryHist2, targetHist2, cv::HISTCMP_CORREL);
    this->similarity = similarity1*5 + similarity2; //so far, arbituary weight

}


void Img::colorTextureHistogram(cv::Mat queryColorHist, std::vector<cv::Mat> queryHists) {
    printf("Color Texture Histogram Matching with %s\n", this->path);
    std::vector<cv::Mat> targetHists;
    cv::Mat targetColorHist;
    targetColorHist = hist_whole_hs(this->path);

    targetHists = hist_whole_texture_laws_subset(this->path);
    // std::cout << "hist0 is " << queryHists[0] <<std::endl;
    // std::cout << "hist1 is " << queryHists[1] <<std::endl;
    this->similarity = (cv::compareHist(queryHists[0], targetHists[0], cv::HISTCMP_INTERSECT)
                    + cv::compareHist(queryHists[1], targetHists[1], cv::HISTCMP_INTERSECT))*10
                    + cv::compareHist(queryColorHist, targetColorHist, cv::HISTCMP_INTERSECT);
}


/*using earth mover's distance, return the value of similarity*/
void Img::earthMoverDistance(cv::Mat queryHist) {
    printf("Earth Mover's distance Matching with %s\n", this->path);
    cv::Mat targetHist;
    targetHist = hist_whole_hs(this->path);

     //compare histogram
     int hbins = 30, sbins = 32; //hard coding for now. from hist_whole_hs function
     int numrows = hbins * sbins;

     //make signature
     cv::Mat sig1(numrows, 3, CV_32FC1);
     cv::Mat sig2(numrows, 3, CV_32FC1);

     //fill value into signature
     for(int h=0; h< hbins; h++){
        for(int s=0; s< sbins; ++s){
        float binval = queryHist.at< float>(h,s);
        sig1.at< float>( h*sbins + s, 0) = binval;
        sig1.at< float>( h*sbins + s, 1) = h;
        sig1.at< float>( h*sbins + s, 2) = s;

        binval = targetHist.at< float>(h,s);
        sig2.at< float>( h*sbins + s, 0) = binval;
        sig2.at< float>( h*sbins + s, 1) = h;
        sig2.at< float>( h*sbins + s, 2) = s;
        }
     }

     //compare similarity of 2images using emd.
     double emd = cv::EMD(sig1, sig2, cv::DIST_L2); //emd 0 is best matching.
     // printf("similarity %5.5f %%\n", (1-emd)*100 );

     this->similarity = -emd;

}

/* comparing red & green & blue & saturation histrograms, giving green more visual weight */
void Img::rgbsHistogram(std::vector<cv::Mat> queryHists){
    printf("rgbs Histogram Matching with %s\n", this->path);
    std::vector<cv::Mat> targetHists = hist_whole_rgbs(this->path);

    // printf("finished whole rgbs function\n" );
    //more visual weight for green and add some for saturation
    // this->similarity = cv::compareHist(queryHists[0], targetHists[0], cv::HISTCMP_INTERSECT);
    this->similarity = cv::compareHist(queryHists[0], targetHists[0], cv::HISTCMP_INTERSECT)
                    + cv::compareHist(queryHists[1], targetHists[1], cv::HISTCMP_INTERSECT)*2
                    + cv::compareHist(queryHists[2], targetHists[2], cv::HISTCMP_INTERSECT)
                    + cv::compareHist(queryHists[3], targetHists[3], cv::HISTCMP_INTERSECT)*0;

}

// destructor
Img::~Img(){
    free(this->path);
}
