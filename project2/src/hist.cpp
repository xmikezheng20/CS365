/*
  hist.cpp
  Mike Zheng and Heidi He
  CS365 project 2
  2/26/19

  a library of histogram functions

*/

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "opencv2/opencv.hpp"

#include "hist.hpp"

/*create whole hue-saturation histogram for a given path*/
cv::Mat hist_whole_hs(char *path) {
    // printf("Calculating hs histogram of %s\n", path);
    cv::Mat src, hsv;

    // read the image
    src = cv::imread(path);
    if(src.data == NULL) {
    printf("Unable to read query image %s\n", path);
    exit(-1);
    }

    // cv::imshow(path, src);
    // cv::waitKey(0);

    // convert to hsv
    cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);

    // quantize the hue to 30 levels
    // saturation to 32 levels
    int hbins = 30, sbins = 32;
    int histSize[] = {hbins, sbins};
    // hue varies from 0 to 179
    float hranges[] = {0, 180};
    // saturation ranges from 0 to 255
    float sranges[] = {0, 256};
    const float* ranges[] = {hranges, sranges};

    cv::Mat hist;
    // channels 0 and 1
    int channels[] = {0,1};
    // cv::calcHist( &hsv, 0, channels, cv::Mat(), hist, 1, histSize, ranges, true, false);
    cv::calcHist( &hsv, 1, channels, cv::Mat(), hist, 2, histSize, ranges, true, false);
    // printf("width*height = %d\n", (int)(src.size().width)*(int)(src.size().height));
    // printf("sum of hist = %d\n", (int)(cv::sum(hist)[0]));

    // normalize the histogram
    // cv::normalize( hist, hist, 0, 1, cv::NORM_MINMAX, -1, cv::Mat() );
    hist /= (int)(src.size().width)*(int)(src.size().height);

    //draw histogram, could be commented out:
    //draw_hist(src, hist, hbins, sbins);

    return hist;
}

/*create whole hue-saturation histogram for an img*/
cv::Mat hist_whole_hs_img(cv::Mat src) {
    cv::Mat hsv;
    // convert to hsv
    cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);

    // quantize the hue to 30 levels
    // saturation to 32 levels
    int hbins = 30, sbins = 32;
    int histSize[] = {hbins, sbins};
    // hue varies from 0 to 179
    float hranges[] = {0, 180};
    // saturation ranges from 0 to 255
    float sranges[] = {0, 256};
    const float* ranges[] = {hranges, sranges};

    cv::Mat hist;
    // channels 0 and 1
    int channels[] = {0,1};
    cv::calcHist( &hsv, 1, channels, cv::Mat(), hist, 2, histSize, ranges, true, false);
    // normalize the histogram
    // cv::normalize( hist, hist, 0, 1, cv::NORM_MINMAX, -1, cv::Mat() );
    hist /= (int)(src.size().width)*(int)(src.size().height);

    //draw histogram, could be commented out
    //draw_hist(src, hist, hbins, sbins);

    return hist;
}


/*create multi whole hue-saturation histogram and return two*/
std::pair<cv::Mat,cv::Mat> multi_hist_whole_hs(char *path){
    cv::Mat src, hsv;
    //divide the query image into blocks,
    //compute histogram in each block and sum together as a whole histogram

    // printf("query image size: %d rows x %d columns\n", (int)src.size().height, (int)src.size().width);
    src = cv::imread(path);
    if(src.data == NULL) {
    printf("Unable to read query image %s\n", path);
    exit(-1);
    }

    //first take the vertical center 1/2 from the query image
    cv::Mat queryBlockCenter;
    src(cv::Rect(((int)src.size().width)/3,
              ((int)src.size().height)/3,
              ((int)src.size().width)/3,
              ((int)src.size().height)/3)).copyTo(queryBlockCenter);
    cv::Mat queryHistCenter = hist_whole_hs_img(queryBlockCenter); // first histogram for multi histogram input

    //then take the edges and corner of the image: 1/4 of the image in all directions
    //top
    cv::Mat queryBlockEdge1;
    src(cv::Rect(0, 0, (int)src.size().width, ((int)src.size().height)/4)).copyTo(queryBlockEdge1);
    cv::Mat queryHistEdge1 = hist_whole_hs_img(queryBlockEdge1);
    //left
    cv::Mat queryBlockEdge2;
    src(cv::Rect(0, 0, ((int)src.size().width)/4, ((int)src.size().height))).copyTo(queryBlockEdge2);
    cv::Mat queryHistEdge2 = hist_whole_hs_img(queryBlockEdge2);
    //bottom
    cv::Mat queryBlockEdge3;
    src(cv::Rect(0, ((int)src.size().height)*3/4, (int)src.size().width, ((int)src.size().height)/4)).copyTo(queryBlockEdge3);
    cv::Mat queryHistEdge3 = hist_whole_hs_img(queryBlockEdge3);
    //right
    cv::Mat queryBlockEdge4;
    src(cv::Rect((int)src.size().width*3/4, 0, (int)src.size().width/4, ((int)src.size().height))).copyTo(queryBlockEdge4);
    cv::Mat queryHistEdge4 = hist_whole_hs_img(queryBlockEdge4);

    cv::Mat queryHistEdge = queryHistEdge1 + queryHistEdge2 + queryHistEdge3 + queryHistEdge4;

    return std::make_pair(queryHistCenter, queryHistEdge);
}


/*draw histogram given src, histogram, hue bins, saturation bins*/
void draw_hist_whole_hs(cv::Mat src, cv::Mat hist, int hbins, int sbins){
    double maxVal = 0;
    cv::minMaxLoc(hist, 0, &maxVal, 0, 0);
    // draw histogram
    int scale = 10;
    cv::Mat histImg = cv::Mat::zeros(sbins*scale, hbins*10, CV_8UC3);
    for( int h = 0; h < hbins; h++ )
      for( int s = 0; s < sbins; s++ )
      {
          float binVal = hist.at<float>(h, s);
          int intensity = cvRound(binVal*255/maxVal);
          cv::rectangle( histImg, cv::Point(h*scale, s*scale),
                      cv::Point( (h+1)*scale - 1, (s+1)*scale - 1),
                      cv::Scalar::all(intensity),
                      -1 );
      }
    cv::namedWindow( "Source", 1 );
    cv::imshow( "Source", src );
    cv::namedWindow( "H-S Histogram", 1 );
    cv::imshow( "H-S Histogram", histImg );
    cv::waitKey(0);
}

/* color and texture histogram of the whole image
* color: HS-histogram
* texture: apply multiple texture filters, aggregate 7*7 box to get energy,
* calculates energy histograms */
std::vector<cv::Mat> hist_whole_texture_laws_subset(char *path) {
    // printf("Calculating color texture histogram of %s\n", path);
    std::vector<cv::Mat> hists;

    cv::Mat src, src_gray, filtered, l5l5Response, filtered_abs, energy, hist;

    int histSize = 50;
    float range[] = { 0, 256 }; //the upper boundary is exclusive
    const float* histRange = { range };

    // read the image
    src = cv::imread(path);
    if(src.data == NULL) {
        printf("Unable to read query image %s\n", path);
        exit(-1);
    }

    // convert to grayscale
    cv::cvtColor( src, src_gray, cv::COLOR_BGR2GRAY );
    // src_gray.convertTo(src_gray, CV_32F);

    // laws filters
    // // l5l5 to reduce illumination
    // float l5l5_data[25] = {1, 4, 6, 4, 1,
    //                        4, 16, 24, 16, 4,
    //                        6, 24, 36, 24, 6,
    //                        4, 16, 24, 16, 4,
    //                        1, 4, 6, 4, 1};
    // cv::Mat l5l5 = cv::Mat(5, 5, CV_32F, l5l5_data);
    // // l5l5 /= 256;
    //
    // cv::filter2D(src_gray, l5l5Response, -1, l5l5, cv::Point(-1, -1), 0,
    //              cv::BORDER_DEFAULT);


    // e5l5
    float e5l5_data[25] = {-1, -4, -6, -4, -1,
                         -2, -8, -12, -8, -2,
                         0, 0, 0, 0, 0,
                         2, 8, 12, 8, 2,
                         1, 4, 6, 4, 1};
    cv::Mat e5l5 = cv::Mat(5, 5, CV_32F, e5l5_data);
    cv::filter2D(src_gray, filtered, -1, e5l5, cv::Point(-1, -1), 0,
               cv::BORDER_DEFAULT);

    // // normalize by l5l5 response
    // cv::divide(filtered, l5l5Response, filtered);
    //
    // average absolute values in 7*7 block to get energy
    filtered_abs = cv::abs(filtered);
    cv::blur(filtered_abs, energy, cv::Size(7, 7));
    // energy = filtered;
    //
    // calculate histogram
    cv::calcHist( &energy, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, true, false);
    // cv::normalize( hist, hist, 0, 1, cv::NORM_MINMAX, -1, cv::Mat() );
    hist /= (int)(src.size().width)*(int)(src.size().height);

    hists.push_back(hist.clone());


    // // l5e5
    float l5e5_data[25] = {-1, -2, 0, 2, 1,
                        -4, -8, 0, 8, 4,
                        -6, -12, 0, 12, 6,
                        -4, -8, 0, 8, 4,
                        -1, -2, 0, 2, 1};
    cv::Mat l5e5 = cv::Mat(5, 5, CV_32F, l5e5_data);

    cv::filter2D(src_gray, filtered, -1, l5e5, cv::Point(-1, -1), 0,
              cv::BORDER_DEFAULT);

    // normalize by l5l5 response
    // cv::divide(filtered, l5l5Response, filtered);

    // average absolute values in 7*7 block to get energy
    filtered_abs = cv::abs(filtered);
    cv::blur(filtered_abs, energy, cv::Size(7, 7));

    // calculate histogram
    cv::calcHist( &energy, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, true, false);
    // cv::normalize( hist, hist, 0, 1, cv::NORM_MINMAX, -1, cv::Mat() );
    hist /= (int)(src.size().width)*(int)(src.size().height);

    hists.push_back(hist.clone());

    // l5r5
    float l5r5_data[25] = {1, -4, 6, -4, 1,
                        4, -16, 24, -16, 4,
                        6, -24, 36, -24, 6,
                        4, -16, 24, -16, 4,
                        1, -4, 6, -4, 1};
    cv::Mat l5r5 = cv::Mat(5, 5, CV_32F, l5r5_data);

    cv::filter2D(src_gray, filtered, -1, l5r5, cv::Point(-1, -1), 0,
              cv::BORDER_DEFAULT);


    // normalize by l5l5 response
    // cv::divide(filtered, l5l5Response, filtered);

    // average absolute values in 7*7 block to get energy
    filtered_abs = cv::abs(filtered);
    cv::blur(filtered_abs, energy, cv::Size(7, 7));
    // energy = filtered;

    // calculate histogram
    cv::calcHist( &energy, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, true, false);
    // cv::normalize( hist, hist, 0, 1, cv::NORM_MINMAX, -1, cv::Mat() );
    hist /= (int)(src.size().width)*(int)(src.size().height);

    hists.push_back(hist.clone());

    // r5l5
    float r5l5_data[25] = {1, 4, 6, 4, 1,
                        -4, -16, -24, -16, -4,
                        6, 24, 36, 24, 6,
                        -4, -16, -24, -16, -4,
                        1, 4, 6, 4, 1};
    cv::Mat r5l5 = cv::Mat(5, 5, CV_32F, r5l5_data);

    cv::filter2D(src_gray, filtered, -1, r5l5, cv::Point(-1, -1), 0,
              cv::BORDER_DEFAULT);

    // normalize by l5l5 response
    // cv::divide(filtered, l5l5Response, filtered);

    // average absolute values in 7*7 block to get energy
    filtered_abs = cv::abs(filtered);
    cv::blur(filtered_abs, energy, cv::Size(7, 7));
    // energy = filtered;

    // calculate histogram
    cv::calcHist( &energy, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, true, false);
    // cv::normalize( hist, hist, 0, 1, cv::NORM_MINMAX, -1, cv::Mat() );
    hist /= (int)(src.size().width)*(int)(src.size().height);

    hists.push_back(hist.clone());


    // s5s5
    float s5s5_data[25] = {1, 0, -2, 0, 1,
                        0, 0, 0, 0, 0,
                        -2, 0, 4, 0, -2,
                        0, 0, 0, 0, 0,
                        1, 0, -2, 0, 1};
    cv::Mat s5s5 = cv::Mat(5, 5, CV_32F, s5s5_data);

    cv::filter2D(src_gray, filtered, -1, s5s5, cv::Point(-1, -1), 0,
              cv::BORDER_DEFAULT);

    // normalize by l5l5 response
    // cv::divide(filtered, l5l5Response, filtered);

    // average absolute values in 7*7 block to get energy
    filtered_abs = cv::abs(filtered);
    cv::blur(filtered_abs, energy, cv::Size(7, 7));
    // energy = filtered;

    // calculate histogram
    cv::calcHist( &energy, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, true, false);
    // cv::normalize( hist, hist, 0, 1, cv::NORM_MINMAX, -1, cv::Mat() );
    hist /= (int)(src.size().width)*(int)(src.size().height);

    hists.push_back(hist.clone());

    //
    // std::cout << "hist0 is " << hists[0] <<std::endl;
    // std::cout << "hist1 is " << hists[1] <<std::endl;
    //
    // draw filterd image
    // cv::namedWindow( "Source_gray", 1 );
    // cv::imshow( "Source_gray", src_gray );
    // cv::namedWindow( "Filtered", 1 );
    // cv::imshow( "Filtered", energy );
    // cv::waitKey(0);


    return hists;
}

// texture histogram using sobelx and sobel y
cv::Mat hist_whole_texture_sobel(char *path) {
    cv::Mat src, src_gray;
    cv::Mat grad;
    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;
    int ksize = 3;

    // read the image
    src = cv::imread(path);
    if(src.data == NULL) {
        printf("Unable to read query image %s\n", path);
        exit(-1);
    }

    // Remove noise by blurring with a Gaussian filter ( kernel size = 3 )
    cv::GaussianBlur(src, src, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);

    // Convert the image to grayscale
    cv::cvtColor(src, src_gray, cv::COLOR_BGR2GRAY);

    cv::Mat grad_x, grad_y;
    cv::Mat abs_grad_x, abs_grad_y;
    cv::Sobel(src_gray, grad_x, ddepth, 1, 0, ksize, scale, delta, cv::BORDER_DEFAULT);
    cv::Sobel(src_gray, grad_y, ddepth, 0, 1, ksize, scale, delta, cv::BORDER_DEFAULT);

    // converting back to CV_8U
    cv::convertScaleAbs(grad_x, abs_grad_x);
    cv::convertScaleAbs(grad_y, abs_grad_y);

    addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

    // calculate histogram
    int histSize = 50;
    float range[] = { 0, 256 }; //the upper boundary is exclusive
    const float* histRange = { range };
    cv::Mat hist;

    cv::calcHist( &grad, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, true, false);
    hist /= (int)(src.size().width)*(int)(src.size().height);


    // cv::namedWindow( "window", 1 );
    // cv::imshow( "window", grad );
    // cv::waitKey(0);
    // cv::destroyWindow("window");

    return hist;
}



/*create whole rgb + saturation histogram for a given path
* returns a list of 1-d histogram*/
std::vector<cv::Mat> hist_whole_rgbs(char *path) {
    std::vector<cv::Mat> histList;
    cv::Mat src, rgb[3], hsv, newHSV[3];


    // printf("splited rgb channels \n");
    // read the image
    src = cv::imread(path);
    if(src.data == NULL) {
    // printf("Unable to read query image %s\n", path);
    exit(-1);
    }

    //seperate rgb channals
    split(src, rgb); //blue rgb[0]
                    //green rgb[1]
                     //red rgb[2]

    const int sizes = 30;
    float rgbsRange[] = {0,256};    // saturation ranges from 0 to 255
    const float* histRange = { rgbsRange };

    cv::Mat histR, histG, histB, histS; // histrogram for red, green, blue and saturation
    // printf("calculate hist for each channel\n");
    // std::cout << "Mat rgb[0] = "<< std::endl << " "  << rgb[0] << std::endl << std::endl;

    cv::calcHist( &rgb[0], 1, 0, cv::Mat(), histB, 1, &sizes, &histRange, true, false); //blue
    cv::calcHist( &rgb[1], 1, 0, cv::Mat(), histG, 1, &sizes, &histRange, true, false); //green
    cv::calcHist( &rgb[2], 1, 0, cv::Mat(), histR, 1, &sizes, &histRange, true, false); //red

    // printf("normalize hist for each channel\n");
    histB /= (int)(src.size().width)*(int)(src.size().height);
    histG /= (int)(src.size().width)*(int)(src.size().height);
    histR /= (int)(src.size().width)*(int)(src.size().height);

    histList.push_back(histB.clone());
    histList.push_back(histG.clone());
    histList.push_back(histR.clone());

    // calculate saturation histogram
    // convert to hsv
    // printf("convert to hsv\n");
    cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);
    split(hsv, newHSV); //blue rgb[0]

    // quantize the hue to 30 levels
    // saturation to 32 levels
    int sbins = 30;
    // saturation ranges from 0 to 255
    float sRange[] = {0,256};    // saturation ranges from 0 to 255
    const float* ranges = {sRange};

    // std::cout << "Mat hsv = "<< std::endl << " "  << hsv << std::endl << std::endl;
    cv::calcHist( &newHSV[1], 1, 0, cv::Mat(), histS, 1, &sbins, &ranges, true, false);
    // printf("normalize histS\n");
    // normalize the histogram
    histS /= (int)(src.size().width)*(int)(src.size().height);

    histList.push_back(histS.clone());
    // printf("return hsitlist\n");


    return histList;
}

// apply fourier transform to the source image
// calculate histogram of the fourier transformed image
// https://docs.opencv.org/3.4/d8/d01/tutorial_discrete_fourier_transform.html
cv::Mat hist_whole_fourier(char *path) {
    cv::Mat src, src_gray;
    // read the image
    src = cv::imread(path);
    if(src.data == NULL) {
        printf("Unable to read query image %s\n", path);
        exit(-1);
    }

    // Convert the image to grayscale
    cv::cvtColor(src, src_gray, cv::COLOR_BGR2GRAY);

    // calculate discrete fourier transform
    cv::Mat padded;                            //expand input image to optimal size
    int m = cv::getOptimalDFTSize( src_gray.rows );
    int n = cv::getOptimalDFTSize( src_gray.cols ); // on the border add zero values
    cv::copyMakeBorder(src_gray, padded, 0, m - src_gray.rows, 0, n - src_gray.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
    cv::Mat planes[] = {cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F)};
    cv::Mat complexI;
    merge(planes, 2, complexI);         // Add to the expanded another plane with zeros
    dft(complexI, complexI);            // this way the result may fit in the source matrix
    // compute the magnitude and switch to logarithmic scale
    // => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
    split(complexI, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
    magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
    cv::Mat magI = planes[0];
    magI += cv::Scalar::all(1);                    // switch to logarithmic scale
    cv::log(magI, magI);
    // crop the spectrum, if it has an odd number of rows or columns
    magI = magI(cv::Rect(0, 0, magI.cols & -2, magI.rows & -2));
    // rearrange the quadrants of Fourier image  so that the origin is at the image center
    int cx = magI.cols/2;
    int cy = magI.rows/2;
    cv::Mat q0(magI, cv::Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
    cv::Mat q1(magI, cv::Rect(cx, 0, cx, cy));  // Top-Right
    cv::Mat q2(magI, cv::Rect(0, cy, cx, cy));  // Bottom-Left
    cv::Mat q3(magI, cv::Rect(cx, cy, cx, cy)); // Bottom-Right
    cv::Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
    q2.copyTo(q1);
    tmp.copyTo(q2);
    cv::normalize(magI, magI, 0, 1, cv::NORM_MINMAX); // Transform the matrix with float values into a
                                        // viewable image form (float between values 0 and 1).
    // cv::imshow("Input Image"       , src_gray   );    // Show the result
    // cv::imshow("spectrum magnitude", magI);
    // cv::waitKey(0);

    // calculate histogram
    int histSize = 50;
    float range[] = { 0, 1 }; //the upper boundary is exclusive
    const float* histRange = { range };
    cv::Mat hist;

    cv::calcHist( &magI, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, true, false);
    hist /= (int)(src.size().width)*(int)(src.size().height);



    return hist;
}
