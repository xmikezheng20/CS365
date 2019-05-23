# CS365
Colby College CS365 SP19 Projects

Mike Zheng and Heidi He
- [Project 2: Content-Based Image Retrieval](https://github.com/xzheng902/CS365/blob/master/README.md#project-2-content-based-image-retrieval)
- [Project 3: 2D Object Recognition](https://github.com/xzheng902/CS365/blob/master/README.md#project-3-2d-object-recognition)
- [Project 4: Calibration and Augmented Reality](https://github.com/xzheng902/CS365/blob/master/README.md#project-4-calibration-and-augmented-reality)
- [Project 5: Recognition using deep network](https://github.com/xzheng902/CS365/blob/master/README.md#project-5-recognition-using-deep-network)
- [Project 6: Deep neural network in artistic data classification](https://github.com/xzheng902/CS365/blob/master/README.md#project-6-deep-neural-network-in-artistic-data-classification)


***
# Project 2: Content-Based Image Retrieval

# Instructions
### Compilation and Running
Makefile - GUI : make cbirgui

Run: ../bin/cbirgui ../data/&lt;queryImageName&gt; &lt;database&gt;

Makefile - CommandLine: make cbir

Run: ../bin/cbir ../data/&lt;queryImageName&gt; &lt;database&gt; &lt;# of image to report&gt; &lt;method index&gt;

### GUI:
the trackbar on top of the images enables a user to select matching methods, according to the following rules:

0- baseline matching

1- baseline hue-saturation histogram matching

2- multiple histograms matching

3- Sobel filtering + color matching

4- color matching utilizing earth mover distance metric (expensive, takes 20 mins & to be improved)

5- Laws filter subset + color matching

6- Fourier transform + color matching (logically unsound)

7- rgb + saturation in weighted average

### project structure
#### cbirgui (main):
recursively read files from a directory.

display GUI

process query image

process database image

show result for top 20 matches and update.

(comparator for comparing similarity.)

#### Img class:
each image class object contains:

an image class has a path to the original image,

a flag for checked/unchecked,

a similarity value that shows its similarity to the query image. The larger the value is, the better the match it is.

When comparing pictures, call related histogram function from the histogram library. And calculate the distance metric within the image class. Return the similarity value to the main function.

#### Histogram library:
the class that processes all histogram functions. the functions usually take in a given path to an image and return a histogram.

#### Compilation and Running
Makefile: make cbirgui

Run: ../bin/cbirgui ../data/&lt;queryImageName&gt; &lt;database&gt;

[back to top](https://github.com/xzheng902/CS365/blob/master/README.md#cs365)

***
# Project 3: 2D Object Recognition

### Compilation and Running
Makefile - CommandLine: make objRec

Run:

 for video: ../bin/objRec ../data/&lt;database&gt; 0

for still images: ../bin/objRec ../data/&lt;database&gt; 1 ../../../&lt;training or testing directory&gt;/

### Project Structure:
#### objRec (main)
The main program identifies command line inputs to switch between video mode/ image mode & training mode/ testing mode.

Mode 0 is video mode while mode 1 is image mode.

State 0 is the training mode while state 1 is the testing mode.

It either trains the model with the given image directory and writes to the database, or recognize the images in a given directory/ streaming video.

When the user opens the OR system, if the user chooses to process still images, then there are training and testing modes. If the objectDB exists, then the system automatically enters testing mode. By pressing b, the user can switch to training mode, where features are collected and label (from file name) is put into the database. The user can pressing c, the user can get back to testing mode. If the user chooses video input, then the system can only do testing.

#### Processing
a library of 2d image processing functions:

morphological operations

find contour of regions,

discard small region,

extract features

visualization

#### Classifier Library
parent class: Classifier

the parent class has a field "type" that indicates which classifier to use;

methods include getting the ojbect's dictionary,  construct and print confusion matrix.

three child classes contain three classifiers: Euclidean Distance, KNN, and Naive Bayes

some helper functions

#### Classifier Test
a helper file that tests each classifier class. Not a part of the main program.

[back to top](https://github.com/xzheng902/CS365/blob/master/README.md#cs365)

***
# Project 4: Calibration and Augmented Reality

### Compilation and Running
augmented:
    make augmented
    ../bin/augmented ../data/params.txt

ar_aruco:
    make ar_aruco
    ../bin/ar_aruco ../data/params.txt

ar_img:
    make ar_img
    ../bin/ar_img &lt;image directory&gt; &lt;params.txt&gt;

ar_harris:
    make ar_harris
    ../bin/ar_harris &lt;params.txt&gt;

ar_harris_detect:
    make ar_harris_detect
    ../bin/ar_harris_detect &lt;params.txt&gt;

calib
    make calib
    ../bin/calib

Run:
    for video: ../bin/objRec ../data/&lt;database&gt; 0

    for still images: ../bin/objRec ../data/&lt;database&gt; 1 ../../.. &lt;training or testing directory&gt;/

### Project Structure:
augmented(main)
read parameters to get the camera matrix and distortion coefficient

get/update extrinsic parameters

open camera and initiate window

detect mark

draw objects

ar_aruco
similar to augmented.cpp but using the Aruco module instead

ar_harris
detect harris corner, find iphone target and apply augmented reality

ar_harris_detect
detect harris corner

ar_img ( augmented reality from image)
read images from a given directory

draw/update shapes on detected targets in each image

shape
draw shapes with given intrinsic and extrinsic parameters

calib
camera calibration

write to params.txt on camera matrix and distortion coefficient

[back to top](https://github.com/xzheng902/CS365/blob/master/README.md#cs365)

***
# Project 5: Recognition using deep network
### Project Structure:
Simple CNN training:
a network with two convolution layers with 32 3x3 filters, a max pooling layer with a 2x2 window, a dropout layer with a 0.25 dropout rate, a flatten layer, a dense layer with 128 nodes and relu activation, a second dropout layer with a 0.5 dropout rate, and a final dense layer for the output with 10 nodes and the softmax activation function. When compiling the model, use categorical cross-entropy as the loss function and adam as the optimizer. The metric should be accuracy.

python3 mnist_cnn_simple.py

Multi Simple CNN training:
Train the networks several times with 108 types of parameters. Then output a csv file the accuracy in each epoch on the training model and test set.

python3 mnist_cnn_multi.py

Evaluate Mnist Test:
Evaluate the trained network on the first ten digits of the mnist dataset and visualize the results or evaluate on hand-written images

evaluate on mnist test set first 10 images:
    python3 mnist_evaluate.py ../models/mnist_cnn_simple.h5 0
evaluate on hand-written digits:
    python3 mnist_evaluate.py ../models/mnist_cnn_simple.h5 1 ../data/digits/

Layer analysis mnist:
examine the first layers of the mnist_cnn_simple network

python3 mnist_layer_analysis.py ../models/mnist_cnn_simple.h5

View mnist:
view images of digits in the mnist dataset

python3 mnist_view.py (default 10) or python3 mnist_view.py 5

CNN Gabor:
a network with a fixed 32 gabor filter first layer, a convolution layers with 32 3x3 filters, a max pooling layer with a 2x2 window, a dropout layer with a 0.25 dropout rate, a flatten layer, a dense layer with 128 nodes and relu activation, a second dropout layer with a 0.5 dropout rate, and a final dense layer for the output with 10 nodes and the softmax activation function. When compiling the model, use categorical cross-entropy as the loss function and adam as the optimizer. The metric should be accuracy.

python3 mnist_cnn_garbor.py

Greek MNIST embedding:
use mnist network as an embedding space to classify greek letters

 greek_mnist_embedding.py ../models/mnist_cnn_simple.h5 ../data/greek_training_data.csv ../data/greek_training_labels.csv ../data/greek_testing_data.csv ../data/greek_testing_labels.csv

Data processing for greek letters:
process greek data images to data+label

python3 greek_data_processing.py ../data/greek/ ../data/

Classifier:
A KNN classifier.  It can use raw intensity data to test the classifier. the classifier is used in greek_mnist_embedding.py

python3 classifiers.py ../data/greek_training_data.csv ../data/greek_training_labels.csv ../data/greek_testing_data.csv ../data/greek_testing_labels.csv

[back to top](https://github.com/xzheng902/CS365/blob/master/README.md#cs365)

***
# Project 6: Deep neural network in artistic data classification
### Proposal:
Summary:
Classifying images from artistic data is an interesting and important task. Due to the amount of data and the interest of the public, accurate classification and identification of artworks is highly relevant. Artistic data also have distinct characteristics, making the problem challenging. While images of artworks usually are taken in museums, the environment and scale can be somewhat controlled. There are also several major categories of artworks, such as painting and sculpture. Though the images between categories are highly different, images of the same category can be very similar. Training a network to handle both the inter-category classification and intra-category classification can be challenging.

The Rijksmuseum Challenge is an artistic image classification challenge in 2014 by the Rijksmuseum in Amsterdam, Netherlands. The dataset consists of 112,039 photographic reproductions of the artworks and there are four classification challenges: artist, type, material, and year. While there are more recent artistic datasets, we think the size of this dataset is suitable for this project. The original paper uses a SVM-based classification, and we propose to use a pre-trained deep network to improve the classification.

Resources:
The Rijksmuseum Challenge (2014):

 Dataset: https://figshare.com/articles/Rijksmuseum_Challenge_2014/5660617 (14.49GB)

 Reference: The Rijksmuseum Challenge: Museum-Centered Visual Recognition, Thomas Mensink and Jan van Gemert, in ACM International Conference on Multimedia Retrieval (ICMR) 2014 https://ivi.fnwi.uva.nl/isis/publications/2014/MensinkICMIR2014/MensinkICMIR2014.pdf

Tasks:
Task 1: Download the dataset and pre-process the images properly to create the input of the convolutional neural network.

Task 2: Load a keras pre-trained VGG network/ResNet network and enable it to process the input data.

Task 3: Figure out an embedding method with the network and implement classifiers on top of it to classify artist, type, material, and year.

Task 4: Compare results between different networks with the original paper (SVM).

Possible task 5: fine-tune the network and try to improve performance.

Final Task: Presentation and write-up.

### Report:
[Final report](https://wiki.colby.edu/pages/viewpage.action?pageId=432964714)

[back to top](https://github.com/xzheng902/CS365/blob/master/README.md#cs365)
