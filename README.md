# CS365
Colby College CS365 SP19 Projects

Mike Zheng and Heidi He
- [Project 2: Content-Based Image Retrieval](https://github.com/xzheng902/CS365/blob/master/README.md#project-2-content-based-image-retrieval)
- [Project 3: 2D Object Recognition](https://github.com/xzheng902/CS365/blob/master/README.md#project-3-2d-object-recognition)


***
# Project 2: Content-Based Image Retrieval

# Instructions
### Compilation and Running
Makefile - GUI : make cbirgui

Run: ../bin/cbirgui ../data/<queryImageName> <database>

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

Run: ../bin/cbirgui ../data/<queryImageName> <database>

[back to top](https://github.com/xzheng902/CS365/blob/master/README.md#cs365)

***
# Project 3: 2D Object Recognition

### Compilation and Running
Makefile - CommandLine: make objRec

Run:

 for video: ../bin/objRec ../data/<database> 0

for still images: ../bin/objRec ../data/<database> 1 ../../../<training or testing directory>/

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
