# CS365
Colby College CS365 SP19 Projects

Mike Zheng and Heidi He

<h1> Project 2: Content-Based Image Retrieval </h1>

<h1> Instructions </h1>
<h2>Compilation and Running </h2>
Makefile - GUI : make cbirgui

Run: ../bin/cbirgui ../data/<queryImageName> <database>

Makefile - CommandLine: make cbir

Run: ../bin/cbir ../data/&lt;queryImageName&gt; &lt;database&gt; &lt;# of image to report&gt; &lt;method index&gt;

<h2>GUI: </h2>
the trackbar on top of the images enables a user to select matching methods, according to the following rules:

0- baseline matching

1- baseline hue-saturation histogram matching

2- multiple histograms matching

3- Sobel filtering + color matching

4- color matching utilizing earth mover distance metric (expensive, takes 20 mins & to be improved)

5- Laws filter subset + color matching

6- Fourier transform + color matching (logically unsound)

7- rgb + saturation in weighted average

<h1>project structure</h1>
<h2>cbirgui (main):</h2>
recursively read files from a directory.

display GUI

process query image

process database image

show result for top 20 matches and update.

(comparator for comparing similarity.)

<h2>Img class:</h2>
each image class object contains:

an image class has a path to the original image,

a flag for checked/unchecked,

a similarity value that shows its similarity to the query image. The larger the value is, the better the match it is.

When comparing pictures, call related histogram function from the histogram library. And calculate the distance metric within the image class. Return the similarity value to the main function.

<h2>Histogram library:</h2>
the class that processes all histogram functions. the functions usually take in a given path to an image and return a histogram.

<h2>Compilation and Running</h2>
Makefile: make cbirgui

Run: ../bin/cbirgui ../data/<queryImageName> <database>
  
</br>

<h1> Project 3: 2D Object Recognition </h1>

<h2>Compilation and Running </h2>
Makefile - CommandLine: make objRec

Run: 

 for video: ../bin/objRec ../data/<database> 0

for still images: ../bin/objRec ../data/<database> 1 ../../../<training or testing directory>/

<h2>Project Structure:</h2>
<h2>objRec (main)</h2>
The main program identifies command line inputs to switch between video mode/ image mode & training mode/ testing mode.

Mode 0 is video mode while mode 1 is image mode.

State 0 is the training mode while state 1 is the testing mode.

It either trains the model with the given image directory and writes to the database, or recognize the images in a given directory/ streaming video. 

When the user opens the OR system, if the user chooses to process still images, then there are training and testing modes. If the objectDB exists, then the system automatically enters testing mode. By pressing b, the user can switch to training mode, where features are collected and label (from file name) is put into the database. The user can pressing c, the user can get back to testing mode. If the user chooses video input, then the system can only do testing.

<h2>Processing </h2>
a library of 2d image processing functions:

morphological operations

find contour of regions,

discard small region,

extract features

visualization

<h2>Classifier Library </h2>
parent class: Classifier

the parent class has a field "type" that indicates which classifier to use; 

methods include getting the ojbect's dictionary,  construct and print confusion matrix.

three child classes contain three classifiers: Euclidean Distance, KNN, and Naive Bayes

some helper functions

<h2>Classifier Test </h2>
a helper file that tests each classifier class. Not a part of the main program.


