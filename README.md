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

<h1> Project 3: 2D Object Recognition </h1>
