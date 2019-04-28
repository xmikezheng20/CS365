# mnist_evaluate.py
# Mike Zheng and Heidi He
# 4/20/19
#
# evaluate the trained network on the first ten digits of the mnist dataset
# and visualize the results
# or evaluate on hand-written images
#
# evaluate on mnist test set first 10 images:
# python3 mnist_evaluate.py ../models/mnist_cnn_simple.h5 0
# evaluate on hand-written digits:
# python3 mnist_evaluate.py ../models/mnist_cnn_simple.h5 1 ../data/digits/

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

import h5py
from keras.models import load_model

import sys
import numpy as np
# import matplotlib.pyplot as plt
import os
import cv2

# return a list of file paths of a given directory
def readdir(dir):
    filelist = []
    for root, directories, filenames in os.walk(dir):
        # for directory in directories:
        #     print(os.path.join(root, directory))
        for filename in filenames:
            if filename[0] != '.':
                filelist.append(os.path.join(root,filename))
    return filelist

def main(argv):

    # usage
    if len(argv)<3:
        print("Usage: python3 %s <model.h5> 0 or python3 %s <model.h5> 1 <directory>" % (argv[0],argv[0]))
        exit()

    # load the model
    print("Loading model from %s" %argv[1])
    model = load_model(argv[1])
    # print("Finished loading model")

    # load mnist test data and predict
    if argv[2] == "0":
        print("Predicting on mnist test data")
        img_rows, img_cols = 28, 28
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        if K.image_data_format() == 'channels_first':
            x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
            input_shape = (1, img_rows, img_cols)
        else:
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
            input_shape = (img_rows, img_cols, 1)

        x_test = x_test.astype('float32')
        x_test /= 255

        # evaluate the model on the first ten test data
        results = model.predict(x_test[:10])

    # load handwritten data, process them, and predict
    elif argv[2] == "1":
        try:
            filelist = readdir(argv[3])
        except:
            print("Unable to read from directory, exiting")
            exit()

        y_test = []
        x_test = []
        for file in filelist:
            # fill y_test
            y_test.append(int(file.split('/')[-1].split('.')[0]))
            # fill x_test
            img = cv2.imread(file)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_resized = cv2.resize(gray, (28, 28))
            img_resized_inverted = np.expand_dims(cv2.bitwise_not(img_resized), axis=2)
            x_test.append(img_resized_inverted)

        # # show processed
        # for i in range(len(x_test)):
        #     plt.imshow(x_test[i], cmap='gray')
        #     plt.title("Digit: "+str(y_test[i]))
        #     plt.show()
        # plt.close()

        # predict
        x_test = np.array(x_test)
        x_test = x_test.astype('float32')
        x_test /= 255
        results = model.predict(x_test)

    else:
        print("Unknown test data source, exiting")
        exit()

    for i,result in enumerate(results):
        s = "Test image %d\nValues: " % i
        for j,val in enumerate(result):
            s += "%d: %.2f; " % (j,val)
        s += "\nPredicted category: %d; True category: %d" % (result.argmax(axis=-1), y_test[i])
        print(s)



if __name__ == "__main__":
    main(sys.argv)
