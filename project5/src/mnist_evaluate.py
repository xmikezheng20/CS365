# mnist_evaluate.py
# Mike Zheng and Heidi He
# 4/20/19
#
# evaluate the trained network on the first ten digits of the mnist dataset
# and visualize the results
#
# python3 mnist_evaluate.py ../data/mnist_cnn_simple.h5

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
import matplotlib.pyplot as plt

def main(argv):

    # usage
    if len(argv)<2:
        print("Usage: python3 %s <model.h5>" % argv[0])
        exit()

    # load the model
    model = load_model(argv[1])

    # load the data
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
    for i,result in enumerate(results):
        s = "Test image %d\nValues: " % i
        for j,val in enumerate(result):
            s += "%d: %.2f; " % (j,val)
        s += "\nPredicted category: %d; True category: %d" % (result.argmax(axis=-1), y_test[i])
        print(s)



if __name__ == "__main__":
    main(sys.argv)
