# mnist_view.py
# Mike Zheng and Heidi He
# 4/20/19
#
# view images of digits in the mnist dataset
#
# python3 mnist_view.py (default 10)
# or
# python3 mnist_view.py 5

import sys

import keras
from keras.datasets import mnist

import numpy as np
import matplotlib.pyplot as plt

def main(argv):

    if len(argv)>1:
        numDigit = int(argv[1])
    else:
        numDigit = 10

    # load data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # plot data
    for i in range(numDigit):
        plt.imshow(x_train[i], cmap='gray')
        plt.title("Digit: "+str(y_train[i]))
        plt.show()
    plt.close()


if __name__ == "__main__":
    main(sys.argv)
