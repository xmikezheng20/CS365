# mnist_layer_analysis.py
# Mike Zheng and Heidi He
# 4/27/19
#
# examine the first layers of the mnist_cnn_simple network
#
# python3 mnist_layer_analysis.py ../models/mnist_cnn_simple.h5
# tested to run on dwarves

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

import h5py
from keras.models import load_model
from keras.models import Model

import sys
import numpy as np

# these two lines are used for running on server
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

import matplotlib.pyplot as plt

import os
import cv2

# load an image from the training set
def load_img(idx):
    img_rows, img_cols = 28, 28
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype('float32')
    x_train /= 255

    return x_train[idx]

def main(argv):

    # usage
    if len(argv)<2:
        print("Usage: python3 %s <model.h5>" % argv[0])
        exit()

    # load the model
    print("Loading model from %s" %argv[1])
    model = load_model(argv[1])

    # analyze first layer filters
    # plot filters
    fig = plt.figure()
    fig.suptitle("Layer 0 filters")
    layer0 = model.layers[0].get_weights()[0]
    for i in range(32):
        filter = layer0[:,:,:,i][:,:,0]
        ax = fig.add_subplot(8, 4, i+1)
        ax.imshow(filter,cmap="gray")
        ax.axis("off")
    # plt.savefig("../results/layer0.png", dpi=300)

    # apply filters to first training sample
    idx = 0
    fig = plt.figure()
    fig.suptitle("Layer 0 filters on training sample %d"%idx)
    img = load_img(idx)
    for i in range(32):
        filter = layer0[:,:,:,i][:,:,0]
        dst = cv2.filter2D(img, -1, filter)
        ax = fig.add_subplot(8, 4, i+1)
        ax.imshow(dst,cmap="gray")
        ax.axis("off")
    # plt.savefig("../results/layer0_filters_training%d.png" %idx, dpi=300)

    img_3dim = np.expand_dims(img, axis=2)
    img_4dim = np.expand_dims(img_3dim, axis=0)

    # first layer model
    first_layer_model = Model( inputs=model.input, outputs=model.layers[0].output )
    first_layer_model_output = first_layer_model.predict(img_4dim)
    print("first layer model output shape: ",first_layer_model_output.shape)
    fig = plt.figure()
    fig.suptitle("Layer 0 model on training sample %d"%idx)
    for i in range(32):
        ax = fig.add_subplot(8, 4, i+1)
        ax.imshow(first_layer_model_output[0,:,:,i],cmap="gray")
        ax.axis("off")
    # plt.savefig("../results/layer0_model_training%d.png" %idx, dpi=300)

    # first two layers model (conv+conv)
    first_two_layer_model = Model( inputs=model.input, outputs=model.layers[1].output )
    first_two_layer_model_output = first_two_layer_model.predict(img_4dim)
    print("first two layers model output shape: ",first_two_layer_model_output.shape)
    fig = plt.figure()
    fig.suptitle("Layer 0+1 model on training sample %d"%idx)
    for i in range(32):
        ax = fig.add_subplot(8, 4, i+1)
        ax.imshow(first_two_layer_model_output[0,:,:,i],cmap="gray")
        ax.axis("off")
    # plt.savefig("../results/layer01_model_training%d.png" %idx, dpi=300)

    # first three layers model (conv+conv+pooling)
    first_three_layer_model = Model( inputs=model.input, outputs=model.layers[2].output )
    first_three_layer_model_output = first_three_layer_model.predict(img_4dim)
    print("first three layers model output shape: ",first_three_layer_model_output.shape)
    fig = plt.figure()
    fig.suptitle("Layer 0+1+2 model on training sample %d"%idx)
    for i in range(32):
        ax = fig.add_subplot(8, 4, i+1)
        ax.imshow(first_three_layer_model_output[0,:,:,i],cmap="gray")
        ax.axis("off")
    # plt.savefig("../results/layer012_model_training%d.png" %idx, dpi=300)




if __name__ == "__main__":
    main(sys.argv)
