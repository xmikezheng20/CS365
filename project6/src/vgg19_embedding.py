'''
    vgg19_embedding.py
    Mike Zheng and Heidi He
    04/13/2019

    This program takes artistic database as input
    and use pre-trained model.
    output a #images*1000 matrix

    We use vgg19
'''

from __future__ import print_function
import keras
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten
from keras.datasets import cifar10
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

import h5py
import ssl

import sys
import numpy as np
import csv
# these two lines are used for running on server
# import matplotlib
# Force matplotlib to not use any Xwindows backend.

import image_resize


def main(argv):

    # ssl._create_default_https_context = ssl._create_unverified_context #debug for system error
    #
    dir = "/var/tmp/xzheng20_mhe_cs365_final/data_subset"
    # dir = '../data/tmp_dataset' #local test dataset
    input_image_data = image_resize.readImgFromDir(dir)
    input_image_data.astype('float64')
    print(input_image_data.shape)

    print("load pretrained model")
    base_model = VGG19(weights='imagenet', include_top = True)
    embedding_model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)
    embedding_model.summary()
    # model_path = '../models/vgg19_1000.h5'
    # model = load_model(model_path)
    results = embedding_model.predict(input_image_data, verbose=1)
    print("with 4096 cat")
    print(results.shape)

    np.savetxt("../data/dataset_matrix.csv", results, delimiter=",")

    #for debug




if __name__ == "__main__":
    main(sys.argv)
