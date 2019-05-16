'''
    vgg19_embedding.py
    Mike Zheng and Heidi He
    04/13/2019

    This program takes artistic database as input
    and use pre-trained model.
    output a #images*1000 matrix

    We use vgg19

    sort using shell command: sort -t"," -k1n,1 ../data/data_subset_embedding_4096.csv > ../data/data_subset_embedding_4096_reorder.csv
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

def getIds(filelist):
    idlist = []
    for filename in filelist:
        idlist.append(int(filename.split('/')[-1].split('_')[0]))

    return np.matrix(idlist).T


def main(argv):

    # ssl._create_default_https_context = ssl._create_unverified_context #debug for system error
    #
    # dir = "/var/tmp/xzheng20_mhe_cs365_final/data_subset/"
    # dir = "/var/tmp/xzheng20_mhe_cs365_final/jpg2/"
    # dir = '../data/tmp_dataset' #local test dataset
    dir = "/var/tmp/xzheng20_mhe_cs365_final/data_final/data_10000/"
    filelist, input_image_data = image_resize.readImgFromDir(dir)
    idmatrix = getIds(filelist)

    print("load pretrained model")
    base_model = VGG19(weights='imagenet', include_top = True)
    embedding_model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)
    # embedding_model.summary()

    results = embedding_model.predict(input_image_data, verbose=1)
    print(results.shape)

    output = np.hstack((idmatrix.astype(np.float32),results.astype(np.float32)))
    print(output.shape)

    np.savetxt("/var/tmp/xzheng20_mhe_cs365_final/data_final/process/data_embedding_4096.csv", output, delimiter=",", fmt="%.6f")

    # np.savetxt("../data/data_subset_embedding_4096.csv", output, delimiter=",", fmt="%.6f")
    # np.savetxt("../data/data_full_embedding_4096.csv", output, delimiter=",", fmt="%.6f")

    #for debug




if __name__ == "__main__":
    main(sys.argv)
