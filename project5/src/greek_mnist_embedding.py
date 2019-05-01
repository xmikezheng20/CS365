# greek_mnist_embedding.py
# Mike Zheng and Heidi He
# 4/28/19
#
# use mnist network as an embedding space to classify greek letters
#
# greek_mnist_embedding.py ../models/mnist_cnn_simple.h5 ../data/greek_training_data.csv ../data/greek_training_labels.csv ../data/greek_testing_data.csv ../data/greek_testing_labels.csv

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
import classifiers

# read in the data file and return a numpy array
def read_data(path):
    print("Reading data from %s"%path)

    data = []
    fp = open(path, "r")
    buf = fp.readline().strip()
    buf = fp.readline().strip()
    while buf != "":
        sample = []
        words = buf.split(",")
        for word in words:
            sample.append(int(word))
        sample = np.matrix(sample).reshape((28,28))
        # plt.imshow(sample, cmap="gray")
        # plt.savefig("test.png")
        data.append(sample)
        buf = fp.readline().strip()
    fp.close()
    return np.array(data)

# read in the label file and return a list
def read_labels(path):
    print("Reading labels from %s"%path)

    labels = []
    fp = open(path, "r")
    buf = fp.readline().strip()
    buf = fp.readline().strip()
    while buf != "":
        labels.append(int(buf))
        buf = fp.readline().strip()

    return labels

# calculate sum square difference between one sample and a database
def ssd(query, db):
    ssd = []
    for i in range(db.shape[0]):
        ssd.append(np.sum((query-db[i,:])*(query-db[i,:])))
    return ssd

def main(argv):

    # usage
    if len(argv)<6:
        print("Usage: python3 %s <model.h5> <greek_data.csv> <greek_training_labels.csv> <greek_test_data.csv> <greek_test_labels.csv>" % argv[0])
        exit()

    # load the model as an embedding space
    print("Loading model from %s" %argv[1])
    model = load_model(argv[1])
    embedding_model = Model( inputs=model.input, outputs=model.layers[-3].output )
    # embedding_model.summary()

    # read training data and training labels
    greek_training_data_input = read_data(argv[2])
    greek_training_data_input = greek_training_data_input.astype('float32')
    greek_training_data_input /= 255
    greek_training_data_input = np.expand_dims(greek_training_data_input, axis=3)
    print("training input shape: ",greek_training_data_input.shape)

    greek_training_data_output = embedding_model.predict(greek_training_data_input)
    print("training output shape: ",greek_training_data_output.shape)

    # read in labels
    greek_training_labels = read_labels(argv[3])

    # reading testing data and testing labels (Mike's handwritten greek letters)
    greek_testing_data_input = read_data(argv[4])
    greek_testing_data_input = greek_testing_data_input.astype('float32')
    greek_testing_data_input /= 255
    greek_testing_data_input = np.expand_dims(greek_testing_data_input, axis=3)
    print("testing input shape: ",greek_testing_data_input.shape)

    greek_testing_data_output = embedding_model.predict(greek_testing_data_input)
    print("testing output shape: ",greek_testing_data_output.shape)

    # read in labels
    greek_testing_labels = read_labels(argv[5])


    idx2letter = {0:"alpha", 1:"beta", 2:"gamma"}

    # # test ssd
    # print("Testing SSD")
    # print("Calculating ssd with respect to alpha (idx 1)")
    # alpha_exp = greek_training_data_output[1,:]
    # alpha_ssd = ssd(alpha_exp, greek_training_data_output)
    # alpha_argsort = np.argsort(alpha_ssd)
    # for i in alpha_argsort:
    #     print("idx: %2d; label: %s; ssd: %.2f"%(i, idx2letter[greek_training_labels[i]], alpha_ssd[i]))
    #
    # print("Calculating ssd with respect to beta (idx 0)")
    # beta_exp = greek_training_data_output[0,:]
    # beta_ssd = ssd(beta_exp, greek_training_data_output)
    # beta_argsort = np.argsort(beta_ssd)
    # for i in beta_argsort:
    #     print("idx: %2d; label: %s; ssd: %.2f"%(i, idx2letter[greek_training_labels[i]], beta_ssd[i]))
    #
    # print("Calculating ssd with respect to gamma (idx 4)")
    # gamma_exp = greek_training_data_output[4,:]
    # gamma_ssd = ssd(gamma_exp, greek_training_data_output)
    # gamma_argsort = np.argsort(gamma_ssd)
    # for i in gamma_argsort:
    #     print("idx: %2d; label: %s; ssd: %.2f"%(i, idx2letter[greek_training_labels[i]], gamma_ssd[i]))

    # test KNN classifier
    print("Testing KNN classifier")
    training_cats = np.matrix(greek_training_labels).T
    testing_cats = np.matrix(greek_testing_labels).T

    K = 3
    print( 'Building KNN Classifier (K=%d)'%K )
    knnc = classifiers.KNN( greek_training_data_output, training_cats, K)

    print( 'KNN Training Set Results' )

    newcats, newlabels = knnc.classify( greek_training_data_output)

    confmtx = knnc.confusion_matrix( np.matrix(greek_training_labels).T, newcats )
    print( knnc.confusion_matrix_str(confmtx) )

    print( 'KNN Test Set Results' )

    newcats, newlabels, d = knnc.classify(greek_testing_data_output, True)

    # print the confusion matrix
    confmtx = knnc.confusion_matrix( testing_cats, newcats )
    print( knnc.confusion_matrix_str(confmtx) )

if __name__ == "__main__":
    main(sys.argv)
