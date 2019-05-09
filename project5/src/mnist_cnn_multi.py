# mnist_cnn_simple.py
# Mike Zheng and Heidi He
# 4/20/19
#
# a network with two convolution layers with 32 3x3 filters, a max pooling layer
# with a 2x2 window, a dropout layer with a 0.25 dropout rate, a flatten layer,
# a dense layer with 128 nodes and relu activation, a second dropout layer with
# a 0.5 dropout rate, and a final dense layer for the output with 10 nodes and
# the softmax activation function. When compiling the model, use categorical
# cross-entropy as the loss function and adam as the optimizer. The metric
# should be accuracy.
#
# based on https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
#
# python3 mnist_cnn_multi.py

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

import h5py

import sys
import numpy as np
import csv
# these two lines are used for running on server
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

import matplotlib.pyplot as plt

np.random.seed(42)

class Model():
    def __init__(self):
        self.curModel = Sequential()
        self.num_classes = 10

    def set_input_shape(self,input_shape):
        self.input_shape = input_shape

    def set_num_classes(self, num_classes):
        self.num_classes = num_classes

    def get_model(self):
        return self.curModel

    #train model
    def trainModel(self,poolingSize, filterSize, numFitler1, numFilter2, denseNode):
        # create the model
        print('training model')
        self.curModel.add(Conv2D(numFitler1, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=self.input_shape))
        self.curModel.add(Conv2D(numFilter2, kernel_size=(3, 3), activation='relu'))
        self.curModel.add(MaxPooling2D(pool_size=(poolingSize, poolingSize)))
        self.curModel.add(Dropout(0.25))
        self.curModel.add(Flatten())
        self.curModel.add(Dense(denseNode, activation='relu'))
        self.curModel.add(Dropout(0.5))
        self.curModel.add(Dense(self.num_classes, activation='softmax')) #num of classes

    def saveModel(self, index ):
        name = '../models/mnist_cnn_multi'+str(index)+'.h5'
        print(name)
        self.curModel.save(name)


def write():
    path = '../data/'
    with open(path+'mnist_cnn_multi.csv', 'w') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        #first row - headers
        indexes = 'header,poolingSize,filterSize,numFilter,denseNode'
        for i in range(12) :
            indexes += ',train' + str(i) + ',test' + str(i)


        csvwriter.writerow(indexes)
        csvwriter.writerow(['Spam', 'Spam1', 'spam2'])

    print('writing done')

def main(argv):

    # newModel.prepare()

    batch_size = 128
    num_classes = 10
    epochs = 12 #1 temperarly

    # input image dimensions
    img_rows, img_cols = 28, 28

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    poolingSizeList = [2, 4, 8]
    filterSizeList = [3, 5, 7]
    numbFilterList = [[32,32], [32,64], [32,64], [64, 64]]
    denseNode = [128, 256, 512]
    # poolingSizeList = [2]
    # filterSizeList = [3]
    # numbFilterList = [[32,32], [32,64], [32,64], [64, 64]]
    # denseNode = [128]

    #number index of model
    index = 0
    path = '../data/'
    csvfile = open(path+'mnist_cnn_multi.csv', 'w')
    csvwriter = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    #first row - headers
    indexes = 'header,poolingSize,filterSize,numFilter,denseNode'
    for i in range(epochs) :
        indexes += ',train' + str(i) + ',test' + str(i)
    csvwriter.writerow(indexes)


    #start training models
    for i in poolingSizeList:
        for j in filterSizeList:
            for k in denseNode:
                newModel = Model()
                newModel.set_input_shape(input_shape)
                newModel.trainModel(2, 3, 32, 32, 128)
                # newModel.trainModel(i, j, 32, 32, denseNode)
                # newModel.trainModel(2, 3, 32, 32, 128)

                #compile model
                newModel.get_model().compile(loss=keras.losses.categorical_crossentropy,
                          optimizer=keras.optimizers.Adam(),
                          metrics=['accuracy'])

                # train and evaluate the model
                history = newModel.get_model().fit(x_train, y_train,
                                      batch_size=batch_size,
                                      epochs=epochs,
                                      verbose=1,
                                      validation_data=(x_test, y_test))


                newModel.saveModel(index)
                index += 1

                trainResult = []
                testResult = []
                #write to csv
                # index + poolingSize + filterSize + numFilter + denseNode + train1 + test1 + train2 + test2...
                for m in range(27):
                    curRow = str(index) + ',' + str(i) + ',' + str(j) + ','+ str(32) + ',' + str(k)
                for n in range(epochs):
                    #train results
                    trainAcc = format(history.history['acc'][n],'.4f')
                    trainResult.append(trainAcc)
                    print("training accuracy is")
                    print(trainAcc)
                    #test results
                    testAcc = format(history.history['val_acc'][n], '.4f')
                    testResult.append(testAcc)
                    print("testing accuracy is")
                    print(testAcc)
                    curRow += ',' + str(trainResult[n]) + ',' + str(testResult[n])
                csvwriter.writerow(curRow)

                print('writing done')

    csvfile.close()

    # write()






if __name__ == "__main__":
    main(sys.argv)
