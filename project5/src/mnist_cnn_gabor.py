# mnist_cnn_gabor.py
# Mike Zheng and Heidi He
# 5/2/19
#
# a network with a fixed 32 gabor filter first layer,
# a convolution layers with 32 3x3 filters, a max pooling layer
# with a 2x2 window, a dropout layer with a 0.25 dropout rate, a flatten layer,
# a dense layer with 128 nodes and relu activation, a second dropout layer with
# a 0.5 dropout rate, and a final dense layer for the output with 10 nodes and
# the softmax activation function. When compiling the model, use categorical
# cross-entropy as the loss function and adam as the optimizer. The metric
# should be accuracy.
#
# based on https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
#
# python3 mnist_cnn_simple.py

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

# these two lines are used for running on server
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

import matplotlib.pyplot as plt

import cv2

np.random.seed(42)

# generate gabor filter kernels as inital kernels of untrainable first layer
def gabor_1channel_32(shape, dtype=None):
    gabor_kernels = []
    for i in range(shape[3]):
        kernel = cv2.getGaborKernel(ksize=(shape[0], shape[1]), sigma=3, theta=180.0/(i+1), lambd=1, gamma=1, psi=0,
                           ktype=cv2.CV_64F)
        kernel = np.expand_dims(kernel, axis=2)
        gabor_kernels.append(kernel)
    gabor_kernels = np.transpose(np.array(gabor_kernels), (1,2,3,0))
    # print(gabor_kernels.shape)
    # for i in range(shape[3]):
    #     print(gabor_kernels[:,:,0,i])
    #     print()
    return K.variable(gabor_kernels, dtype=dtype)

# draw gabor filters
def drawGaborFilters(model):
    fig = plt.figure()
    fig.suptitle("Layer 0 filters")
    layer0 = model.layers[0].get_weights()[0]
    for i in range(32):
        filter = layer0[:,:,:,i][:,:,0]
        ax = fig.add_subplot(8, 4, i+1)
        ax.imshow(filter,cmap="gray")
        ax.axis("off")
    plt.savefig("../results/gabor_filters.png", dpi=300)
    plt.close(fig)


def main(argv):

    batch_size = 128
    num_classes = 10
    epochs = 12

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

    # create the model
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5),
                     kernel_initializer=gabor_1channel_32,
                     activation='relu',
                     input_shape=input_shape,
                     trainable=False))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    # compile the model
    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

    drawGaborFilters(model)
    # model.summary()

    # train and evaluate the model
    history = model.fit(x_train, y_train,
                          batch_size=batch_size,
                          epochs=epochs,
                          verbose=1,
                          validation_data=(x_test, y_test))

    model.save('../models/mnist_cnn_gabor.h5')

    # Plot training & validation accuracy values
    # https://keras.io/visualization/#training-history-visualization
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig("../results/mnist_cnn_gabor_training.png", dpi=300)
    # plt.show()




if __name__ == "__main__":
    main(sys.argv)
