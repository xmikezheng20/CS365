'''
    vgg19_test.py
    Mike Zheng and Heidi He
    04/13/2019

    This program takes keras small image classification dataset as input
    and build up a model to use in further network for embedding space.

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
import tensorflow
from keras.backend.tensorflow_backend import set_session
from keras.backend.tensorflow_backend import clear_session
from keras.backend.tensorflow_backend import get_session

# Reset Keras Session
def reset_keras():
    sess = get_session()
    clear_session()
    sess.close()
    sess = get_session()

    try:
        del classifier # this is from global space - change this as you need
    except:
        pass

    # print(gc.collect()) # if it's done something you should see a number being outputted

    # use the same config as you used to create the session
    config = tensorflow.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1
    config.gpu_options.visible_device_list = "0"
    set_session(tensorflow.Session(config=config))

def main(argv):

    ssl._create_default_https_context = ssl._create_unverified_context #debug for system error
    # K.clear_session()
    # reset_keras()
    print("backend cleared")
    dir = "/var/tmp/xzheng20_mhe_cs365_final/data_subset"
    # dir = '/Users/Heidi/Desktop/tmp_dataset'
    input_image_data = image_resize.readImgFromDir(dir)
    input_image_data.astype('float64')

    print(input_image_data.shape)


    print("training model")
    base_model = VGG19(weights='imagenet', include_top = True)
    print("training model 2")
    # base_model.save("../models/vgg19_1000.h5")
    results = base_model.predict(input_image_data, verbose=1)
    print("with 1000 cat")
    print(results.shape)

    embedding_model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)
    embedding_model.summary()
    # embedding_model.save("../models/vgg19_4096.h5")
    results2 = embedding_model.predict(input_image_data, verbose=1)
    print("with 4096 cat")
    print(results2.shape)

    #for debug




if __name__ == "__main__":
    main(sys.argv)
