'''
    classify.py
    Mike Zheng and Heidi He
    5/15/19

    classify art type using the vgg-embedding (4096-vector)

    input: image in JPG
    	image_resize
    	vgg19 process
    	classifier
    output: label of type

    python3 classify_image.py ../data
'''

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

import image_resize
import classifiers

import sys
import numpy as np
import csv

# # these two lines are used for running on server
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import cv2

#for drawing image
# from PIL import Image, ImageDraw, ImageFont


def classify(x_test):

    dict = {
        0  :  "painting",
        1  :  "jug",
        2  :  "bowl",
        3  :  "sculpture" ,
        4  :  "drawing" ,
        5  :  "miniature" ,
        6  :  "decorative item" ,
        7  :  "pot" ,
        8  :  "vase" ,
        9  :  "tsuba" ,
        10  :  "dish" ,
        11  :  "scale (object name)" ,
        12  :  "box" ,
        13  :  "can" ,
        14  :  "plate (crockery)" ,
        15  :  "bottle" ,
        16  :  "head" ,
        17  :  "chaire" ,
        18  :  "jewelry" ,
        19  :  "statue" ,
        20  :  "candle holder" ,
        21  :  "kop-en-schotel" ,
        22  :  "Cup and saucer" ,
        23  :  "pastel" ,
        24  :  "tapestry"
    }
    print(dict)

    pred_text = []


    path = '../models/NNClassifier.h5'

    try:
        classifier = load_model(path)
        y_pred = classifier.predict(x_test)

        print("y_pred shape is", y_pred.shape)

        # y_pred_final =  np.empty([len(x_test),1])

        for i,result in enumerate(y_pred):
            pred_index = result.argmax(axis=-1)
            pred_text.append(dict.get(pred_index))
            # y_pred_final[i][0] = result.argmax(axis=-1)

        # y_pred_final = np.array(y_pred_final)
        # y_pred_final = y_pred_final.T

        # print("y_pred_final shape is", y_pred_final.shape)

        return pred_text

    except IOError:
        print("Invalid path")
        sys.exit()



def draw(filepath, type):

    fn = "".join(filepath.split("/")[-1].split(".")[:-1])

    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.imshow(img)
    plt.title(type)
    plt.savefig("../results/testimgs/%s.png"%fn, dpi=300)
    plt.close()



def main(argv):

    # usage
    if len(argv)<2:
        print("Usage: python3 %s <image_directory>"%argv[0])
        exit()

    datafilename = argv[1]

    filelist, data = image_resize.readImgFromDir(datafilename)

    print(filelist)
    print(data.shape)

    base_model = VGG19(weights='imagenet', include_top = True)
    embedding_model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)
    # embedding_model.summary()

    result = embedding_model.predict(data, verbose=1)
    print(result)

    cat_list = classify(result)
    print("new cat is", cat_list)

    for i in range(len(filelist)):
        draw(filelist[i], cat_list[i])





    return

if __name__ == "__main__":
    main(sys.argv)
