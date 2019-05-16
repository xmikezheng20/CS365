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
from keras import backend as K

import h5py
import ssl

import sys
import numpy as np
import csv
# these two lines are used for running on server

# # these two lines are used for running on server
import matplotlib
# # Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

import matplotlib.pyplot as plt

def main(argv):

    print("load pretrained model")
    base_model = VGG19(weights='imagenet', include_top = True)
    embedding_model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)
    # embedding_model.summary()

    batch = 100
    i=0
    fp = open("../data/image_resize_data_ordered.csv","r")

    idlist = []
    imglist = []

    buf = fp.readline().strip()
    i+=1

    while buf!="":

        words = buf.split(",")
        id = words[0]
        idlist.append(id)
        img = np.array(words[1:]).astype(np.uint8).reshape((224,224,3))
        img = np.expand_dims(img,axis=3)

        # print(img)
        # plt.imshow(img)
        # plt.savefig("test.png")

        imglist.append(img)

        if i==batch:
            imgbatch = np.concatenate(imglist,axis=3)
            imgbatch = np.transpose(imgbatch, (3,0,1,2))
            # print(imgbatch.shape)
            results = embedding_model.predict(imgbatch, verbose=1)
            # print(results.shape)
            # print()
            resultslist = results.tolist()

            fp_w = open("../data/data_full_embedding_4096.csv","a")
            for j in range(len(idlist)):
                fp_w.write(idlist[j]+",")
                fp_w.write(",".join(str(x) for x in resultslist[j]))
                fp_w.write("\n")

            fp_w.close()

            i=0
            imglist = []
            idlist = []


        i += 1
        buf = fp.readline().strip()

    fp.close()

    imgbatch = np.concatenate(imglist,axis=3)
    imgbatch = np.transpose(imgbatch, (3,0,1,2))
    # print(imgbatch.shape)
    results = embedding_model.predict(imgbatch, verbose=1)
    # print(results.shape)
    # print()
    resultslist = results.tolist()

    fp_w = open("../data/data_full_embedding_4096.csv","a")
    for j in range(len(idlist)):
        fp_w.write(idlist[j]+",")
        fp_w.write(",".join(str(x) for x in resultslist[j]))
        fp_w.write("\n")

    fp_w.close()

    # np.savetxt("../data/data_subset_embedding_4096.csv", output, delimiter=",", fmt="%.6f")
    # np.savetxt("../data/data_full_embedding_4096.csv", output, delimiter=",", fmt="%.6f")

    #for debug




if __name__ == "__main__":
    main(sys.argv)
