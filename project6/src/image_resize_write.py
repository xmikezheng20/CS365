# image_resize.py
# Mike Zheng and Heidi He
# 5/14/19
#
# read directory of jpg files and resize them to 224*224*3 (channel last)
#
# python3 image_resize.py /Users/xiaoyuezheng/Desktop/Rijksmuseum_data_raw/data_subset

import sys
import os
import numpy as np

# # these two lines are used for running on server
# import matplotlib
# # Force matplotlib to not use any Xwindows backend.
# matplotlib.use('Agg')

import matplotlib.pyplot as plt
import cv2

# return a list of file paths of a given directory
def readdir(dir):
    filelist = []
    for root, directories, filenames in os.walk(dir):
        # for directory in directories:
        #     print(os.path.join(root, directory))
        for filename in filenames:
            if filename[0] != '.':
                filelist.append(os.path.join(root,filename))
    return filelist

# read all images from a directory and put them into a numpy matrix (n*224*224*3)
def readImgFromDir(dir):
    filelist = readdir(dir)
    imglist = []
    for file in filelist:
        print("Processing ",file)
        img = readResizeImage(file)
        img = np.expand_dims(img,axis=3)
        imglist.append(img)

    # fp = open("../data/image_resize_data.csv","w")
    fp = open("/var/tmp/xzheng20_mhe_cs365_final/image_resize_data.csv","w")
    for i in range(len(imglist)):
        fp.write(str(filelist[i].split('/')[-1].split('_')[0])+",")
        fp.write(",".join(str(x) for x in imglist[i].reshape((1,150528)).tolist()[0]))
        fp.write("\n")
    fp.close()

    return filelist, imglist


# read a single image from a path and resize into
def readResizeImage(path):
    # read the image
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    shape = img.shape
    # print(path," has dimension ",shape)

    # resize it to 224*224
    # resize short side to 224
    if shape[0]>shape[1]:
        scale = 224/shape[1]
        img_resized = cv2.resize(img, (224,int(shape[0]*scale)))
        # print("resized image has dimension ",img_resized.shape)
        # crop the middle of the long side
        mid = int(img_resized.shape[0]/2)
        # print(mid)
        img_cropped = img_resized[mid-112:mid+112,:,:]

    elif shape[1]>shape[0]:
        scale = 224/shape[0]
        img_resized = cv2.resize(img, (int(shape[1]*scale),224))
        # print("resized image has dimension ",img_resized.shape)
        # crop the middle of the long side
        mid = int(img_resized.shape[1]/2)
        # print(mid)
        img_cropped = img_resized[:, mid-112:mid+112,:]
    else:
        scale = 224/shape[0]
        img_resized = cv2.resize(img,(224,224))
        # print("resized image has dimension ",img_resized.shape)
        img_cropped=img_resized

    # print("cropped image has dimension ",img_cropped.shape)

    # plt.imshow(img)
    # plt.show()
    # plt.imshow(img_resized)
    # plt.show()
    # plt.imshow(img_cropped)
    # plt.show()
    # exit()

    return img_cropped


def main(argv):

    # usage
    if len(argv)<2:
        print("Usage: python3 %s <image data dir>", argv[0])
        exit()

    dir = argv[1]

    filelist, data = readImgFromDir(dir)

    # print(data.shape)



if __name__ == "__main__":
    main(sys.argv)
