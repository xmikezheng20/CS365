# greek_data_processing.py
# Mike Zheng and Heidi He
# 4/27/19
#
# process greek data images to data+label
#
# python3 greek_data_processing.py ../data/greek/ ../data/

import sys
import os
import cv2
# these two lines are used for running on server
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

import matplotlib.pyplot as plt

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

def main(argv):

    # usage
    if len(argv)<3:
        print("Usage: python3 %s <image dir> <output dir>" % argv[0])
        exit()

    # read image directory
    try:
        filelist = readdir(argv[1])
    except:
        print("Unable to read from directory, exiting")
        exit()

    letter2idx = {"alpha":0, "beta":1, "gamma":2}

    # open files to write
    if argv[2][-1]=="/":
        path = argv[2]
    else:
        path = argv[2]+"/"
    fp_d = open(path+"greek_data.csv","w")
    s = ""
    for i in range(784):
        s+="intensity%d," % i
    s = s[:-1]+"\n"
    fp_d.write(s)
    fp_l = open(path+"greek_labels.csv", "w")
    fp_l.write("category\n")

    # go through filelist and write to csv
    for file in filelist:

        # write labels
        idx = letter2idx[file.split('/')[-1].split('.')[0].split('_')[0]]
        fp_l.write("%d\n"%idx)

        # write data
        img = cv2.imread(file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_resized = cv2.resize(gray, (28, 28))
        img_resized_inverted = cv2.bitwise_not(img_resized)

        # plt.imshow(img_resized_inverted, cmap="gray")
        # plt.savefig("test.png")
        s = ""
        for i in range(img_resized_inverted.shape[0]):
            for j in range(img_resized_inverted.shape[1]):
                s+=str(img_resized_inverted[i,j])+","
        s = s[:-1]+"\n"
        fp_d.write(s)


    fp_d.close()
    fp_l.close()

if __name__ == "__main__":
    main(sys.argv)
