'''
    classify.py
    Mike Zheng and Heidi He
    5/15/19

    classify art type using the vgg-embedding (4096-vector)

    python3 classify.py ../data/data_subset_embedding_4096_reorder.csv ../data/metadata_subset_first_reorder.csv

'''

import sys
import classifiers
import numpy as np

def readlabels(filename):

    # read labels
    dict = {}
    labels = []
    fp = open(filename, "r")
    buf = fp.readline().strip()
    buf = fp.readline().strip()
    while buf!="":
        words = buf.split(",")
        type = words[2]
        if type not in dict:
            dict[type] = len(dict)
        labels.append(dict[type])
        buf = fp.readline().strip()
    fp.close()

    labelsmat = np.matrix(labels).T

    return labelsmat, dict


def main(argv):

    # usage
    if len(argv)<3:
        print("Usage: python3 %s <data.csv> <metadata.csv>"%argv[0])
        exit()

    datafilename = argv[1]
    metadatafilename = argv[2]

    # read data
    datamat = np.genfromtxt(datafilename, delimiter=',')
    # print(datamat)
    print(datamat.shape)

    data = datamat[:,1:].astype(np.float32)
    # print(data)
    print(data.shape)

    # read labels
    labelsmat, dict = readlabels(metadatafilename)
    print(dict)
    # print(labelsmat)
    print(labelsmat.shape)

    # unique, counts = np.unique(labelsmat, return_counts=True, axis=0)
    # print(unique)
    # print(counts)

    # CLASSIFY
    K = 3
    print( 'Building KNN Classifier (K=%d)'%K )
    knnc = classifiers.KNN( data, labelsmat, K)

    print( 'KNN Training Set Results' )

    newcats, newlabels = knnc.classify( data)

    # confmtx = knnc.confusion_matrix( np.matrix(labelsmat).T, newcats )
    # print( knnc.confusion_matrix_str(confmtx) )
    for i in range(labelsmat.shape[0]):
        print(labelsmat[i,0], newcats[i,0])




if __name__ == "__main__":
    main(sys.argv)
