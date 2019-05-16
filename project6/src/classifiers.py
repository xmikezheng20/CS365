# classifiers.py
# Mike Zheng and Heidi He
# 4/30/19
# based on Mike's CS251 project
#
# KNN classifier
#
# can use raw intensity data to test the classifier is working
# python3 classifiers.py ../data/greek_training_data.csv ../data/greek_training_labels.csv ../data/greek_testing_data.csv ../data/greek_testing_labels.csv
# which does not work well
#
# the classifier is used in greek_mnist_embedding.py

import sys
import numpy as np
import scipy.spatial
import random

import keras
from keras.models import Model
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout, Flatten


class Classifier:

    def __init__(self, type):
        '''The parent Classifier class stores only a single field: the type of
        the classifier.  A string makes the most sense.

        '''
        self._type = type

    def type(self, newtype = None):
        '''Set or get the type with this function'''
        if newtype != None:
            self._type = newtype
        return self._type

    def accuracy(self, truecats, classcats):
        num = truecats.shape[0]
        correct = 0
        for i in range(num):
            print("pred / true ",int(truecats[i,0]),'/', int(classcats[i,0]))
            if int(truecats[i,0])==int(classcats[i,0]):
                correct+=1
        print("Correct/all = ",correct,"/",num)
        return float(correct)/num

    def confusion_matrix( self, truecats, classcats ):
        '''Takes in two Nx1 matrices of zero-index numeric categories and
        computes the confusion matrix. The rows represent true
        categories, and the columns represent the classifier output.

        '''
        unique = np.unique( np.array(truecats.T) )
        confmtx = np.zeros((len(unique),len(unique)))
        print("")

        for i in range(truecats.shape[0]):
            confmtx[int(truecats[i,0]),int(classcats[i,0])] += 1

        return confmtx

    def confusion_matrix_str( self, cmtx ):
        '''Takes in a confusion matrix and returns a string suitable for printing.'''
        dim = cmtx.shape[0]
        s = ''
        s+="Label  "
        for i in range(dim):
            s+="|   %d"%i
        s+="\n"
        s+=(8+5*dim)*"-"+"\n"
        for i in range(dim):
            s+="True %2d|"%i
            for k in range(dim):
                s+=" %4d"%(int(cmtx[i,k]))
            s+="\n"

        return s

    def __str__(self):
        '''Converts a classifier object to a string.  Prints out the type.'''
        return str(self._type)

# KNN classifier
class KNN(Classifier):

    def __init__(self, data=None, categories=None, K=3):

        # call the parent init with the type
        Classifier.__init__(self, 'KNN Classifier')

        self.num_classes = None
        self.num_features = None
        self.class_labels = None
        self.exemplars = None
        self.K = K

        self.build(data, categories)

    def build( self, A, categories):
        '''Builds the classifier give the data points in A and the categories'''

        unique, mapping = np.unique( np.array(categories.T), return_inverse=True)

        self.num_classes = len(unique)
        self.num_features = A.shape[1]
        self.class_labels = unique

        self.exemplars = []

        for i in range(self.num_classes):
            self.exemplars.append(A[(mapping==i),:])

        return

    def classify(self, A, return_distances=False):
        '''Classify each row of A into one category. Return a matrix of
        category IDs in the range [0..C-1], and an array of class
        labels using the original label values. If return_distances is
        True, it also returns the NxC distance matrix.'''
        K = self.K

        # error check to see if A has the same number of columns as number of features
        if A.shape[1] != self.num_features:
            print("Error! A does not have the same number of columns as num features")
            return


        # make a matrix that is N x C to store the distance to each class for each data point
        D = np.zeros((A.shape[0],self.num_classes)) # a matrix of zeros that is N (rows of A) x C (number of classes)

        # for each class i
            # make a temporary matrix that is N x M where M is the number of examplars (rows in exemplars[i])
            # calculate the distance from each point in A to each point in exemplar matrix i (for loop)
            # sort the distances by row
            # sum the first three columns
            # this is the distance to the first class
        for i in range(self.num_classes):
            tmp = np.zeros((A.shape[0],self.exemplars[i].shape[0]))
            for j in range(A.shape[0]):
                for k in range(self.exemplars[i].shape[0]):
                    tmp[j,k] = scipy.spatial.distance.euclidean(A[j,:], self.exemplars[i][k,:])

            tmp.sort(axis=1)
            D[:,i] = np.sum(tmp[:,:K],axis=1)

        # calculate the most likely class for each data point
        cats = np.matrix(np.argmin(D, axis = 1)).T # take the argmin of D along axis 1 (minimum distance)

        # use the class ID as a lookup to generate the original labels
        labels = self.class_labels[cats]

        if return_distances:
            return cats, labels, D

        return cats, labels

    def __str__(self):
        '''Make a pretty string that prints out the classifier information.'''
        s = "\nKNN Classifier\n"
        for i in range(self.num_classes):
            s += 'Class %d --------------------\n' % (i)
            s += 'Number of Exemplars: %d\n' % (self.exemplars[i].shape[0])
            s += 'Mean of Exemplars  :' + str(np.mean(self.exemplars[i], axis=0)) + "\n"

        s += "\n"
        return s



# Neural Network Cliassifier
class NeuralNet(Classifier):
    def __init__(self, data=None, categories=None):
        # call the parent init with the type
        Classifier.__init__(self, 'Neural Network Classifier')

        self.num_classes = None
        self.num_features = None
        self.num_item = None
        self.class_labels = None

        self.x_train = data
        self.y_train = categories

        self.model = None

        self.build(data, categories)


    def build( self, A, categories):
        unique, mapping = np.unique( np.array(categories.T), return_inverse=True)

        self.num_classes = len(unique)
        print('num of classes is %i', self.num_classes)
        self.num_features = A.shape[1]
        self.num_item = A.shape[0]
        self.class_labels = unique


        return

    def train(self):

        batch_size = 10
        epoches = 8

        # convert class vectors to binary class matrices
        self.y_train = keras.utils.to_categorical(self.y_train, self.num_classes)

        '''classify through neural network with 2 dense layers'''
        classifier = Sequential()
        #First Hidden Layer
        classifier.add(Dense(1024, input_shape=(4096,), activation='relu'))
        #Second  Hidden Layer
        classifier.add(Dense(1024, input_shape=(1024,), activation='relu'))
        #Output Layer
        classifier.add(Dense(self.num_classes, activation='softmax'))
        # classifier.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))

        #Compiling the neural network
        classifier.compile(optimizer ='adam',
                    loss='categorical_crossentropy',
                    metrics =['accuracy'])
        classifier.fit(self.x_train,self.y_train, batch_size=batch_size, epochs=epoches)

        eval_model=classifier.evaluate(self.x_train, self.y_train)#evaluate model
        classifier.save('../models/NNClassifier.h5')
        self.model = classifier
        print(eval_model)

    def getModel(self, path = None):
        if(self.model):
            return self.model
        else:
            try:
                return load_model(path)
            except IOError:
                print("Invalid path")
                sys.exit()

    #classify the the data with given network
    def classify(self, x_test, path = '../models/NNClassifier.h5'):
        classifier = self.getModel(path)
        y_pred = classifier.predict(x_test)
        y_pred_final =  np.empty([self.num_item,1])

        for i,result in enumerate(y_pred):
            # for j,val in enumerate(result):
                # s += "%d: %.2f; " % (j,val)
            pre_cat = result.argmax(axis=-1)
            # s += "\nPredicted category: %d; True category: %d" % (result.argmax(axis=-1), y_test[i])
            # print(s)
            y_pred_final[i][0] = result.argmax(axis=-1)

        # y_pred_final = np.array(y_pred_final)
        # y_pred_final = y_pred_final.T

        print(y_pred_final.shape)

        return y_pred_final


# read file to np matrix
def read_data(filename):

    data = []
    fp = open(filename, 'r')
    buf = fp.readline()
    buf = fp.readline().strip()
    while buf != "":
        words = buf.split(",")
        data.append(words)
        buf = fp.readline().strip()
    fp.close()
    data = np.matrix(data)
    return data


def main(argv):

    # usage
    if len(argv)<5:
        print("Usage: python3 %s <training data> <training labels> <testing data> <testing labels>"%argv[0])
        exit()

    train_data_file = argv[1]
    train_cats_file = argv[2]
    test_data_file = argv[3]
    test_cats_file = argv[4]

    train_data = read_data(train_data_file).astype('float32')
    train_cats = read_data(train_cats_file).astype('int8')
    test_data = read_data(test_data_file).astype('float32')
    test_cats = read_data(test_cats_file).astype('int8')

    nnc= NeuralNet(train_data, train_cats)
    nnc.train()
    print("NN training done")

    print("NN testing data")
    test_new_cats = classify(test_data)
    confmtx_test = nnc.confusion_matrix( test_cats, test_new_cats )
    print( knnc.confusion_matrix_str(confmtx_test) )



    ''' #KNN
    print( 'Building KNN Classifier (K=3)' )
    knnc = KNN( train_data, train_cats, 3)
    # print(knnc)

    print( 'KNN Training Set Results' )

    newcats, newlabels = knnc.classify( train_data)

    confmtx = knnc.confusion_matrix( train_cats, newcats )
    print( knnc.confusion_matrix_str(confmtx) )

    print( 'KNN Test Set Results' )

    newcats, newlabels = knnc.classify(test_data)

    # print the confusion matrix
    confmtx = knnc.confusion_matrix( test_cats, newcats )
    print( knnc.confusion_matrix_str(confmtx) )

    '''

    return


if __name__ == "__main__":
    main(sys.argv)
