# -*- coding: utf-8 -*-

import numpy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from time import clock

start_time = clock()

def img2vector(filename):
    returnVect = numpy.zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def loadImages(dirName):
    from os import listdir
    hwLabels = []
    trainingFileList = listdir(dirName)
    m = len(trainingFileList)
    trainingMat = numpy.zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        if classNumStr == 9: hwLabels.append(-1)
        else: hwLabels.append(1)
        trainingMat[i,:] = img2vector('%s/%s' % (dirName, fileNameStr))
    return trainingMat, hwLabels

train_feature, train_output = loadImages('digits/trainingDigits')
test_feature, test_output = loadImages('digits/testDigits')

#kNN algorithm
knn = KNeighborsClassifier()

knn.fit(train_feature,train_output)
predict_output = knn.predict(test_feature)
print("Accuracy is %a" % (accuracy_score(test_output,predict_output)))
print("Running time is %f" % (clock() - start_time))

#SVM algroithm
range_of_gamma = 10.0 ** numpy.arange(-3, 3)
range_of_C = 10.0 ** numpy.arange(-3, 3)
arr = []
kernels = ['linear', 'poly', 'rbf']
best_result = [1000,'',0,0]
support_vector_number = 0

for kernelType in kernels:
    for i in range(len(range_of_gamma)):
        for j in range(len(range_of_C)):
            current_gamma = range_of_gamma[i]
            current_C = range_of_C[j]
            sss = SVC(kernel = kernelType, gamma = current_gamma, C = current_C)
            sss.fit(train_feature, train_output)
            predict_output = sss.predict(test_feature)
            accuracy = accuracy_score(test_output,predict_output)
            test_error = 1.0 - accuracy
            support_vector_number += len(sss.support_vectors_)
            print("Test error: %s | Kernel: %s | Gamma: %s | C: %s" % (
                    test_error, 
                    kernelType,
                    current_gamma,
                    current_C))
            if test_error < best_result[0]:
                best_result = [test_error, kernelType, current_gamma, current_C]
            
print("Min error is %s, kernel is %s, gamma is %s, C is %s" % (best_result[0],best_result[1],best_result[2],best_result[3]))
print("Support Vector number is %s" % (support_vector_number))