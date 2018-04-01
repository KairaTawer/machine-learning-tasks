import numpy as np
import pickle
import os
from sklearn.linear_model import SGDClassifier

class classifier:

    def saveClassifier(self, clf):
        pickle.dump(clf, open(os.path.join(os.path.dirname(__file__), 'classifier.pkl'), 'wb'), protocol=4)

    def fetchClassifier(self):
        if not os.path.exists(os.path.join(os.path.dirname(__file__), 'classifier.pkl')):
            clf = self.constructClassifier()
            self.saveClassifier(clf)
            return clf
        else:
            clf = pickle.load(open(os.path.join(os.path.dirname(__file__), 'classifier.pkl'), 'rb'))
            return clf

    def img2vector(self,filename):
        returnVect = np.zeros((1,1024))
        fr = open(filename)
        for i in range(32):
            lineStr = fr.readline()
            for j in range(32):
                returnVect[0,32*i+j] = int(lineStr[j])
        return returnVect

    def retrainClassifier(self):
        import sqlite3
        conn = sqlite3.connect('digits.sqlite')
        c = conn.cursor()
        c.execute('SELECT * FROM digits')
        l = c.fetchall()
        hwLabels=[]
        m = len(l)
        trainingMat = np.zeros((m,1024))
        asd = open('classifierUpgraded.txt','w')
        asd.write('upgrade')
        asd.close()
        for i in range(m):
            trainingMat[i,:] = self.img2vector('testDigits/%s' % l[i][0])
            hwLabels.append(1 if l[i][1] == 1 else -1)
        clf = pickle.load(open(os.path.join(os.path.dirname(__file__), 'classifier.pkl'), 'rb'))
        clf.partial_fit(trainingMat,hwLabels)
        self.saveClassifier(clf)
        conn.commit()
        conn.close()

    def loadImages(self,dirName):
        from os import listdir
        hwLabels = []
        trainingFileList = listdir(dirName)
        m = len(trainingFileList)
        trainingMat = np.zeros((m,1024))
        for i in range(m):
            fileNameStr = trainingFileList[i]
            fileStr = fileNameStr.split('.')[0]
            classNumStr = int(fileStr.split('_')[0])
            if classNumStr == 9: hwLabels.append(-1)
            else: hwLabels.append(1)
            trainingMat[i,:] = self.img2vector('%s/%s' % (dirName, fileNameStr))
        return trainingMat, hwLabels

    def constructClassifier(self):
        feature_train, label_train = self.loadImages("trainingDigits")
        feature_test, label_test = self.loadImages("testDigits")
        clf = SGDClassifier()
        clf.fit(feature_train,label_train)
        return clf