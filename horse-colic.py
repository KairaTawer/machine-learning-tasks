from sklearn import neighbors
from sklearn import linear_model
from sklearn import svm
from sklearn.ensemble import BaggingClassifier,VotingClassifier,AdaBoostClassifier
import matplotlib.pyplot as plt
from sklearn import preprocessing
import numpy
from sklearn.metrics import accuracy_score

lookup = '##'
train_data = []
test_data = []

with open('horse-colic.data.txt') as f:
    for line in f:
        if not line.startswith(lookup):
            train_data.append(line.rstrip("\r\n"))
            
with open('horse-colic.test.txt') as f:
    for line in f:
        if not line.startswith(lookup):
            test_data.append(line.rstrip("\r\n"))

input_test = []
target_test = []
for i in test_data:
    results = i.split("\t")
    results = [float(i) for i in results]
    target_test.append(results.pop())
    input_test.append(results[6:8])

input_test = numpy.asarray(input_test)
input_test = preprocessing.scale(input_test)
target_test = numpy.asarray(target_test)

input_train = []
target_train = []
for i in train_data:
    results = i.split("\t")
    results = [float(i) for i in results]
    target_train.append(results.pop())
    input_train.append(results[6:8])

input_train = numpy.asarray(input_train)
input_train = preprocessing.scale(input_train)
target_train = numpy.asarray(target_train)

bag = BaggingClassifier(base_estimator=neighbors.KNeighborsClassifier())
bag.fit(input_train, target_train)
target_predict = bag.predict(input_test)
print(accuracy_score(target_test,target_predict))

ada = AdaBoostClassifier()
ada.fit(input_train, target_train)
target_predict = ada.predict(input_test)
print(accuracy_score(target_test,target_predict))

kNNReg = neighbors.KNeighborsClassifier()
logisticReg = linear_model.LogisticRegression()
sVMReg = svm.SVC(probability=True)

maj = VotingClassifier(estimators=[('lr', logisticReg), ('svc', sVMReg), ('knn', kNNReg)],voting='soft')

from sklearn.metrics import roc_curve
from sklearn.metrics import auc
colors = ['black', 'orange', 'blue', 'green']
linestyles = [':', '--', '-.', '-']
for clf, label, clr, ls \
    in zip([logisticReg,sVMReg,kNNReg,maj], ['lr','svc','knn','maj'], colors, linestyles):
    # assuming the label of the positive class is 1
    y_pred = clf.fit(input_train,target_train).predict_proba(input_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_true=target_test,y_score=y_pred)
    roc_auc = auc(x=fpr, y=tpr)
    plt.plot(fpr, tpr,color=clr,linestyle=ls,label='%s (auc = %0.2f)' % (label, roc_auc))
plt.legend(loc='lower right')
plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.grid()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

colors = ['black', 'orange']
linestyles = [':', '--']
for clf, label, clr, ls \
    in zip([ada,bag], ['ada','bag'], colors, linestyles):
    # assuming the label of the positive class is 1
    y_pred = clf.fit(input_train,target_train).predict_proba(input_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_true=target_test,y_score=y_pred)
    roc_auc = auc(x=fpr, y=tpr)
    plt.plot(fpr, tpr,color=clr,linestyle=ls,label='%s (auc = %0.2f)' % (label, roc_auc))
plt.legend(loc='lower right')
plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.grid()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()