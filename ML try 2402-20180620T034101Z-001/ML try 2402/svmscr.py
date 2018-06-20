# -*- coding: cp1252 -*-
from sklearn import datasets
from sklearn import svm
import pandas as pd
import numpy as np
from sklearn import metrics
from Tkinter import Tk
from tkFileDialog import askopenfilename
from scipy import misc
import matplotlib.pyplot as plt

digits= datasets.load_digits()

features=digits.data
labels=digits.target

#print (features,labels)

clf=svm.SVC(kernel='linear')

print 'Dataset size is ',labels.size
print ('We will split it into 50-50 training-test split')

train_features= features[:900]
train_labels= labels[:900]

test_features= features[900:]
test_labels= labels[900:]
print 'Training kNN algorithm'
clf.fit(train_features,train_labels)
print train_labels
print 'Training complete'

print 'Testing algorithm'
result = clf.predict(test_features)
#print result

correct=0
for i in range(result.size):
    if (result[i]==test_labels[i]):
        correct+=1
accuracy = correct*100.0/result.size
print 'Testing complete, Accuracy = ', float(accuracy),'%'
print(metrics.classification_report(test_labels, result))
print(metrics.confusion_matrix(test_labels, result))
print(metrics.confusion_matrix(test_labels, result)[1,2])
