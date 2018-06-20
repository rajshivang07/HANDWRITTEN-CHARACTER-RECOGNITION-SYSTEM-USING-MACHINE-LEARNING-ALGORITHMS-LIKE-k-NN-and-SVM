from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
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

#print features

clf=KNeighborsClassifier(n_neighbors=2)

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
'''
retest=1
while (retest):
    imageno = input('Enter image index to display')

    plt.imshow(digits.images[imageno],interpolation='nearest')
    plt.show()
    retest= input('Test another image?(1/0):')

retest=1
while(retest):
    print '\n\nDEMO'
    Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
    filename = askopenfilename() # show an "Open" dialog box and return the path to the selected file

    img = misc.imread(filename)
    img = misc.imresize(img, (8,8))
    img = img.astype(digits.images.dtype)
    img = misc.bytescale(img, high=16, low=0)


    x_test = []

    for eachRow in img:
        for eachPixel in eachRow:
            x_test.append(sum(eachPixel)/3.0)

    print(clf.predict([x_test]))
    retest= input('Test another image?(1/0):')
'''
