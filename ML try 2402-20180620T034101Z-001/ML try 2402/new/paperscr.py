from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from Tkinter import Tk
from tkFileDialog import askopenfilename
from sklearn import metrics

data = pd.read_csv('digits2.csv', header=None)
targ = pd.read_csv('digitstarget3.csv', header=None)
test = pd.read_csv('test.csv', header=None)
'''
print data
print targ
print test
'''
clf=KNeighborsClassifier(n_neighbors=3)
train_data= data[:3000]
train_targ= targ[:3000]
train_targ= train_targ.values
test_data= data[3000:]
test_targ= targ[3000:]
test_targ= test_targ.values

print 'Training kNN algorithm'
clf.fit(train_data,train_targ)
print 'Training complete'

print 'Testing algorithm'
'''
result = clf.predict(test_data)
print type(test_targ)
#print result

correct=0
for i in range(result.size):
    if (result[i]==(test_targ[i])):
        correct+=1
accuracy = correct*100.0/result.size
print 'Testing complete, Accuracy = ', float(accuracy),'%'
print(metrics.classification_report(test_targ, result))
print(metrics.confusion_matrix(test_targ, result))
#print test
#i=input('Enter row number from above to test: ')
'''
retest=1
while(retest):
    #print '\n\nDEMO'
    Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
    filename = askopenfilename() # show an "Open" dialog box and return the path to the selected file
    i = int(filename[-5])
    result = clf.predict(test.iloc[[i]])
    print ('The selected image is predicted to be ')
    print result[0]
    retest= input('Test another image?(1/0):')
