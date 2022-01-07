import matplotlib.pyplot as plt
import brainapplibrary as bap
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

#bap.plotBandpassDifference()
#bap.plotEMG()
#bap.plotFilteredSignals()
bap.plotEMGwithPeaks()

#rest - list of all rest phases
#active - list of all active phases
#whole - list of all mrcp movements(one movement is rest + active + the tiny region in between)
# bap.splitChannel will get you all the lists above from the channel u write

rest, active, whole = bap.splitChannel(4)



#All the stuff bellow here shows a plot of the average MRCP's on the selected channel above
xrest = range(2400)
xactive = range(3600, 6000)
xwhole = range(6000)

minima = np.argmin(np.average(whole, axis = 0))

plt.plot(xwhole, np.average(whole, axis = 0))
plt.plot(xrest, np.average(rest, axis = 0), label = "Rest state average")
plt.plot(xactive, np.average(active, axis = 0), label = "Active state average")
plt.axvline(4800, -2, 2, linestyle='dashed', label = "EMG Peak Location")
plt.axvline(minima, -2, 2, linestyle='dotted', label = "Average EEG Minima Location")
plt.legend()
plt.show()


avg = abs(4800-minima)
print(f"Average delta between EMG peak and EEG minima: {avg/1200}s or {avg} points")

##################################
#All the stuff bellow here is pretty much copy paste from the svm you did
#with the addition of tpr and ppv calculation

xValuesRest = []
yValuesRest = []
cValuesRest = []

for i in rest:
    firstList = i[:5]
    lastList = i[-5:]
    cValuesRest.append(max(i) - min(i))
    
    meanFirst = sum(firstList) / len(firstList)
    meanLast = sum(lastList) / len(lastList)
    
    slope = meanFirst - meanLast
    xValuesRest.append(slope)
    mean = sum(i) / len(i)
    yValuesRest.append(mean)
    

xValuesActive = []
yValuesActive = []
cValuesActive = []

for i in active:
    firstList = i[:5]
    lastList = i[-5:]
    cValuesActive.append(max(i) - min(i))
    
    meanFirst = sum(firstList) / len(firstList)
    meanLast = sum(lastList) / len(lastList)
    
    slope = meanFirst - meanLast
    xValuesActive.append(slope )
    mean = sum(i) / len(i)
    yValuesActive.append(mean + 0.00007)
    
#plt.scatter(xValuesRest, yValuesRest, c="b", label="Rest")
#plt.scatter(xValuesActive, yValuesActive, c="r", label="Active")
#plt.ylabel("Mean amp")
#plt.xlabel("Slope")
#plt.legend()



xValues = []
gTruth = []

#Creating the datasets
count = 0
for x in xValuesActive:
    xValues.append([x, yValuesActive[count]])
    gTruth.append(0)
    count = count + 1

restDataSVM = []

#Creating the datasets
count = 0
for x in xValuesRest:
    xValues.append([x, yValuesRest[count]])
    gTruth.append(1)
    count = count + 1

#Traning our SVM
clf = svm.SVC()

#split into traning and test
X_train, X_test, y_train, y_test = train_test_split(xValues, gTruth, test_size=0.25, random_state=42)
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
#print(len(gTruth))
#print(prediction)
#print(y_test)
print("Confusion matrix:")

cm = confusion_matrix(prediction,y_test)

print(cm)

tn, fp, fn, tp = cm.ravel()

ppv = tp / (tp + fp)
tpr = tp / (tp + fn)

print(f'TPR: {tpr}')
print(f'PPV: {ppv}')