import scipy
from scipy import signal
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import datetime
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix



sampFz = 1200
nyq = 0.5 * sampFz
start = 0.05
fin = 5
delta = (start-fin) / 12

data = sio.loadmat('Data/P10/Cue_Set2.mat')
t = np.transpose(data['data_device1'])

xaxis = data['time_axis_all_device1'][0]
rowEMG = t[13]
testData = t[5]

nyq = 0.5 * sampFz

low = start / nyq
high = fin / nyq

#Creating the butterWorth filter
b,a = signal.butter(2,[low,high], btype='bandpass', analog = False)
y = signal.filtfilt(b, a, testData[1000:])

plt.figure(1)
#Check for what freq are in
yfEMG = np.fft.fft(rowEMG[50:])
xfEMG = np.fft.fftfreq(len(rowEMG[50:]), 1/sampFz)
plt.plot(xfEMG, np.abs(yfEMG))
plt.title("Before highpass filter")


cutOutHz = 20


#Filter out with highpass filter
b,a = signal.butter(10,cutOutHz / (0.5 * sampFz), analog=False, btype="high")
newEMG = signal.lfilter(b,a, rowEMG)

     
#Check if we filter freq out
plt.figure(2)
yfEMGAfter = np.fft.fft(newEMG)
xfEMGAfter = np.fft.fftfreq(len(newEMG), 1/sampFz)
plt.plot(xfEMGAfter, np.abs(yfEMGAfter))
plt.title("After highpass filter")


plt.figure(3)
#Create plot from EMG data 
plt.plot(abs(newEMG[500:]))
plt.title("EMG after filter")

plt.figure(4)
#Create plot from old EMG data
plt.title("EMG before filter")
plt.plot(rowEMG)



#Threshhold for finding EMG burst
threshholdEMG = 0.0007
epochThreshhold = 5000
EMGPeaks = []
counter = 0
index = 0
for i in newEMG:
    counter = counter + 1
    index = index + 1
    if i > threshholdEMG and counter > epochThreshhold:
        EMGPeaks.append(index)
        counter = 0

print(len(EMGPeaks))
print(EMGPeaks)


#Noted in s
behindOffset = 4
aheadOffset = 1

dataFromTrigger = []

def millidelta(x):
    return x.seconds * 1000 + x.microseconds / 1000

def makedate(x):
    return datetime.datetime(year = int(x[0]),
                                   month = int(x[1]),
                                   day = int(x[2]),
                                   hour = int(x[3]),
                                   minute = int(x[4]),
                                   second = int(str(x[5]).split('.')[0]),
                                   microsecond= int(str(x[5]).split('.')[1]) * 1000)


#NEW EMGPEAKS TESTER


tstart = data['time_start_device1'][0]
tend = data['time_stop_device1'][0]

deltastart = makedate(tstart)
print(deltastart)

print((millidelta(makedate(tstart) - makedate(tend))))

triggerpoints = data['TriggerPoint']
tpone = []
for val in triggerpoints:
    if(val[0] == 1):
        tpone.append(makedate(val[1:]))
        
timecueon = []
for val in data["time_cue_on"]:
    timecueon.append(makedate(val))


for i in EMGPeaks: 
    before = i - (behindOffset * sampFz)
    after = i + (aheadOffset * sampFz)
    dataFromTrigger.append(testData[before:after])

#Split the dataFromTrigger into two datasets for active and rest

restData = []
activeData = []

def bandPass(signalDATA):
    fs = 1200
    lowcut = 0.05
    highcut = 5
    
    order = 2
    passtype = 'bandpass'
    
    nyq = 0.5 * fs
    
    low = lowcut / nyq
    high = highcut / nyq
    
    b,a = signal.butter(order, [low, high], passtype, analog = False)
    y = signal.filtfilt(b, a, signalDATA)
    
    return y

for i in dataFromTrigger:
    restData.append(bandPass(i[:2400]))
    activeData.append(bandPass(i[3600:]))
    
    
#We have no split the data into restdata and active
#We now create features for the SVM

xValuesRest = []
yValuesRest = []
cValuesRest = []



for i in restData:
    firstList = i[:5]
    lastList = i[-5:]
    
    meanFirst = sum(firstList) / len(firstList)
    meanLast = sum(lastList) / len(lastList)

    cValuesRest.append(max(i) - min(i))
    
    slope = meanFirst - meanLast
    xValuesRest.append(slope)
    mean = sum(i) / len(i)
    yValuesRest.append(mean)
        
    
#Values for activedata

xValuesActive = []
yValuesActive = []
cValuesActive = []

for i in activeData:
    firstList = i[:5]
    lastList = i[-5:]
    cValuesActive.append(max(i) - min(i))
    
    meanFirst = sum(firstList) / len(firstList)
    meanLast = sum(lastList) / len(lastList)
    
    slope = meanFirst - meanLast
    xValuesActive.append(slope )
    mean = sum(i) / len(i)
    yValuesActive.append(mean + 0.00007)




plt.figure(5)
yf = np.fft.fft(testData)
xf = np.fft.fftfreq(len(testData),1/sampFz)
plt.plot(xf,np.abs(yf))



plt.figure(6)
plt.plot(testData[150000:200000])
plt.title('RAW DATA')


plt.figure(7)
plt.title('BUTTER FITLER')
plt.plot(y[150000:200000])


#Take one of the MCRP information from emg burst
plt.figure(8)
index = EMGPeaks[5] + 600 + 1000
lowindex = index - (sampFz * 4)
highindex = index - (sampFz * 2)
plt.plot(bandPass(testData[lowindex: highindex]))
plt.title("REST")


plt.figure(10)
index = EMGPeaks[5] + 600
lowindex = index - (sampFz * 1) + 500
highindex = index + (sampFz * 1) + 500
plt.plot(bandPass(testData[lowindex: highindex]))
plt.title("MOVEMENT")

plt.figure(9)
plt.scatter(xValuesRest, yValuesRest, c="b", label="Rest")
plt.scatter(xValuesActive, yValuesActive, c="r", label="Active")
plt.ylabel("Mean amp")
plt.xlabel("Slope")
plt.legend()


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
print(len(gTruth))
print(prediction)
print(y_test)

print(confusion_matrix(prediction,y_test))
#plt.legend(bbox_to_anchor=(0.95,0.95), loc="upper left")
#plt.title('Last n values of first 10 channels from CueSet1.mat')

plt.show()
