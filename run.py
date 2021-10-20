import scipy
import scipy.io as sio
from scipy.fft import fft, fftfreq, irfft
from scipy.signal import butter, lfilter, filtfilt
import numpy as np
import matplotlib.pyplot as plt
import datetime
import pandas as pd


def bandPass(signal):
    fs = 1200
    lowcut = 0.05
    highcut = 5
    
    order = 2
    passtype = 'bandpass'
    
    nyq = 0.5 * fs
    
    low = lowcut / nyq
    high = highcut / nyq
    
    b,a = butter(order, [low, high], passtype, analog = False)
    y = filtfilt(b, a, signal)
    
    return y

def highPass(signal):
    fs = 1200
    cutoff = 80
    
    order = 4
    passtype = 'highpass'
    
    nyq = 0.5 * fs
    
    high = cutoff / nyq
    
    b,a = butter(order, high, passtype, analog = False)
    y = filtfilt(b, a, signal)
    
    return y
    

def millidelta(x):
    return x.seconds * 1000 + x.microseconds / 1000

def h2m(x):
    return x/1000*1200

def makedate(x):
    return datetime.datetime(year = int(x[0]),
                                   month = int(x[1]),
                                   day = int(x[2]),
                                   hour = int(x[3]),
                                   minute = int(x[4]),
                                   second = int(str(x[5]).split('.')[0]),
                                   microsecond= int(str(x[5]).split('.')[1]) * 1000)

    
setname = 'Cue_Set1.mat'
pset = 'P10'
path = f'Data/{pset}/{setname}'


data = sio.loadmat(path)
#actual = len(np.transpose(data['time_axis_all_device1'][0])) + 1

t = np.transpose(data['data_device1'])

tstart = data['time_start_device1'][0]
tend = data['time_stop_device1'][0]

deltastart = makedate(tstart)

deltaend = makedate(tend)

datasize = len(t[0])

print("delta in miliseconds: ", millidelta(deltaend - deltastart))
print("occurences: ", h2m(millidelta(deltaend - deltastart)))

triggerpoints = data['TriggerPoint']
tpone = []
for val in triggerpoints:
    if(val[0] == 1):
        tpone.append(makedate(val[1:]))
        
timecueon = []
for val in data["time_cue_on"]:
    timecueon.append(makedate(val))
    

plt.suptitle(f"Average of filtered brain signals using trigger points on {setname} from {pset}")

filteredValues = []
#if you set this to 600, the drop is in the middle, dont ask me why cuz idfk
offset = 0

for i in range(9):
    filteredValues.append([])
    for val in tpone:
        arra = []
        #current = int(h2m(millidelta(val - deltastart)))
        
        before = val - datetime.timedelta(seconds = 2)
        beforei = int(h2m(millidelta(before - deltastart))) + offset
        
        after = val + datetime.timedelta(seconds = 2)
        afteri = int(h2m(millidelta(after - deltastart)))  + offset
    
        arra = t[i][beforei:afteri]
        filteredValues[i].append(bandPass(arra))
      

def plotAverageButter():
    xsec = np.arange(-2, 2, 0.00083333)[:4800]        
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.title(f"Channel {i+1}")
        plt.plot(xsec, np.average(filteredValues[i], axis=0))
        
#needs the power and treshold
def plotEMG():
    plt.subplot(2, 1, 1)
    plt.title(f"Channel {14}")
    plt.plot(t[13])
    
    plt.subplot(2, 1, 2)
    plt.title(f"Channel {14} filtered")
    plt.plot(highPass(t[13]))
    xdim = [-0.002] * 30
    check = list(map((lambda x: int(h2m(millidelta(x - deltastart)))), tpone))
    plt.plot(check, xdim, marker = 'p')
    

plotAverageButter()
#plotEMG()
plt.show()
