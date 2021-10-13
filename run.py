import scipy
import scipy.io as sio
from scipy.fft import fft, fftfreq, irfft
from scipy.signal import butter, lfilter, filtfilt
import numpy as np
import matplotlib.pyplot as plt
import datetime
import pandas as pd

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def bandPass(signal):
    fs = 1200
    lowcut = 0.5
    highcut = 10
    
    nyq = 0.5 * fs
    
    low = lowcut / nyq
    high = highcut / nyq
    
    order = 2
    
    b,a = butter(order, [low, high], 'bandpass', analog = False)
    y = filtfilt(b, a, signal, axis=0)
    
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


#def m2h(x):
    
setname = 'Cue_Set1.mat'
pset = 'P10'
path1 = f'Data/{pset}/{setname}'

start = 0.8
fin = 0.2
delta = (start-fin) / 10

data = sio.loadmat(path1)
t = np.transpose(data['data_device1'])

tstart = data['time_start_device1'][0]
tend = data['time_stop_device1'][0]

deltastart = makedate(tstart)

deltaend = makedate(tend)

datasize = len(t[0])
timeseries = pd.date_range(start = deltastart, periods = datasize, freq = "0.83333L" )


print("delta in miliseconds: ", millidelta(deltaend - deltastart))
print("occurences: ", h2m(millidelta(deltaend - deltastart)))

triggerpoints = data['TriggerPoint']
tpone = []
tpzero = []
for val in triggerpoints:
    if(val[0] == 1):
        tpone.append(makedate(val[1:]))
    else:
        tpzero.append(makedate(val[1:]))
        
timecueon = []
for val in data["time_cue_on"]:
    timecueon.append(makedate(val))
    
#plt.subplot(3,1,1)

#for i in [0, 13]:
#    plt.plot(timeseries, t[i], label = f'Channel {i+1}', color = f'{start - delta*i}')
#    minimax = (min(t[i][1000:]) + max(t[i][1000:])) / 2
#    plt.plot(tpone, [minimax] * len(tpone), marker = 'p', linestyle = 'None', label="Trigger=1")
#    plt.plot(timecueon, [minimax] * len(timecueon), marker = 'x', linestyle = 'None', label = "Time Cue On")
    
    

#plt.legend(bbox_to_anchor=(0.95,0.95), loc="upper left")
#plt.title(f'{setname} from {pset}')


#plt.subplot(3,1,2)


#b,a = scipy.signal.butter(2, [0.00353,0.00875], 'bandpass')
#filtered = butter_bandpass_filter(t[0], min(t[0]), max(t[0]), 1200, 2)

#yf = fft(t[0])
#xf = fftfreq(len(t[0]), 1/1200)
#newsig = irfft(yf)
#for i in range(len(newsig)):
#    newsig[i] = newsig[i] * -1
#plt.title("Inverse Fast Fourier Transform?")
#plt.plot(newsig)
#filtered = butter_bandpass_filter(t[0], min(t[0]), max(t[0]), 1200, 2)
#plt.plot(b,a)

#plt.subplot(3,1,2)

#plt.title("Butter on Channel")
#filtered = butter_bandpass_filter(newsig, min(newsig), max(newsig), 1200, 2)
#filtered = bandPass(t[0])
#plt.plot(filtered, label = "Butterworth bandpass")
#i = 0
#plt.plot(t[0], label = f'Channel {i+1}', color = f'{start - delta*i}')
#plt.legend(bbox_to_anchor=(0.95,0.95), loc="upper left")

 

#plt.subplot(3,1,3)

plt.title("Butter stuff")

butt = []
for val in tpone:
    arra = []
    current = int(h2m(millidelta(val - deltastart)))
    
    before = val - datetime.timedelta(seconds = 2)
    beforei = int(h2m(millidelta(before - deltastart)))
    
    after = val + datetime.timedelta(seconds = 2)
    afteri = int(h2m(millidelta(after - deltastart)))

    arra = t[0][beforei:afteri]
    butt.append(arra)
    
def rm(x):
    arrr = range(4800)
    rez = []
    for val in arrr:
        rez.append(val + 4800 * x + 50 * x)
    return rez

j = 1    
for val in butt:
    plt.plot(rm(j), val)
    t = bandPass(val)
    nt = []
    for num in t:
        num += 
        nt.append(num)
    plt.plot(rm(j), nt)
    j+=1

plt.show()
