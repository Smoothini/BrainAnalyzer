import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt


ignored = 5
sampleRate = 1200

PeakOffset = 0

EMGchannel = 13
EMGepochSize = 5000
EMGthreshold = 0.0007
EMGoffset = ignored + PeakOffset

offsetBefore = 4
offsetAfter = 1

path = 'Data/P10/Cue_Set1.mat'
rawdata = sio.loadmat(path)


def removeNoiseAtStart(rawinput):
    rawrawsignals = np.transpose(rawinput['data_device1'])
    rawsignals = []
    
    for sgn in rawrawsignals:
        rawsignals.append((sgn[ignored:]))
        
    return rawsignals, rawsignals[EMGchannel]   

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
    
    #print(f"nyq: {high}")
    
    b,a = butter(order, high, passtype, analog = False)
    y = filtfilt(b, a, signal)
    
    return y

def lowPass(signal):
    fs = 1200
    cutoff = 5
    
    order = 4
    passtype = 'lowpass'
    
    nyq = 0.5 * fs
    
    high = cutoff / nyq
    
    print(f"nyq: {high}")
    
    b,a = butter(order, high, passtype, analog = False)
    y = filtfilt(b, a, signal)
    
    return y


def multiBandPass(signals):
    """Bandpass those hoes"""
    fs = []
    for sgn in signals:
        fs.append(bandPass(sgn))
    return fs


def plotRawSignals():
    plt.title("Overview on raw input data")
    for i in range(9):
        plt.subplot(3, 3, i+1)
        plt.title(f"Channel {i+1}")
        plt.plot(rawsignals[i])
    plt.show()
    
def plotFilteredSignals():
    plt.title("Overview on raw input data")
    for i in range(9):
        plt.subplot(3, 3, i+1)
        plt.title(f"Channel {i+1}")
        plt.plot(filteredsignals[i])
    plt.show()
    
def plotBandpassDifference():
    plt.title("Overview on bandpass filter")
    plt.subplot(2, 1, 1)
    plt.title("Raw signal data")
    plt.plot(rawsignals[3][20000:24000])
    plt.subplot(2, 1, 2)
    plt.title("Bandpass filtered signal data")
    plt.plot(filteredsignals[3][20000:24000])
    plt.show()

def plotEMG():
    plt.subplot(2, 1, 1)
    plt.title("Raw EMG data")
    plt.plot(rawEMG)
    plt.subplot(2, 1, 2)
    plt.title("Highpass filtered EMG data")
    plt.plot(filteredEMG)
    plt.show()
    
def plotEMGwithPeaks():
    ydim = [EMGthreshold] * len(peaks)
    plt.title("Highpass filtered EMG data with peaks from threshold")
    plt.plot(filteredEMG)
    plt.plot(peaks, ydim, marker='p', linestyle='None', label='EMG peaks')
    plt.legend()
    plt.show()
    

def findEMGpeaks(signal):
    peaks = []
    pointindex = 0
    epochindex = 0
    for point in signal:
        if point > EMGthreshold and epochindex > EMGepochSize:
            peaks.append(pointindex + EMGoffset)
            epochindex = 0
        epochindex += 1
        pointindex += 1
    return peaks

def splitChannel(index):
    """Splits a signal channel into rest and active phases based on EMG peaks.
    ----
    Parameters:
        index: The index of the channel (eg. 5 for channel 6)
    ----
    Output:
        rest: List of all rest phase lists from that channel
        active: list of all active phase lists from that channel"""
    signal = filteredsignals[index]
    restactive = []
    rest = []
    active = []
    for peak in peaks:
        before = peak - offsetBefore * sampleRate
        after = peak + offsetAfter * sampleRate
        
        restactive.append(signal[before:after])
        
        rest.append(signal[before : before + sampleRate*2])
        active.append(signal[peak - sampleRate : after])
    
    return rest, active, restactive




rawsignals, rawEMG = removeNoiseAtStart(rawdata)

filteredsignals = multiBandPass(rawsignals)
filteredEMG = highPass(rawEMG)

peaks = findEMGpeaks(filteredEMG)


