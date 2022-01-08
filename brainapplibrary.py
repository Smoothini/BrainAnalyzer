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
    
def plotAverageMRCPonChannel(index):
    rest, active, whole = splitChannel(index)

    #All the stuff bellow here shows a plot of the average MRCP's on the selected channel above
    xrest = range(2400)
    xactive = range(3600, 6000)
    xwhole = range(6000)

    minima = np.argmin(np.average(whole, axis = 0))
    minima_active = np.argmin(np.average(active, axis = 0)) + 3600

    plt.plot(xwhole, np.average(whole, axis = 0))
    plt.plot(xrest, np.average(rest, axis = 0), label = "Rest state average")
    plt.plot(xactive, np.average(active, axis = 0), label = "Active state average")
    plt.axvline(4800, -2, 2, linestyle='dashed', label = "EMG Peak Location")
    plt.axvline(minima, -2, 2, linestyle='dotted', label = "Avg. full MRCP minima")
    plt.axvline(minima_active, -2, 2, color='red', linestyle='dotted', label = "Avg. active phase of MRCP minima")
    plt.legend()
    plt.show()
    
    
def plotAllMRCPonChannel(index):
    rest, active, whole = splitChannel(index)
    rows = int(len(whole)**0.5)
    cols = int(len(whole)/rows)
    
    #currentPlot = 0
    #xwhole = range(6000)
    
    for currentPlot in range(len(whole)):     
        minima = np.argmin(whole[currentPlot])
        minima_active = np.argmin(active[currentPlot]) + 3600
        plt.subplot(rows, cols, currentPlot + 1)
        plt.plot(whole[currentPlot])
        plt.axvline(4800, -2, 2, linestyle='dashed', label = "EMG Peak Location")
        plt.axvline(minima, -2, 2, linestyle='dotted', label = "Average whole EEG Minima Location")
        plt.axvline(minima_active, -2, 2, color='red', linestyle='dotted', label = "Average active EEG Minima Location")
        
    #plt.tight_layout()
    plt.show()
    
    print("Legend for the graph of all MRCP\'s on the channel:")
    print(f" * Brain channel: {index + 1}")
    print(" * Dashed line: detected EMG peak")
    print(" * Blue dotted line: MRCP minima")
    print(" * Red dotted line: detected active phase of MRCP minima")

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


