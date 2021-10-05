import scipy
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

start = 0.8
fin = 0.2
delta = (start-fin) / 10

data = sio.loadmat('Cue_Set1.mat')
t = np.transpose(data['data_device1'])

xaxis = data['time_axis_all_device1'][0]

tcueon = np.transpose(data['time_cue_on'])
wtf = []
dlt = ((25*60 + 25.077) - 3)
for i in range (len(tcueon[4])):
    wtf.append(tcueon[4][i] * 60 + tcueon[5][i] - dlt)

#plt.plot(xaxis, t[12][-len(xaxis):], label = f'Channel 13', color = 'r')
#for i in [10, 11]: 
for i in [9]:#range(10): 
    plt.plot(xaxis, t[i][-len(xaxis):], label = f'Channel {i+1}', color = f'{start - delta*i}')
    mini = (min(t[i][-len(xaxis):]) + max(t[i][-len(xaxis):])) / 2
    plt.plot(wtf, [mini]*len(wtf), marker='P')    
    
plt.legend(bbox_to_anchor=(0.95,0.95), loc="upper left")
plt.title('Last n values of first 10 channels from CueSet1.mat')
plt.show()

#print(xaxis[0])
#print(xaxis[-1])
#print((xaxis[-1]-xaxis[0])/(1/1200))

startdevice = 25*60 + 4.081
stopdevice = 30*60 + 39.268
#print((stopdevice-startdevice)/(1/1200))

tcueon = np.transpose(data['time_cue_on'])
wtf = []
dlt = ((25*60 + 25.077) - 3)
for i in range (len(tcueon[4])):
    wtf.append(tcueon[4][i] * 60 + tcueon[5][i] - dlt)
    
#print(wtf)