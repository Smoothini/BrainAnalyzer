import keras
import os
from keras.layers import Conv1D, LSTM, Dropout, Dense
from keras.utils.np_utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

import brainapplibrary as bap

#disable gpu for tensorflow cuz am too lazy to install cuda drivers
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


before, after, whole = bap.splitChannel(4)

#bap.plotEMG()
#bap.plotAllMRCPonChannel(4)
bap.plotAverageMRCPonChannel(4)

model = keras.models.Sequential()

model.add(Conv1D(1, kernel_size=120, input_shape=(1200,36)))

#model.summary()

#[samples, timesteps, features]
# samples = len(set)
# timesteps = 600
# features = 8


quota = 25

trainX = []
trainy = []
testX = []
testy = []

for i in range(quota):
    t = before[i]
    #k = len(t)
    #split = 8
    #freq = int(k/split) 
    #h = []
    #for j in range(split):
        #h.append(t[j*freq:(j+1)*freq])
    
    trainX.append(t)
    trainy.append(0)
    
    
    t = after[i]
    trainX.append(t)
    trainy.append(1)
    
for i in range(quota, len(before)):
    t = before[i]
    #k = len(t)
    #split = 8
    #freq = int(k/split) 
    #h = []
    #for j in range(split):
        #h.append(t[j*freq:(j+1)*freq])
    
    testX.append(t)
    testy.append(0)
    
    
    t = after[i]
    testX.append(t)
    testy.append(1)
    


trainX=np.array(trainX)
testX=np.array(testX)

trainy=np.array(trainy)
testy=np.array(testy)

trainX = np.reshape(trainX, (50, 2400, 1))
testX = np.reshape(testX, (22, 2400, 1))

trainy = to_categorical(trainy)
testy = to_categorical(testy)

print(trainX.shape)
n_timestep, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
print(n_timestep, n_features, n_outputs)

epochs = 32
batch_size = 8
verbose = 1



model = keras.models.Sequential()
model.add(LSTM(16, input_shape=(n_timestep, n_features)))
#model.add(Dropout(0.5))
model.add(Dense(16, activation='relu'))
model.add(Dense(n_outputs, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
	# evaluate model
_, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=verbose)
print(accuracy)
#X=np.dstack(X)
