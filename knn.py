import keras
import os
from keras.layers import Conv1D, LSTM, Dropout, Dense
from keras.utils.np_utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

import brainapplibrary as bap

#disable gpu for tensorflow cuz am too lazy to install cuda drivers
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


before = []
after = []

for i in range(9):
    before.append([])
    after.append([])
    
    before[i], after[i], _ = bap.splitChannel(i)
    
    
def makeFeature(inpu):
    feats = []
    means = []
    slopes = []
    mm = []
    for channel in inpu:
        channelfeats = []
        for i in channel:
            firstList = i[:5]
            lastList = i[-5:]
            
            meanFirst = sum(firstList) / len(firstList)
            meanLast = sum(lastList) / len(lastList)
            
            slope = meanFirst - meanLast
            mean = sum(i) / len(i)
            maxmindif = max(i) - min(i)
            
            means.append(mean)
            slopes.append(slope)
            mm.append(maxmindif)
            
            feat = [mean, slope, maxmindif]
            #channelfeats.append(feat)
            feats.append(feat)
    return feats, means, slopes, mm
            
bf, means0, slopes0, mm0 = makeFeature(before)
af, means1, slopes1, mm1 = makeFeature(after)

bfy = [0] * len(bf)
afy = [1] * len(af)

means = means0 + means1
slopes = slopes0 + slopes1
mm = mm0 + mm1

features = list(zip(means, slopes, mm))

labels = bfy + afy


testcount = 10
avg = 0
neighbournumber = 5
model = KNeighborsClassifier(n_neighbors=neighbournumber)
for i in range(testcount):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3) # 70% training and 30% test
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    print(f"Run # {i + 1} accuracy:", acc)
    avg += acc
    
print(f"{testcount} runs average accuracy for {neighbournumber} neighbors: {avg/10}")