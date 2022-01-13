from typing import Counter
from scipy.sparse import data
from scipy.sparse.construct import random
import brainapplibrary as bap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
import numpy as np
from sklearn.metrics import classification_report

for x in range(10):
    print("Run number " + str(x+1))

    before = []
    after = []

    for i in range(9):
        before.append([])
        after.append([])
        
        before[i], after[i], _ = bap.splitChannel(i)
    
    if (len(before[0]) < 25):
        print("SIZE IS ONLY, ADJUST PARAMETERS" + str(len(before[0])))
        break
    #Not used atm, we use the featues from the channels
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


    #We have to train 9 indivdual boosting algorithms for each channels, since they are uniq
    combined_x = []
    combined_y = []

    for index, channel in enumerate(range(0,len(before))):
        combined_x.append(before[index] + after[index])
        total_0 = [0] * len(before[index])
        total_1 = [1] * len(after[index])
        combined_y.append(total_0 + total_1)

    #We now train amount of channels, to n different AdaBoosts.
    random_state = np.random.randint(1000000)
    classifers = []
    combined_x_test = []
    combined_y_test = []
    for index, channel in enumerate(range(0, len(combined_x))):
        X_train, X_test, y_train, y_test = train_test_split(combined_x[index], combined_y[index], test_size=0.3, random_state=random_state)
        clf = AdaBoostClassifier(n_estimators=100)
        clf.fit(X_train, y_train)
        classifers.append(clf)
        combined_x_test.append(X_test)
        combined_y_test.append(y_test)


    #We can begin the testing, each time we give it a dataset with do a vote and the majority wins. Since there is 9 channels it is perfect.
    voted_y_values = []
    for n, _ in enumerate(range(0, len(combined_x_test[0]))):
        predictions = []

        for i, _ in enumerate(range(0, len(combined_x_test))):
            reshaped_data = np.reshape(combined_x_test[i][n],(1,-1))
            predictions.append(classifers[i].predict(reshaped_data))
        
        #We now do voting on which is most confident.
        amount_of_0 = predictions.count(0)
        if amount_of_0 > 4:
            voted_y_values.append(0)
        else:
            voted_y_values.append(1)

    print(classification_report(y_test, voted_y_values, labels=[0,1]))
    print("---------------------------------------")

