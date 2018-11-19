import pandas as pd
import numpy as np
import os
import sklearn
from sklearn import preprocessing
from sklearn.utils import shuffle
import tensorflow as tf
import random

#przygotowanie danych
data = pd.read_csv(os.path.dirname(os.path.abspath(__file__)) + '\stats.csv')
data2 = shuffle(data)


#data = shuffle(data)
predictors = data.iloc[:,:21]
predictors = preprocessing.scale(predictors)
target = pd.DataFrame(0, index = range(data.shape[0]), columns = range(3))
odds = data[['odds-home', 'odds-draw', 'odds-away']]
target.columns = ['HOME', 'DRAW', 'AWAY']

for i in range(3432):
    temp = data.iloc[i,21]
    if(temp==0):
        target.iloc[i, 1] = 1
    elif(temp == 1):
        target.iloc[i, 0] = 1
    else: target.iloc[i, 2] = 1



predictors2 = data2.iloc[:,:21]
predictors2 = preprocessing.scale(predictors2)
target2 = pd.DataFrame(0, index = range(data2.shape[0]), columns = range(3))
odds2 = data2[['odds-home', 'odds-draw', 'odds-away']]
target2.columns = ['HOME', 'DRAW', 'AWAY']

for i in range(3432):
    temp = data2.iloc[i,21]
    if(temp==0):
        target2.iloc[i, 1] = 1
    elif(temp == 1):
        target2.iloc[i, 0] = 1
    else: target2.iloc[i, 2] = 1


import keras
from keras.layers import Dense
from keras.models import Sequential
import matplotlib.pyplot as plt


n_cols = predictors.shape[1]


def testModel(predictors, target, testSize, allSize=3431, n_cols=20):
    model = Sequential()
    model.add(Dense(16, activation='relu', input_shape=(n_cols,)))
    model.add(Dense(9, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    trainPredictors = predictors[:testSize, :]
    trainTarget = target.loc[0:(testSize-1)]
    testPredictors = predictors[testSize:, :]
    testTarget = target.loc[testSize:allSize]
    
    model_training = model.fit(trainPredictors, trainTarget, epochs = 20, verbose=False)
    loss, acc = model.evaluate(testPredictors, testTarget, verbose=False)

    return model_training, loss, acc


model_training1, loss1, acc1 = testModel(predictors=predictors, target=target, testSize=1716, allSize=3431, n_cols=n_cols)
model_training2, loss2, acc2 = testModel(predictors=predictors, target=target, testSize=2402, allSize=3431, n_cols=n_cols)
model_training3, loss3, acc3 = testModel(predictors=predictors, target=target, testSize=3089, allSize=3431, n_cols=n_cols)
model_training4, loss4, acc4 = testModel(predictors=predictors2, target=target2, testSize=1716, allSize=3431, n_cols=n_cols)
model_training5, loss5, acc5 = testModel(predictors=predictors2, target=target2, testSize=2402, allSize=3431, n_cols=n_cols)
model_training6, loss6, acc6 = testModel(predictors=predictors2, target=target2, testSize=3089, allSize=3431, n_cols=n_cols)


print("acc1 - " + str(acc1))
print("acc2 - " + str(acc2))
print("acc3 - " + str(acc3))
print("acc4 - " + str(acc4))
print("acc5 - " + str(acc5))
print("acc6 - " + str(acc6))

plt.plot(model_training1.history['acc'], '-r', label = '50%')
plt.plot(model_training2.history['acc'], '-b', label = '70%')
plt.plot(model_training3.history['acc'], '-y', label = '90%')
plt.plot(model_training4.history['acc'], '-g', label = '50% - wymieszane')
plt.plot(model_training5.history['acc'], '-m', label = '70% - wymieszane')
plt.plot(model_training6.history['acc'], '-k', label = '90% - wymieszane')
plt.legend(loc = 'lower right')
plt.title('Models accuracy')
plt.savefig(os.path.dirname(os.path.abspath(__file__)) + '\plot2.png')
