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

# trainPredictors = predictors[:3100, :]
# trainTarget = target.loc[0:3099]

# testPredictors = predictors[3100:, :]
# testTarget = target.loc[3100:3431]
# testOdds = odds.loc[3100:3431]

n_cols = predictors.shape[1]



model = Sequential()
model.add(Dense(16, activation='relu', input_shape=(n_cols,)))
model.add(Dense(9, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



trainPredictors1 = predictors[:2745, :]
trainTarget1 = target.loc[0:2744]
testPredictors1 = predictors[2745:, :]
testTarget1 = target.loc[2745:3431]
model_training1 = model.fit(trainPredictors1, trainTarget1, epochs = 20, verbose=False)
loss1, acc1 = model.evaluate(testPredictors1, testTarget1)


trainPredictors2 = predictors[:3089, :]
trainTarget2 = target.loc[0:3088]
testPredictors2 = predictors[3089:, :]
testTarget2 = target.loc[3089:3431]
model_training2 = model.fit(trainPredictors2, trainTarget2, epochs = 20, verbose=False)
loss2, acc2 = model.evaluate(testPredictors2, testTarget2)


trainPredictors3 = predictors[:3260, :]
trainTarget3 = target.loc[0:3259]
testPredictors3 = predictors[3260:, :]
testTarget3 = target.loc[3260:3431]
model_training3 = model.fit(trainPredictors3, trainTarget3, epochs = 20, verbose=False)
loss3, acc3 = model.evaluate(testPredictors3, testTarget3)

##################

trainPredictors4 = predictors2[:2745, :]
trainTarget4 = target2.loc[0:2744]
testPredictors4 = predictors2[2745:, :]
testTarget4 = target2.loc[2745:3431]
model_training4 = model.fit(trainPredictors4, trainTarget4, epochs = 20, verbose=False)
loss4, acc4 = model.evaluate(testPredictors4, testTarget4)


trainPredictors5 = predictors2[:3089, :]
trainTarget5 = target2.loc[0:3088]
testPredictors5 = predictors2[3089:, :]
testTarget5 = target2.loc[3089:3431]
model_training5 = model.fit(trainPredictors5, trainTarget5, epochs = 20, verbose=False)
loss5, acc5 = model.evaluate(testPredictors5, testTarget5)


trainPredictors6 = predictors2[:3260, :]
trainTarget6 = target2.loc[0:3259]
testPredictors6 = predictors2[3260:, :]
testTarget6 = target2.loc[3260:3431]
model_training6 = model.fit(trainPredictors6, trainTarget6, epochs = 20, verbose=False)
loss6, acc6 = model.evaluate(testPredictors6, testTarget6)



print("acc1 - " + str(acc1))
print("acc2 - " + str(acc2))
print("acc3 - " + str(acc3))
print("acc4 - " + str(acc4))
print("acc5 - " + str(acc5))
print("acc6 - " + str(acc6))

plt.plot(model_training1.history['acc'], '-r', label = '80%')
plt.plot(model_training2.history['acc'], '-b', label = '90%')
plt.plot(model_training3.history['acc'], '-y', label = '95%')
plt.plot(model_training4.history['acc'], '-g', label = '80% - wymieszane')
plt.plot(model_training5.history['acc'], '-m', label = '90% - wymieszane')
plt.plot(model_training6.history['acc'], '-k', label = '95% - wymieszane')
plt.legend(loc = 'lower right')
plt.title('Models accuracy')
plt.savefig("c:\\Users\\huber\\Desktop\\abcd16.png")







# print(trainPredictors1.shape)
# print(trainTarget1.shape)

# print(testPredictors1.shape)
# print(testTarget1.shape)

# print(predictors.shape)

