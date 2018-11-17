import pandas as pd
import numpy as np
import os
import sklearn
from sklearn import preprocessing
from sklearn.utils import shuffle
import tensorflow as tf

#przygotowanie danych
data = pd.read_csv(os.path.dirname(os.path.abspath(__file__)) + '\stats.csv')

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

import keras
from keras.layers import Dense
from keras.models import Sequential
import matplotlib.pyplot as plt

trainPredictors = predictors[:3100, :]
trainTarget = target.loc[0:3099]

testPredictors = predictors[3100:, :]
testTarget = target.loc[3100:3431]
testOdds = odds.loc[3100:3431]

n_cols = trainPredictors.shape[1]


#na razie z tego najlepsza skutecznosc
model1 = Sequential()
model1.add(Dense(15, activation='relu', input_shape=(n_cols,)))
model1.add(Dense(3, activation='softmax'))
model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model1_training = model1.fit(trainPredictors, trainTarget, epochs=20, verbose=False)

#nawet ok
model2 = Sequential()
model2.add(Dense(15, activation='relu', input_shape=(n_cols,)))
model2.add(Dense(15, activation='relu'))
model2.add(Dense(15, activation='relu'))
model2.add(Dense(3, activation='softmax'))
model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model2_training = model2.fit(trainPredictors, trainTarget, epochs=20, verbose=False)

model3 = Sequential()
model3.add(Dense(150, activation='relu', input_shape=(n_cols,)))
model3.add(Dense(3, activation='softmax'))
model3.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model3_training = model3.fit(trainPredictors, trainTarget, epochs=20, verbose=False)

model4 = Sequential()
model4.add(Dense(150, activation='relu', input_shape=(n_cols,)))
model4.add(Dense(150, activation='relu'))
model4.add(Dense(150, activation='relu'))
model4.add(Dense(3, activation='softmax'))
model4.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model4_training = model4.fit(trainPredictors, trainTarget, epochs = 20, verbose=False)

model5 = Sequential()
model5.add(Dense(15, activation='sigmoid', input_shape=(n_cols,)))
model5.add(Dense(3, activation='softmax'))
model5.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model5_training = model5.fit(trainPredictors, trainTarget, epochs = 20, verbose=False)


#test_loss, test_acc = model1.evaluate(testPredictors, testTarget)
# print('Test loss', test_loss)
# print('Test accuracy:', test_acc)

plt.plot(model1_training.history['acc'], '-r', label = 'model1')
plt.plot(model2_training.history['acc'], '-b', label = 'model2')
plt.plot(model3_training.history['acc'], '-y', label = 'model3')
plt.plot(model4_training.history['acc'], '-g', label = 'model4')
plt.plot(model5_training.history['acc'], '-m', label = 'model5')
plt.legend(loc = 'upper left')
plt.title('Models accuracy')
plt.savefig(os.path.dirname(os.path.abspath(__file__)) + '\plot1.png')

loss1, acc1 = model1.evaluate(testPredictors, testTarget, verbose=False)
loss2, acc2 = model2.evaluate(testPredictors, testTarget, verbose=False)
loss3, acc3 = model3.evaluate(testPredictors, testTarget, verbose=False)
loss4, acc4 = model4.evaluate(testPredictors, testTarget, verbose=False)
loss5, acc5 = model5.evaluate(testPredictors, testTarget, verbose=False)

print("acc1 - " + str(acc1))
print("acc2 - " + str(acc2))
print("acc3 - " + str(acc3))
print("acc4 - " + str(acc4))
print("acc5 - " + str(acc5))
