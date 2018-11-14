import pandas as pd
import numpy as np
import os
import sklearn
from sklearn import preprocessing
from sklearn.utils import shuffle
import tensorflow as tf

#przygotowanie danych
data = pd.read_csv(os.path.dirname(os.path.abspath(__file__)) + '\stats.csv')

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

#modele

# #na razie z tego najlepsza skutecznosc
# model1 = Sequential()
# model1.add(Dense(16, activation='relu', input_shape=(n_cols,)))
# model1.add(Dense(8, activation='relu'))
# model1.add(Dense(3, activation='softmax'))
# model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model1_training = model1.fit(trainPredictors, trainTarget, epochs=20, verbose=False)

# #nawet ok
# model2 = Sequential()
# model2.add(Dense(15, activation='relu', input_shape=(n_cols,)))
# model2.add(Dense(3, activation='softmax'))
# model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model2_training = model2.fit(trainPredictors, trainTarget, epochs=20, verbose=False)

# model3 = Sequential()
# model3.add(Dense(200, activation='relu', input_shape=(n_cols,)))
# model3.add(Dense(150, activation='relu'))
# model3.add(Dense(3, activation='softmax'))
# model3.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model3_training = model3.fit(trainPredictors, trainTarget, epochs=20, verbose=False)

# model4 = Sequential()
# model4.add(Dense(300, activation='relu', input_shape=(n_cols,)))
# model4.add(Dense(300, activation='relu'))
# model4.add(Dense(300, activation='relu'))
# model4.add(Dense(300, activation='relu'))
# model4.add(Dense(300, activation='relu'))
# model4.add(Dense(3, activation='softmax'))
# model4.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model4_training = model4.fit(trainPredictors, trainTarget, epochs = 20, verbose=False)

# #test_loss, test_acc = model1.evaluate(testPredictors, testTarget)
# # print('Test loss', test_loss)
# # print('Test accuracy:', test_acc)

# plt.plot(model1_training.history['acc'], '-r', label = 'model1')
# plt.plot(model2_training.history['acc'], '-b', label = 'model2')
# plt.plot(model3_training.history['acc'], '-y', label = 'model3')
# plt.plot(model4_training.history['acc'], '-g', label = 'model4')
# plt.legend(loc = 'upper left')
# plt.title('Models accuracy')
# plt.show()


# #calculate betting return
#predictions = predictions = model1.predict(testPredictors)

def calculateBettingReturn(predictions, testPredictors, testTarget, testOdds):
    betting_difference = 0.05
    result = {
        'spend': 0,
        'income': 0,
    }
    for i in range(predictions.shape[0]):
        # if predictions[i, 0] - (1/testOdds.iloc[i]['odds-home']) > betting_difference and testOdds.iloc[i]['odds-home'] > 1.5:
        #     result['spend'] += 10
        #     if testTarget.iloc[i]['HOME'] == 1:
        #         result['income'] += 10 * testOdds.iloc[i]['odds-home']
        if predictions[i, 1] - (1/testOdds.iloc[i]['odds-draw']) > betting_difference:
            result['spend'] += 10
            if testTarget.iloc[i]['DRAW'] == 1:
                result['income'] += 10 * testOdds.iloc[i]['odds-draw']
        # if predictions[i, 2] - (1/testOdds.iloc[i]['odds-away']) > betting_difference and testOdds.iloc[i]['odds-away'] > 1.5:
        #     result['spend'] += 10
        #     if testTarget.iloc[i]['AWAY'] == 1:
        #         result['income'] += 10 * testOdds.iloc[i]['odds-away']
    result['rateOfReturn'] = result['income'] / result['spend']
    return result

numberOfRuns = 10
resultArray = np.array((numberOfRuns,3), float)
temp = []
#produce results array
for i in range(numberOfRuns):
    print(i)
    model = Sequential()
    model.add(Dense(16, activation='relu', input_shape=(n_cols,)))
    model.add(Dense(9, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model_training = model.fit(trainPredictors, trainTarget, epochs=50, verbose=False)
    predictions = predictions = model.predict(testPredictors)
    bettingReturn = calculateBettingReturn(predictions, testPredictors, testTarget, testOdds)
    rateOfReturn = bettingReturn['rateOfReturn']
    loss, acc = model.evaluate(testPredictors, testTarget)
    temp.append([loss, acc, rateOfReturn])

resultArray = np.asarray(temp)
plt.plot(resultArray[:,0], '-r', label = 'Loss')
plt.plot(resultArray[:,1], '-g', label = 'Accuracy')
plt.plot(resultArray[:,2], '-b', label = 'Return')
plt.legend(loc = 'upper left')
plt.title('Model performance')
plt.show()
print(np.mean(resultArray[:,2]))
print(resultArray)

