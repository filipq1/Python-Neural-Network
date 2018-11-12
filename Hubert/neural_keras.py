import pandas as pd
import numpy as np
import os

data = pd.read_csv(os.path.dirname(os.path.abspath(__file__)) + '\inputdata.csv')

predictors = data.iloc[:,:20]
target = pd.DataFrame(0, index = range(data.shape[0]), columns = range(3))
target.columns = ['WIN', 'DRAW', 'LOSE']

for i in range(380):
    temp = data.iloc[i,20]
    if(temp==0):
        target.iloc[i, 1] = 1
    elif(temp == 1):
        target.iloc[i, 0] = 1
    else: target.iloc[i, 2] = 1

import keras
from keras.layers import Dense
from keras.models import Sequential
import matplotlib.pyplot as plt


n_cols = predictors.shape[1]

model1 = Sequential()
model1.add(Dense(40, activation='relu', input_shape=(n_cols,)))
model1.add(Dense(3, activation='softmax'))
model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model1_training = model1.fit(predictors, target, epochs=10, verbose=False)

model2 = Sequential()
model2.add(Dense(40, activation='relu', input_shape=(n_cols,)))
model2.add(Dense(40, activation='relu'))
model2.add(Dense(40, activation='relu'))
model2.add(Dense(40, activation='relu'))
model2.add(Dense(3, activation='softmax'))
model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model2_training = model2.fit(predictors, target, epochs=10, verbose=False)

model3 = Sequential()
model3.add(Dense(200, activation='relu', input_shape=(n_cols,)))
model3.add(Dense(150, activation='relu'))
model3.add(Dense(3, activation='softmax'))
model3.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model3_training = model3.fit(predictors, target, epochs=10, verbose=False)

model4 = Sequential()
model4.add(Dense(300, activation='relu', input_shape=(n_cols,)))
model4.add(Dense(300, activation='relu'))
model4.add(Dense(300, activation='relu'))
model4.add(Dense(300, activation='relu'))
model4.add(Dense(300, activation='relu'))
model4.add(Dense(3, activation='softmax'))
model4.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model4_training = model4.fit(predictors, target, epochs=10, verbose=False)


plt.plot(model1_training.history['acc'], 'r', model2_training.history['acc'], 'b', model3_training.history['acc'], 'y', model4_training.history['acc'], 'g')
plt.show()
