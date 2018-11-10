#%%
import numpy as np
import pandas as pd
import tensorflow as tf
import sklearn
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

data = pd.read_csv('inputdata.csv')
inputData = data.iloc[:, 0:21]
inputData = inputData.values
inputData = shuffle(inputData)
trainData = inputData[:360, :]
outputData = data.iloc[:,21].values
outputData = shuffle(outputData)
testData = inputData[360:, :]

def map_result_to_output(x):
    outputData = np.empty([x.shape[0], 1])
    for i in range(x.shape[0]):
        outputData[i] = x[i]
    return outputData

outputData = map_result_to_output(data.iloc[:trainData.shape[0], 21])
outputTestData = map_result_to_output(data.iloc[:testData.shape[0], 21])

model = keras.Sequential([
    keras.layers.Dense(15, activation=tf.nn.relu, input_shape=(trainData.shape[1],)),
    keras.layers.Dense(8, activation=tf.nn.relu),
    keras.layers.Dense(3, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(trainData, outputData, epochs=7)

test_loss, test_acc = model.evaluate(testData, outputTestData)

print('Test accuracy:', test_acc)

predictions = model.predict(testData)

for i in range(predictions.shape[0]):
    print((predictions[i]))
    print(outputTestData[i])
