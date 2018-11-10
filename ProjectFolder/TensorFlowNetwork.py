#%%
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

data = pd.read_csv('inputdata.csv')
inputData = data.iloc[:, 0:21]
inputData = inputData.values
trainData = inputData[:320, :]
outputData = data.iloc[:,21].values
testData = inputData[320:, :]

def map_result_to_output(x):
    outputData = np.empty([x.shape[0], 1])
    for i in range(x.shape[0]):
        outputData[i] = x[i]
    return outputData

outputData = map_result_to_output(data.iloc[:trainData.shape[0], 21])
outputTestData = map_result_to_output(data.iloc[:testData.shape[0], 21])

def build_model():
  model = keras.Sequential([
    keras.layers.Dense(64, activation=tf.nn.relu,
                       input_shape=(trainData.shape[1],)),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(1)
  ])

  optimizer = tf.train.RMSPropOptimizer(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae'])
  return model

model = build_model()
model.summary()

class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 500

# Store training stats
history = model.fit(trainData, outputData, epochs=EPOCHS,
                    validation_split=0.2, verbose=0,
                    callbacks=[PrintDot()])

def plot_history(history):
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error')
  plt.plot(history.epoch, np.array(history.history['mean_absolute_error']),
           label='Train Loss')
  plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']),
           label = 'Val loss')
  plt.legend()
  plt.ylim([0, 5])

plot_history(history)

model = build_model()

# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

history = model.fit(trainData, outputData, epochs=EPOCHS,
                    validation_split=0.2, verbose=0,
                    callbacks=[early_stop, PrintDot()])

plot_history(history)

[loss, mae] = model.evaluate(testData, outputTestData, verbose=0)
print("Testing set Mean Abs Error: {:7.2f}".format(mae))

test_predictions = model.predict(testData).flatten()
print(data.iloc[320:,21])
print(test_predictions)

# plt.scatter(outputTestData, test_predictions)
# plt.xlabel('True Values')
# plt.ylabel('Predictions')
# plt.axis('equal')
# plt.xlim(plt.xlim())
# plt.ylim(plt.ylim())
# _ = plt.plot([-100, 100], [-100, 100])

# error = test_predictions - outputTestData
# plt.hist(error, bins = 50)
# plt.xlabel("Prediction Error")
# _ = plt.ylabel("Count")