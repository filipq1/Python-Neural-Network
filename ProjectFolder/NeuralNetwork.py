import pandas as pd
import numpy as np

data = pd.read_csv('inputdata.csv')
inputData = data.iloc[:, 0:21]
inputData = inputData.values
inputData = inputData[:340, :]
outputData = data.iloc[:,21].values

def map_result_to_output(x):
    outputData = np.empty([x.shape[0], 1])
    for i in range(x.shape[0]):
        outputData[i] = x[i]
    return outputData

# def map_result_to_output(x):
#     outputData = np.empty([x.shape[0], 3])
#     for i in range(x.shape[0]):
#         if(x[i] == 1):
#             newRow = [1, 0, 0]
#         elif(x[i] == 0):
#             newRow = [0, 1, 0]
#         elif(x[i] == 2):
#             newRow = [0, 0, 1]
#         outputData[i] = newRow
#     return outputData

outputData = map_result_to_output(data.iloc[:inputData.shape[0], 21])

def sigmoid(x):
    return 1.0/(1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1.0 - x)


# class NeuralNetwork:
#     def __init__(self, x, y):
#         self.input = x
#         self.weights1 = np.random.rand(self.input.shape[1], 21)
#         self.weights2 = np.random.rand(21, 12)
#         self.weights3 = np.random.rand(12, 7)
#         self.weights4 = np.random.rand(7, 1)
#         self.y = y
#         self.output = np.zeros(y.shape)

#     def feedforward(self):
#         self.layer1 = sigmoid(np.dot(self.input, self.weights1))
#         self.layer2 = sigmoid(np.dot(self.weights1, self.weights2))
#         self.layer3 = sigmoid(np.dot(self.weights2, self.weights3))
#         self.output = sigmoid(np.dot(self.weights3, self.weights4))

#     def backprop(self):
#         # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
#         d_weights4 = np.dot(self.layer3.T, (2*(self.y - self.output) * sigmoid_derivative(self.output))) 
#         d_weights3 = np.dot(self.layer2.T, (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights4.T) * sigmoid_derivative(self.layer3)))
#         d_weights2 = np.dot(self.layer1.T, (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights3.T) * sigmoid_derivative(self.layer2)))
#         d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))
#         # update the weights with the derivative (slope) of the loss function
#         self.weights1 += d_weights1
#         self.weights2 += d_weights2
#         self.weights3 += d_weights3
#         self.weights4 += d_weights4

class NeuralNetwork:
    def __init__(self, x, y):
        self.input      = x
        self.weights1   = np.random.rand(self.input.shape[1],12) 
        self.weights2   = np.random.rand(12,1)                 
        self.y          = y
        self.output     = np.zeros(y.shape)

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2

#############################################################
nn = NeuralNetwork(inputData, outputData)
print(inputData.shape)
print(outputData)
print(inputData.shape)
print(outputData.shape)

for i in range(10000):
    nn.feedforward()
    nn.backprop()

print(nn.weights1)
print(nn.output)
print(np.corrcoef(data.iloc[:,0], data.iloc[:,21]))