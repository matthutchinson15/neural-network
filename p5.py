
#hidden layer calc relu 


import sys
import numpy as np
import matplotlib
import nnfs
from nnfs.datasets import spiral_data


nnfs.init()


# Convert X to a numpy array instead of a list of lists
#batching
X = np.array([[1, 2, 3, 2.5],
              [2.0, 5.0, -1.0, 2.0],
              [-1.5, 2.7, 3.3, -0.8]])

X, y = spiral_data(100, 3)
#100 feature sets 3 classes


# Define the Dense layer class
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

# Define the ReLU activation function class
class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

# Instantiate the layers and activation
layer1 = Layer_Dense(2, 5)  # n_inputs = 4 to match the number of features in X
activation1 = Activation_ReLU()

# Perform the forward pass through the dense layer
layer1.forward(X)

print(layer1.output)
activation1.forward(layer1.output)
print(activation1.output)
