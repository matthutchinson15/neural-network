import numpy as np 
import nnfs
from nnfs.datasets import vertical_data
import matplotlib.pyplot as plt


nnfs.init()

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        # Derivative of loss wrt weights and biases
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # Derivative of loss wrt inputs
        self.dinputs = np.dot(dvalues, self.weights.T)

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        # Derivative of loss wrt inputs
        self.dinputs = dvalues.copy()
        self.dinputs[self.output <= 0] = 0

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]

        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backward(self, output, y_true):
        samples = len(output)
        gradients = output.copy()
        if len(y_true.shape) == 1:
            gradients[range(samples), y_true] -= 1
        elif len(y_true.shape) == 2:
            gradients -= y_true

        gradients = gradients / samples
        return gradients

X, y = vertical_data(samples=100, classes=3)

#plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap='brg')
#plt.show()

dense1 = Layer_Dense(2,3)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

loss_function = Loss_CategoricalCrossentropy()

learning_rate = 0.05
lowest_loss = float('inf')
best_dense1_weights = dense1.weights.copy()
best_dense1_biases = dense1.biases.copy()
best_dense2_weights = dense2.weights.copy()
best_dense2_biases = dense2.biases.copy()

for iteration in range(100000):

    # Forward pass
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    # Calculate loss
    loss = loss_function.calculate(activation2.output, y)

    # Backward pass
    loss_gradient = loss_function.backward(activation2.output, y)
    dense2.backward(loss_gradient)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    # Update weights and biases
    dense1.weights -= learning_rate * dense1.dweights
    dense1.biases -= learning_rate * dense1.dbiases
    dense2.weights -= learning_rate * dense2.dweights
    dense2.biases -= learning_rate * dense2.dbiases
    
    # Track best weights and bias
    predictions = np.argmax(activation2.output, axis=1)
    accuracy = np.mean(predictions == y)

    if loss < lowest_loss:
        print(f'New set of weights found, iteration: {iteration}, loss: {loss:.4f}, accuracy: {accuracy:.4f}')
        best_dense1_weights = dense1.weights.copy()
        best_dense1_biases = dense1.biases.copy()
        best_dense2_weights = dense2.weights.copy()
        best_dense2_biases = dense2.biases.copy()
        lowest_loss = loss

#print(activation2.output[:5])
print(f"Final loss: {loss:.4f}")