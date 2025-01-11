import sys
import numpy as np
import matplotlib

#matrix is (3,4)
inputs = [[1, 2, 3, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]]
#matrix is (3,4) needs to be (4,3) index 2 of first element and index 1 of 2nd must match


weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]

weights2= [[0.1, -0.14, 0.5],
           [-0.5, 0.12, -0.33],
           [-0.44, 0.73, -0.13]]

biases2 = [-1, 2, -0.5]

layer1_outputs = np.dot(inputs, np.array(weights).T) + biases
layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2

print(layer2_outputs)
# .dot merges a matrix of vectors with vector
#shape error so use np . array converts rows into columns when mismatch in input or weight
# np.array is used and associated with transpose
#compute output to get new table generated from 1 and 2 matrix


