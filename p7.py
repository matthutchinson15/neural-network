# softmax actiation
#input - exponentiate - normalize - output

#import math
import numpy as np
layer_outputs = [[4.8, 1.21, 2.385],
                 [8.9, -1.81, 0.2],
                 [1.41, 1.051, 0.026]]
import nnfs
nnfs.init()
#E = 2.71828182846
#euler's number - muy importante
#E = math.e
exp_values = np.exp(layer_outputs)

#print(np.sum(layer_outputs, axis=0)) columns on 2d matrix
#we want sum by rows
print(np.sum(layer_outputs, axis=1, keepdims=True))

norm_values = exp_values / np.sum(exp_values, axis=1, keepdims=True)
print(norm_values)
#try with or without keepdims
#normalize value
#  a single output neurons value divided by sum ofall ouput neurons in the output layer
# this gives probability distribution
#get rid of negative values but not lose meaning of negative value
#convert negative to positive without losing meaning of negative value
#code normalization
#norm_values = exp_values / np.sum(exp_values)

#print(norm_values)
#print(sum(norm_values))
#top row exponentiated
#2nd row normalized exponentiated


# combot overflow
# take all values in output, subtract largest value in layer from all values
# largest value = 0 and everything else less then 0
# range of possibilites is between 0 and 1
