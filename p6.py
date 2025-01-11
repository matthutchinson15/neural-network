# softmax actiation
#input - exponentiate - normalize - output

import math
import numpy as np
layer_outputs = [4.8, 1.21, 2.385]
import nnfs
nnfs.init()
E = 2.71828182846
#euler's number - muy importante
#E = math.e
exp_values = np.exp(layer_outputs)

print(exp_values)
#normalize value
#  a single output neurons value divided by sum ofall ouput neurons in the output layer
# this gives probability distribution
#get rid of negative values but not lose meaning of negative value
#convert negative to positive without losing meaning of negative value
#code normalization
norm_values = exp_values / np.sum(exp_values)

print(norm_values)
print(sum(norm_values))
#top row exponentiated
#2nd row normalized exponentiated