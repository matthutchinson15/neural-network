#softmax_output = [[0.7, 0.1, 0.2],
#                  [0.1, 0.5, 0.4],
#                 [0.02, 0.9, 0.08]]
#class_targets = [0, 1, 1]
#class: 0 dog 1 cat 2 human

# class_targets = [dog, cat, cat]
#class_targets = [0, 1, 1]

# 0.7, 0.5, .9

#print(softmax_outputs[range(len(softmax_outputs)), class_targets])

import numpy as np


softmax_outputs = np.array([[0.7, 0.1, 0.2],
                  [0.1, 0.5, 0.4],
                  [0.02, 0.9, 0.08]])
class_targets = [0, 1, 1]


print(softmax_outputs[[0, 1, 2], class_targets])
