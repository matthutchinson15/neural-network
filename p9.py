# how right and how wrong 
#loss function
# calculate loss with categorical cross entropy
# metric for error
#one hot encoding
# classes label one-hot
#classes # of # in [] #one-hot if 
#classes 5, labels 0 one- hot is [1,0,0,0,0]
# classes 5 labels 1 one-hot [0,1,0,0,0]

import math
#categorical cross entropy
softmax_output = [0.7, 0.1, 0.2]
target_output = [1,0,0]

loss = -(math.log(softmax_output[0])*target_output[0] +
        math.log(softmax_output[1])*target_output[1] +
        math.log(softmax_output[2])*target_output[2])

print(loss)
loss = -math.log(softmax_output[0])
print(loss)

print( -math.log(0.7))

# .35 is low ish for calc loss
#when we do print on random input .5 its .69 which is much higher
#print( -math.log(0.5)) this is random number not related to any data just seeing if i put it in what happens, result was higher loss

#print loss
#the lower the better
