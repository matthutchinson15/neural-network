# model loss
import matplotlib.pyplot as plt
import numpy as np
#slope is change in y/change in x  or delta y/ delta x
#slope is 2

#def f(x):
#    return 2*x

#x= np.array(range(5))
#y = f(x)

#print(x)
#print(y)




#add small delta to calculate derivatives
#update model parameters is what this does
#def f(x):
#    return 2*x**2

#p2_delta =0.0001

#x1 = 1
#x2 = x1+p2_delta

#y1 = f(x1)
#y2 = f(x2)

#approximate_derivative = (y2-y1)/ (x2-x1)
#print(approximate_derivative)



#def f(x):
#    return 2*x**2

#x = np.arange (0, 50, 0.001)
#y = f(x)

#plt.plot(x, y)
#plt.show()

# need b in y= mx+ b already have slope
# b = y-mx

def f(x):
    return 2*x**2

x = np.arange (0, 50, 0.001)
y = f(x)
plt.plot(x, y)
p2_delta = 0.0001
x1 = 2
x2 = x1+p2_delta

y1 = f(x1)
y2 = f(x2)

print((x1, y1), (x2, y2))

approximate_derivative = (y2-y1)/ (x2-x1)
b = y2 - approximate_derivative*x2

def approximate_tangent_line(x):
    return approximate_derivative*x + b

to_plot = [x1-0.9, x1, x1+0.9]
plt.plot(to_plot, [approximate_tangent_line(point) for point in to_plot])

print('approximate derivative for f(x)',
      f'where x = {x1} is {approximate_derivative}')

plt.show()
      