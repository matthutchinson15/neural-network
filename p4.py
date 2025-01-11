import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

#data set
# Modified CS231 course neural networks
#how many feature set is points, classes is classes
#see below 3 classes with 100 feature sets

def create_data(points, classes):
    x = np.zeros((points * classes, 2))
    y = np.zeros(points * classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points * class_number, points * (class_number + 1))
        r = np.linspace(0.0, 1, points)  # radius
        t = np.linspace(class_number * 4, (class_number + 1) * 4, points) + np.random.randn(points) * 0.2
        x[ix] = np.c_[r * np.sin(t * 2.5), r * np.cos(t * 2.5)]
        y[ix] = class_number
    return x, y

# Generate the data using the create_data function
x, y = create_data(100, 3)

# Visualize the data
plt.scatter(x[:, 0], x[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.show()

plt.scatter(x[:,0], x[:,1], c=y, cmap= "brg")
plt.show()
