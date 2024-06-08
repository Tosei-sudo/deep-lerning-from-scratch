import numpy as np
import sys, os
import pickle
sys.path.append(r"C:\Users\tosei\work\deep-lerning-from-scratch")
from data import dataset
from func import step_function, sigmoid, relu, softmax

from matplotlib import pyplot as plt

def img_show(img):
    plt.imshow(img, cmap='gray')
    plt.show()

x_train, t_train, x_test, t_test = dataset.load_dataset()

def sum_squared_error(y, t):
    return .5 * np.sum((y - t) ** 2)

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size

t = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
y = np.array([.1, .05, .6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0, 0])

print(sum_squared_error(y, t))
print(cross_entropy_error(y, t))