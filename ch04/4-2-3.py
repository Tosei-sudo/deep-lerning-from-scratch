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

x_train, t_train, x_test, t_test = dataset.load_dataset(one_hot_label=True)

def sum_squared_error(y, t):
    return .5 * np.sum((y - t) ** 2)

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size

train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]



