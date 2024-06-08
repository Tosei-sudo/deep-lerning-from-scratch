
import numpy as np
import sys, os
import pickle
sys.path.append(r"C:\Users\tosei\work\deep-lerning-from-scratch")
from data import dataset
from main import step_function, sigmoid, relu, softmax

from matplotlib import pyplot as plt

def img_show(img):
    plt.imshow(img, cmap='gray')
    plt.show()

x_train, t_train, x_test, t_test = dataset.load_dataset()

print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)

def init_network():
    with open("ch03/sample_weight_py2.pkl", "rb") as f:
        network = pickle.load(f)
    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)
    
    return y

network = init_network()

batch_size = 100
accuracy_cnt = 0.0
for i in range(0, len(x_test), batch_size):
    x_batch = x_test[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)

    accuracy_cnt += np.sum(p == t_test[i:i+batch_size])

print("acc:" + str(accuracy_cnt / len(x_test)))