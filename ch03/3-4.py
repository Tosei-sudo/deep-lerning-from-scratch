import numpy as np
from main import step_function, sigmoid, relu

def identity_function(x):
    return x

X = np.array([1.0, .5])
W1 = np.array([
    [.1, .3, .5],
    [.2, .4, .6]
])
B1 = np.array([.1, .2, .3])

A1 = np.dot(X, W1) + B1

Z1 = sigmoid(A1)

W2 = np.array([
    [.1, .4],
    [.2, .5],
    [.3, .6],
])
B2 = np.array([
    .1, .2
])

A2 = np.dot(Z1, W2) + B2
Z2 = sigmoid(A2)

W3 = np.array([
    [.1, .3],
    [.2, .4]
])
B3 = np.array([.1, .2])

A3 = np.dot(Z2, W3) + B3
Y = identity_function(A3)
print Y