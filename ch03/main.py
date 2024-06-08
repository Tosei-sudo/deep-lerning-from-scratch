import numpy as np

def step_function(x):
    y = x > 0
    return y.astype(np.int)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def f_3_11():
    A = np.array([
        [1, 2, 3],
        [4, 5, 6]
    ])
    
    B = np.array([
        [1, 2],
        [3, 4],
        [5, 6]
    ])
    
    print(np.dot(A, B))
    
def softmax(a):
    c = np.max(a)
    exp = np.exp(a - c)
    sum_exp = np.sum(exp)
    return exp / sum_exp
print softmax(np.array([.3,2.9,4.0]))