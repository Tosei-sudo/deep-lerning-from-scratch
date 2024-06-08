import numpy as np

def network(x1, x2):
    X = np.array([x1, x2])
    W = np.array([
        [1, 3, 5],
        [2, 4, 6]
    ])
    Y = np.dot(X, W)
    print(Y)

network(1, 1 )