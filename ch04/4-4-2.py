import numpy as np


def func2(x):
    return x[0] **2 + x[1] ** 2

# init_x = np.array([-3.0, 4.0])
# print(gradient_descent(func2, init_x, 10.0))

from func import softmax, cross_entropy_error, numerical_gradient

class SimpleNet():
    def __init__(self):
        self.W = np.random.randn(2, 3)
    
    def predict(self, x):
        return np.dot(x, self.W)
    
    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)
        
        return loss

net = SimpleNet()

x = np.array([.6, .9])
p = net.predict(x)

t = np.array([0, 0, 1])

def f(W):
    return net.loss(x, t)

dW = numerical_gradient(f, net.W)
print dW