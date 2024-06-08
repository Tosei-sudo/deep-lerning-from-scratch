import numpy as np


def numrical_diff(f, x):
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2 * h)

def func1(x):
    return .01 * x ** 2 + .1 * x

import matplotlib.pylab as plt

x = np.arange(-10.0, 10.0, .1)
y = func1(x)
plt.plot(x, y)

y = numrical_diff(func1, x)
plt.plot(x, y)

plt.show()