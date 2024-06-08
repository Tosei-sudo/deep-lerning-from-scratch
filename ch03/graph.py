import numpy as np
import matplotlib.pylab as plt

from main import step_function, sigmoid, relu

x = np.arange(-10.0, 10.0, .1)
y = step_function(x)
plt.plot(x, y)

y = sigmoid(x)
plt.plot(x, y)

y = relu(x)
plt.plot(x, y)

plt.ylim(-.1, 1.1)
plt.show()