import numpy as np

def numrical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    
    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)
        
        x[idx] = tmp_val - h
        fxh2 = f(x)
        
        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val
    
    return grad

def numrical_diff(f, x):
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2 * h)

def gradient_descent(f, init_x, lr = .01, step_sum=100):
    x = init_x
    
    for i in range(step_sum):
        grad = numrical_gradient(f, x)
        x -= lr * grad

    return x

def func2(x):
    return x[0] **2 + x[1] ** 2

init_x = np.array([-3.0, 4.0])
print(gradient_descent(func2, init_x, 10.0))
