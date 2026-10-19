import numpy as np

def sqrt(x):
    y = np.where(x < 0, -np.sqrt(-x), np.sqrt(x))
    return y

def chaos(x):
    modified_x = x
    with np.errstate(divide='ignore', invalid='ignore'):
        y = np.sin(1 / modified_x)
        y = np.where(modified_x == 0, 0, y)
    return y

def parabola(x):
    modified_x = x / 2
    y = modified_x ** 2
    return y

def heartbeat(x):
    modified_x = x * 2
    y = np.cos(modified_x) - np.cos(5 * modified_x)
    return y

def sin(x):
    modified_x = x * 8
    y = np.sin(modified_x)
    return y