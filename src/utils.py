import numpy as np
import functions as fn

import numpy as np

def generate_sampled_x_values(x_min, x_max, amount_samples):
    if amount_samples < 2:
        raise ValueError("amount_samples moet minstens 2 zijn om een range te vormen.")
    jumps = np.random.uniform(0.01, 1.0, size=amount_samples - 1)
    reeks = np.zeros(amount_samples)
    reeks[1:] = np.cumsum(jumps)
    reeks_min = reeks.min()
    reeks_max = reeks.max()
    sampled_values = x_min + (reeks - reeks_min) * (x_max - x_min) / (reeks_max - reeks_min)
    np.random.shuffle(sampled_values)
    return sampled_values

def get_full_x_values(x_min, x_max, amount):
    return np.linspace(start=x_min, stop=x_max, num=amount)

def function_stringparser(function_string, x):
    if function_string == "sqrt":
        return fn.sqrt(x)
    elif function_string == "chaos":
        return fn.chaos(x)
    elif function_string == "parabola":
        return fn.parabola(x)
    elif function_string == "heartbeat":
        return fn.heartbeat(x)
    elif function_string == "sin":
        return fn.sin(x)
    else:
        raise ValueError("This function does not exist.")