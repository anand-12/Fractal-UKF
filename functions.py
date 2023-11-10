import numpy as np
from matplotlib import pyplot as plt



def generate_fBm(x, hurst_exponent):
    """
    Generate the next value in a fractional Brownian motion (fBm) time series.

    Args:
        x (list or array): History of previous values.
        hurst_exponent (float): Hurst exponent controlling the self-similarity.

    Returns:
        float: Next value in the fBm series.
    """
    if len(x) >= 1:
        
        increment = np.random.randn()
        next_value = x[-1] + (increment / (2.0 ** hurst_exponent))
        return next_value
    else:
        
        return x[0]  
    
def logistic_map(x, r):
    """
    Generate the next value in a Logistic Map time series.

    Args:
        x (list or array): History of previous values.
        r (float): Parameter controlling the behavior of the map.

    Returns:
        float: Next value in the Logistic Map series.
    """
    return r * x[-1] * (1 - x[-1])

def mackey_glass(x, beta, gamma, tao, n):
    if len(x) >= tao + 1:
        return x[-1] + (beta * x[-tao-1] / (1 + x[-tao-1]**n)) - (gamma * x[-1])
    else:
        return x[-1]  
    

def tent_map(x, mu=1.99):
    """
    Iterates the tent map equation for a given x.

    Parameters:
    - x: Current value in the time series.
    - mu: Scaling parameter, typically set to 2 for chaotic behavior.

    Returns:
    - Next value in the Tent Map series.
    """
    if x[-1] < 0.5:
        return mu * x[-1]
    else:
        return mu * (1 - x[-1])
