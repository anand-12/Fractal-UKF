import numpy as np
import matplotlib.pyplot as plt
from hurst import compute_Hc
from scipy.signal import lfilter

def get_hurst_exp(arr, plotting):
    
    arr_np = np.array(arr)
    H, c, data = compute_Hc(arr_np, kind='price', simplified=True)

    # Plot
    if plotting:
        f, ax = plt.subplots()
        ax.plot(data[0], c*data[0]**H, color="deepskyblue")
        ax.scatter(data[0], data[1], color="purple")
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Time interval')
        ax.set_ylabel('R/S ratio')
        ax.grid(True)
        plt.show()

    print("H={:.4f}, c={:.4f}".format(H,c))

    return H




def fractional_difference(series, d):
    """
    Compute fractional difference of a time series.
    
    Parameters:
    series (array-like): Input time series.
    d (float): Fractional differencing parameter (0 <= d <= 1).
    
    Returns:
    array-like: Fractionally differenced time series.
    """
    n = len(series)
    weights = [1.0]
    for k in range(1, n):
        weights.append(abs(-weights[-1] * (d - k + 2) / k))
    return lfilter(weights, 1.0, series)


def modify_state_to_match_hurst(estimated_state, true_state, estimated_hurst, hurst_true,  tol=0.05, max_iter=100):
    """
    Modify the estimated state to match the desired Hurst exponent of the true state.

    Parameters:
    estimated_state (array-like): Estimated state.
    true_state (array-like): True state.
    hurst_true (float): Desired Hurst exponent for the true state.
    tol (float): Tolerance for convergence.
    max_iter (int): Maximum number of iterations.

    Returns:
    array-like: Modified estimated state with a similar Hurst exponent to true_state.
    """
    # print(type(estimated_state), type(true_state), estimated_hurst, hurst_true)
    d = 0.5 


    modified_hurst = estimated_hurst
    for _ in range(max_iter):

        if abs(modified_hurst - hurst_true) < tol:
            break
        modified_state = fractional_difference(estimated_state, d)/3
        modified_hurst = get_hurst_exp(modified_state, plotting = False)
        d -= 0.1 * (modified_hurst - hurst_true)  # Adjust dd

    return modified_state

        

