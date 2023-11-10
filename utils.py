import numpy as np
import matplotlib.pyplot as plt
from hurst import compute_Hc
from scipy.signal import lfilter
from scipy.interpolate import UnivariateSpline
from scipy.special import gamma as gamma_function

def compute_RS(series, lag):
    """
    Compute Rescaled Range (R/S) for a given time series and lag.
    
    Parameters:
    - series (numpy array): The time series data.
    - lag (int): The lag for which R/S needs to be computed.
    
    Returns:
    - RS (float): Rescaled Range for the given lag.
    """
    
    # Split the time series into non-overlapping sub-series of length 'lag'
    N = len(series)
    n_splits = N // lag
    split_series = np.array_split(series, n_splits)
    
    RS_list = []
    for sub_series in split_series:
        mean_val = np.mean(sub_series)
        diff_series = sub_series - mean_val
        cum_series = np.cumsum(diff_series)
        
        R = np.max(cum_series) - np.min(cum_series)
        S = np.std(sub_series)
        
        RS = R / S if S != 0 else 0
        RS_list.append(RS)
    
    return np.mean(RS_list)

def calculate_hurst_exponent(time_series, max_lag=20):
    """
    Calculate the Hurst exponent for a given time series.
    
    Parameters:
    - time_series (numpy array): The time series data.
    - max_lag (int, optional): Maximum lag to consider for R/S computation. Default is 20.
    
    Returns:
    - H (float): Estimated Hurst exponent.
    """
    
    lags = range(2, max_lag + 1)
    RS_values = [compute_RS(time_series, lag) for lag in lags]
    
    H, _ = np.polyfit(np.log(lags), np.log(RS_values), 1)
    return H

def spline_interpolation(data_points, smoothness=0):
    """
    Perform spline interpolation on the given list of numbers.
    
    Parameters:
    - data_points (list or numpy array): The list of numbers representing the state estimate.
    - smoothness (float, optional): Amount of smoothing. If 0, spline will pass through all data points.

    Returns:
    - s (UnivariateSpline object): Spline object which can be used to evaluate interpolated values.
    """

    # Assuming data_points are evenly spaced. 
    # If time or other x-values are available, replace this with the appropriate x-values.
    x = np.arange(len(data_points))
    y = np.array(data_points)

    # Create a spline object
    s = UnivariateSpline(x, y, s=smoothness)

    return s


def get_hurst_exp(arr, plotting, print_h = True):
    
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

    if print_h:
        print("H={:.4f}, c={:.4f}".format(H,c))

    return H


def fractional_derivative(data, q, N):
    n = len(data)
    derivative = np.zeros(n)
    
    for t in range(n):
        summation = 0
        for j in range(N):
            if t-j < 0:  # Handle the case where t-j is out of bounds
                continue
            coeff = (-1)**j * gamma_function(j-q) / (gamma_function(j+1) * gamma_function(-q))
            summation += coeff * data[t-j]
        derivative[t] = summation * (n/N)**(-q)
    return derivative


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


def modify_state_to_match_hurst(estimated_state, true_state, estimated_hurst, hurst_true,  tol=0.05, max_iter=100, series_name = ""):
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
        modified_hurst = get_hurst_exp(modified_state, plotting = False, series_name = series_name)
        d -= 0.1 * (modified_hurst - hurst_true)  # Adjust dd

    return modified_state

        

