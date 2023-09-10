import numpy as np
import matplotlib.pyplot as plt
from hurst import compute_Hc
import fbm, hurst, pywt
from scipy.signal import lfilter
from fbm import FBM
# from test_funcs.mackey_glass import mackey_glass
# from test_funcs.fBm import generate_fBm
# from filters.unscented_kalman_filter import unscented_kalman_filter
# from utils import get_hurst_exp, modify_state_to_match_hurst


# from test_funcs.mackey_glass import mackey_glass
# from test_funcs.fBm import generate_fBm

def unscented_kalman_filter_mackey(y, beta, gamma, tao, n, Q, R, x_init, P_init):
    # Number of state variables
    n_states = len(x_init)

    # Initialize state estimate and covariance
    x_hat = x_init
    P = P_init

    # UKF parameters
    alpha = 0.001  # Tuning parameter
    kappa = 0.1   # Tuning parameter

    # Weights for sigma points
    lambda_ = alpha**2 * (n_states + kappa) - n_states

    # UKF weights
    W_m = np.zeros(2 * n_states + 1)
    W_c = np.zeros(2 * n_states + 1)

    W_m[0] = lambda_ / (n_states + lambda_)
    W_c[0] = lambda_ / (n_states + lambda_) + (1 - alpha**2 + beta)

    for i in range(1, 2 * n_states + 1):
        W_m[i] = 1 / (2 * (n_states + lambda_))
        W_c[i] = W_m[i]

    estimated_state = []

    for measurement in y:
        # Sigma points generation
        sigma_points = np.zeros((2 * n_states + 1, n_states))
        sigma_points[0] = x_hat

        for i in range(n_states):
            sqrt_term = np.sqrt((n_states + lambda_) * P[i, i])
            sigma_points[i + 1] = x_hat + sqrt_term
            sigma_points[n_states + i + 1] = x_hat - sqrt_term

        # Predict step
        x_pred = np.zeros(n_states)
        for i in range(2 * n_states + 1):
            x_pred += W_m[i] * mackey_glass(sigma_points[i], beta, gamma, tao, n)


        P_pred = np.zeros((n_states, n_states))
        for i in range(2 * n_states + 1):
            err = mackey_glass(sigma_points[i], beta, gamma, tao, n) - x_pred

            P_pred += W_c[i] * np.outer(err, err)

        P_pred += Q

        # Update step
        K = P_pred @ np.linalg.inv(P_pred + R)
        # print("x_pred type and shape:", type(x_pred), x_pred.shape)
        # print("K type and shape:", type(K), K.shape)
        # print("measurement type", type(measurement))
        x_hat = x_pred + K @ (measurement - x_pred)
        P = P_pred - K @ P_pred

        # Store or use x_hat as your estimated state at this time step
        estimated_state.append(x_hat[0])

    return estimated_state


def unscented_kalman_filter_fbm(y, hurst_exponent, Q, R, x_init, P_init):
    # Number of state variables
    n_states = len(x_init)

    # Initialize state estimate and covariance
    x_hat = x_init
    P = P_init

    # UKF parameters
    alpha = 0.001  # Tuning parameter
    kappa = 0.1   # Tuning parameter

    # Weights for sigma points
    lambda_ = alpha**2 * (n_states + kappa) - n_states

    # UKF weights
    W_m = np.zeros(2 * n_states + 1)
    W_c = np.zeros(2 * n_states + 1)

    W_m[0] = lambda_ / (n_states + lambda_)
    W_c[0] = lambda_ / (n_states + lambda_) + (1 - alpha**2 + beta)

    for i in range(1, 2 * n_states + 1):
        W_m[i] = 1 / (2 * (n_states + lambda_))
        W_c[i] = W_m[i]

    estimated_state = []

    for measurement in y:

        
        # Sigma points generation
        sigma_points = np.zeros((2 * n_states + 1, n_states))
        sigma_points[0] = x_hat

        for i in range(n_states):
            sqrt_term = np.sqrt((n_states + lambda_) * P[i, i])
            sigma_points[i + 1] = x_hat + sqrt_term
            sigma_points[n_states + i + 1] = x_hat - sqrt_term

        # Predict step
        x_pred = np.zeros(n_states)
        for i in range(2 * n_states + 1):

            x_pred += W_m[i] * generate_fBm(sigma_points[i],hurst_exponent=0.7)

        P_pred = np.zeros((n_states, n_states))
        for i in range(2 * n_states + 1):
            err = generate_fBm(sigma_points[i], hurst_exponent=0.7) - x_pred
            P_pred += W_c[i] * np.outer(err, err)

        P_pred += Q

        # Update step
        K = P_pred @ np.linalg.inv(P_pred + R)
        # print("x_pred type and shape:", type(x_pred), x_pred.shape)
        # print("K type and shape:", type(K), K.shape)
        # print("measurement type", type(measurement))
        x_hat = x_pred + K @ (measurement - x_pred)
        P = P_pred - K @ P_pred

        # Store or use x_hat as your estimated state at this time step
        estimated_state.append(x_hat[0])

    return estimated_state

def mackey_glass(x, beta, gamma, tao, n):

    if len(x) >= tao + 1:

        return x[-1] + (beta * x[-tao-1] / (1 + x[-tao-1]**n)) - (gamma * x[-1])
    else:

        return x[-1]  # Return the last known value when there's not enough history

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
        d -= 0.1 * (modified_hurst - hurst_true)  # Adjust d to approach hurst_true

    return modified_state

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
        # Calculate the next value using fBm model
        increment = np.random.randn()
        next_value = x[-1] + (increment / (2.0 ** hurst_exponent))
        return next_value
    else:
        # If there's not enough history, return the initial value
        return x[0]  # Return the initial value


# Mackey equations parameters
t_min = 18
t_max = 1100
beta = 0.2
gamma = 0.1
tao = 18
n = 10

hurst_exponent = 0.7
series_name = "mackey"
np.random.seed(42)
true_state = [1.2]
measurement_noise_std = 0.1
measurements = [true_state[0] + np.random.normal(0, measurement_noise_std)]

for t in range(t_min + 1, t_max + 1):
    # if t <= t_max:  
    
    if series_name == "mackey":
        true_state.append(mackey_glass(true_state, beta, gamma, tao, n))
    elif series_name == "fbm":
        true_state.append(generate_fBm(true_state, hurst_exponent))
    measurements.append(true_state[-1] + np.random.normal(0, measurement_noise_std))

# print(true_state, type(true_state))
true_state = true_state[:t_max - t_min + 1]
# print(true_state, measurements, len(true_state), len(measurements))
initial_estimate = np.array([1.0]) 
initial_covariance = np.array([[0.1]]) 


# Start plain UKF
# print(measurements, type(measurements))
if series_name == "mackey":
    estimated_state = unscented_kalman_filter_mackey(measurements, beta, gamma, tao, n, np.eye(1) * 0.001, np.eye(1) * 0.01, initial_estimate, initial_covariance)
elif series_name == "fbm":
    estimated_state = unscented_kalman_filter_fbm(measurements, hurst_exponent, np.eye(1) * 0.001, np.eye(1) * 0.01, initial_estimate, initial_covariance)

# print(len(estimated_state), len(true_state))

true_state, estimated_state = [abs(ele1) for ele1 in true_state], [abs(ele2) for ele2 in estimated_state]
print("True")
H_true = get_hurst_exp(true_state, plotting = True)
print("Estimated")
H_estimated = get_hurst_exp(estimated_state, plotting = True)



# hurst_difference = H_true - H_estimated
'''
Detrended Fluctuation Analysis

TBI
'''


modified_estimated_state = modify_state_to_match_hurst(estimated_state, true_state, H_estimated, H_true)

get_hurst_exp(modified_estimated_state, plotting=False)


time = range(t_min, t_max + 1)

error_estimated = np.array(true_state) - np.array(estimated_state)
error_modified = np.array(true_state) - np.array(modified_estimated_state)

print(error_estimated.mean())
print(error_modified.mean())

plt.figure(figsize=(12, 8))
plt.plot(time, true_state, label='True State', linestyle='--')
plt.plot(time, measurements, label='Noisy Measurements', marker='o', linestyle='None', markersize=4)
plt.plot(time, estimated_state, label='Estimated State', linestyle='-')
plt.plot(time, modified_estimated_state, label='Hurst corrected Estimated State', linestyle='-')
plt.plot(time, error_estimated, label='Error for estimated', linestyle='-.', color='red')
plt.plot(time, error_modified, label='Error for modified', linestyle='-.', color='green')
plt.xlabel('Time')
plt.ylabel('State')
plt.legend()
plt.grid(True)
plt.title('Fractional differenced Mackey-Glass State Estimation with UKF')
plt.show()

