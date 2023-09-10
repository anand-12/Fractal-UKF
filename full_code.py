import numpy as np
import matplotlib.pyplot as plt
from hurst import compute_Hc
import fbm, hurst, pywt
from scipy.signal import lfilter
from fbm import FBM
from test_funcs.mackey_glass import mackey_glass
from test_funcs.fBm import generate_fBm
from filters.unscented_kalman_filter_mackey import unscented_kalman_filter_mackey
from filters.unscented_kalman_filter_fbm import unscented_kalman_filter_fbm
from utils import get_hurst_exp, modify_state_to_match_hurst


from test_funcs.mackey_glass import mackey_glass
from test_funcs.fBm import generate_fBm


# params for mackey
t_min = 18
t_max = 1100
beta = 0.2
gamma = 0.1
tao = 18
n = 10

hurst_exponent = 0.7
series_name = "fbm"
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

