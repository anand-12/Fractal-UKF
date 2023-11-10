import numpy as np
import matplotlib.pyplot as plt
import argparse
from test_funcs.functions import mackey_glass, generate_fBm, logistic_map
from filters.filters import unscented_kalman_filter_mackey, unscented_kalman_filter_fbm, unscented_kalman_filter_logistic
from utils import get_hurst_exp, fractional_derivative
import json

np.random.seed(42)

def load_config():
    with open('params.json', 'r') as file:
        config = json.load(file)
    return config

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, help="Name of the time series")
args = parser.parse_args()
series_name = args.name

config = load_config()

if series_name == "mackey":
    t_min, t_max, beta, gamma, tao, n, true_state, q_min, q_max, N, scaling_factor = config["mackey"]["t_min"], config["mackey"]["t_max"], config["mackey"]["beta"], config["mackey"]["gamma"], config["mackey"]["tao"], config["mackey"]["n"], config["mackey"]["true_state"], config["mackey"]["q_min"], config["mackey"]["q_max"], config["mackey"]["N"], config["mackey"]["scaling_factor"]
elif series_name == "fbm":
    t_min, t_max, hurst_exponent, true_state, q_min, q_max, N, scaling_factor = config["fbm"]["t_min"], config["fbm"]["t_max"], config["fbm"]["hurst_exponent"], config["fbm"]["true_state"], config["fbm"]["q_min"], config["fbm"]["q_max"], config["fbm"]["N"], config["fbm"]["scaling_factor"]
elif series_name == "logmap":
    t_min, t_max, r, true_state, q_min, q_max, N, scaling_factor = config["logmap"]["t_min"], config["logmap"]["t_max"], config["logmap"]["r"], config["logmap"]["true_state"], config["logmap"]["q_min"], config["logmap"]["q_max"], config["logmap"]["N"], config["logmap"]["scaling_factor"]


measurement_noise_std, plotting_val = config["general"]["measurement_noise_std"], config["general"]["plotting_val"]
measurements = [true_state[0] + np.random.normal(0, measurement_noise_std)]

for t in range(t_min + 1, t_max + 1):

    if series_name == "mackey":
        true_state.append(mackey_glass(true_state, beta, gamma, tao, n))
    elif series_name == "fbm":
        true_state.append(generate_fBm(true_state, hurst_exponent))
    elif series_name == "logmap":
        true_state.append(logistic_map(true_state, r = r))
    measurements.append(true_state[-1] + np.random.normal(0, measurement_noise_std))

true_state = true_state[:t_max - t_min + 1]

initial_estimate = np.array([1.0]) 
initial_covariance = np.array([[0.1]]) 

if series_name == "mackey":
    estimated_state = unscented_kalman_filter_mackey(measurements, beta, gamma, tao, n, np.eye(1) * 0.001, np.eye(1) * 0.01, initial_estimate, initial_covariance)
elif series_name == "fbm":
    estimated_state = unscented_kalman_filter_fbm(measurements, hurst_exponent, np.eye(1) * 0.001, np.eye(1) * 0.01, initial_estimate, initial_covariance)
elif series_name == "logmap":
    estimated_state = unscented_kalman_filter_logistic(measurements, r, np.eye(1) * 0.001, np.eye(1) * 0.01, initial_estimate, initial_covariance)

true_state, estimated_state = [abs(ele1) for ele1 in true_state], [abs(ele2) for ele2 in estimated_state]
print(f"True State H")
H_true = get_hurst_exp(true_state, plotting = plotting_val, series_name=series_name)
print(f"Estimated State H")
H_estimated = get_hurst_exp(estimated_state, plotting = plotting_val, series_name=series_name)



optimal_q, min_H = q_min, H_estimated
for q in np.arange(q_min, q_max, 0.01):
    print(f'Current Q value {q}')
    adjusted_state = fractional_derivative(estimated_state, q, N)
    H_adjusted = get_hurst_exp(adjusted_state, plotting = plotting_val, series_name=series_name)
    if H_adjusted <= min_H:
        min_H = H_adjusted
        optimal_q = q

print(f'Optimal Q value is {optimal_q} and the corresponding Hurst exponent is {min_H}')

adjusted_state = fractional_derivative(estimated_state, optimal_q, N)
adjusted_state = [i * scaling_factor for i in adjusted_state]

time = range(t_min, t_max + 1)

error_estimated = np.array(true_state) - np.array(estimated_state)
error_adjusted = np.array(true_state) - np.array(adjusted_state)

print(f'Mean error between true state and UKF estimated state is {error_estimated.mean()}')
print(f'Mean error between true state and Fractal UKF estimated state is {error_adjusted.mean()}')

plt.figure(figsize=(12, 8))
plt.plot(time, true_state, label='True State', linestyle='--')
plt.plot(time, measurements, label='Noisy Measurements', marker='o', linestyle='None', markersize=4)
plt.plot(time, estimated_state, label='Estimated State', linestyle='-')
plt.plot(time, adjusted_state, label='Modified Estimated State', linestyle='-')
plt.plot(time, error_estimated, label='Error for estimated', linestyle='-.', color='red')
plt.plot(time, error_adjusted, label='Error for modified', linestyle='-.', color='green')
plt.xlabel('Time')
plt.ylabel('State')
plt.legend()
plt.grid(True)
plt.title('Fractionally Differintegrated modified UKF')
plt.show()

