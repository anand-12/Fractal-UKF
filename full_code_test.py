import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.power import TTestIndPower 
from scipy.stats import wilcoxon
import pandas as pd
import argparse, math, time
from functions import mackey_glass, generate_fBm, logistic_map, tent_map
from filters import (
    unscented_kalman_filter_mackey,
    unscented_kalman_filter_fbm,
    unscented_kalman_filter_logistic,
    unscented_kalman_filter_tent_map
)
from utils import get_hurst_exp, fractional_derivative
import json
from common_plot2 import plot_trace


# np.random.seed(42)


def load_config(filename='params.json'):
    with open(filename, 'r') as file:
        return json.load(file)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, help="Name of the time series: mackey, logmap, tent_map")
    return parser.parse_args().name


def get_config_params(series_name, config):
    params = config[series_name]
    return {key: params[key] for key in params}


def generate_measurement_series(series_name, params, t_min, t_max, measurement_noise_std):
    # true_state = params['true_state']
    true_state = [np.random.uniform(0.1,1)]
    print(f'Initial state is {true_state}')
    measurements = [true_state[0] + np.random.normal(0, measurement_noise_std)]

    for t in range(t_min + 1, t_max + 1):
        if series_name == "mackey":
            true_state.append(mackey_glass(true_state, params['beta'], params['gamma'], params['tao'], params['n']))
        # elif series_name == "fbm":
        #     true_state.append(generate_fBm(true_state, params['hurst_exponent']))
        elif series_name == "logmap":
            true_state.append(logistic_map(true_state, r=params['r']))
        elif series_name == "tent_map":
            true_state.append(tent_map(true_state, mu=params['mu']))
        measurements.append(true_state[-1] + np.random.normal(0, measurement_noise_std))

    return true_state[:t_max - t_min + 1], measurements


def apply_ukf(series_name, measurements, params):
    initial_estimate = np.array([1.0])
    initial_covariance = np.array([[0.1]])

    if series_name == "mackey":
        return unscented_kalman_filter_mackey(
            measurements, params['beta'], params['gamma'], params['tao'], params['n'],
            np.eye(1) * 0.001, np.eye(1) * 0.01, initial_estimate, initial_covariance
        )
    # elif series_name == "fbm":
    #     return unscented_kalman_filter_fbm(
    #         measurements, params['hurst_exponent'], np.eye(1) * 0.001,
    #         np.eye(1) * 0.01, initial_estimate, initial_covariance
    #     )
    elif series_name == "logmap":
        return unscented_kalman_filter_logistic(
            measurements, params['r'], np.eye(1) * 0.001,
            np.eye(1) * 0.01, initial_estimate, initial_covariance
        )
    elif series_name == "tent_map":
        return unscented_kalman_filter_tent_map(
            measurements, params['mu'], np.eye(1) * 0.001,
            np.eye(1) * 0.01, initial_estimate, initial_covariance
        )


def adjust_state_with_optimal_q(true_state, estimated_state, q_min, q_max, N, plotting_val, H_true):
    print(f'UKF estimation Hurst exponent')
    UKF_H = get_hurst_exp(estimated_state, plotting=plotting_val)
    optimal_q, optimal_H_difference, optimal_H = q_min, abs(UKF_H - H_true), UKF_H
    print(f'Starting fractional modification...')
    H_trace = []
    for q in np.arange(q_min, q_max, 0.01):
        
        # adjusted_state = fractional_derivative(estimated_state, q, N)
        adjusted_state = fractional_derivative(true_state, q, N)
        # print("Adjusted stage")
        H_adjusted = get_hurst_exp(adjusted_state, plotting=plotting_val, print_h=False)
        # print(f'Q value {q} and H adjusted is {H_adjusted}')
        H_trace.append(H_adjusted)
        current_difference = abs(H_adjusted - H_true)
        if current_difference < optimal_H_difference:
            optimal_H_difference = current_difference
            optimal_H = H_adjusted
            optimal_q = q

    print(f'Optimal q value is {optimal_q} and the corresponding Hurst exponent is {optimal_H}')
    return optimal_q, H_trace, optimal_H, UKF_H


def plot_results(t_min, t_max, true_state, measurements, estimated_state, adjusted_state):
    time = range(t_min, t_max + 1)
    power_signal = np.mean(np.square(true_state))
    power_noise = np.mean(np.square(measurements))

    snr_db = 10 * math.log10(power_signal / power_noise)

    print(f"SNR (dB): {snr_db}")

    error_estimated = (np.array(true_state) - np.array(estimated_state))**2
    error_adjusted = (np.array(true_state) - np.array(adjusted_state))**2 # FOR MSE
    # error_estimated = (np.array(true_state) - np.array(estimated_state)) # FOR SE
    # error_adjusted = (np.array(true_state) - np.array(adjusted_state))
    var_estimated = np.var(error_estimated, ddof=1)
    var_adjusted = np.var(error_adjusted, ddof=1)

    print(f'Mean squared error between true state and UKF estimated state is {error_estimated.mean()}')
    print(f'Mean squared error between true state and Fractal UKF estimated state is {error_adjusted.mean()}')
    # print(f'Normalized Mean squared error between true state and UKF estimated state is {nmse_estimated}')
    # print(f'Normalized Mean squared error between true state and Fractal UKF estimated state is {nmse_adjusted}')
    print(f'Estimation variance between true state and UKF estimated state is {var_estimated}')
    print(f'Estimation variance between true state and Fractal UKF estimated state is {var_adjusted}')
    ukf_errors.append(error_estimated.mean())
    frac_ukf_errors.append(error_adjusted.mean())
    ukf_variances.append(var_estimated)
    frac_ukf_variances.append(var_adjusted)

    #Uncomment below lines for the plots
    # plt.figure(figsize=(12, 8))
    # plt.plot(time, true_state, label='True State', linestyle='--')
    # plt.plot(time, measurements, label='Noisy Measurements', marker='o', linestyle='None', markersize=4)
    # plt.plot(time, estimated_state, label='Estimated State', linestyle='-')
    # plt.plot(time, adjusted_state, label='Modified Estimated State', linestyle='-')
    # plt.plot(time, error_estimated, label='Error for estimated', linestyle='-.', color='red')
    # plt.plot(time, error_adjusted, label='Error for modified', linestyle='-.', color='green')
    # plt.xlabel('Time')
    # plt.ylabel('State')
    # plt.legend()
    # plt.grid(True)
    # plt.title('Fractionally Differintegrated modified UKF')
    # plt.show()

ukf_errors, frac_ukf_errors, ukf_variances, frac_ukf_variances = [], [], [], []
def main():


    global_H_trace = []
    frac_q_list, frac_H_list, UKF_H_list, true_H_list = [], [], [], []
    frac_true_diff, ukf_true_diff = [], []
    num_runs = 50
    series_name = parse_args()

    config = load_config()
    general_config = config["general"]
    params = get_config_params(series_name, config)
    for i in range(1,num_runs+1):
        print(f'Realization number: {i}')
        s = time.time()
        series_name = parse_args()

        config = load_config()
        general_config = config["general"]
        params = get_config_params(series_name, config)
        true_state, measurements = generate_measurement_series(
            series_name, params, params['t_min'], params['t_max'], general_config["measurement_noise_std"]
        )
        estimated_state = apply_ukf(series_name, measurements, params)
        true_state, estimated_state = [abs(ele1) for ele1 in true_state], [abs(ele2) for ele2 in estimated_state]
        print(f'True Hurst exponent')
        H_true = get_hurst_exp(true_state, plotting = config["general"]["plotting_val"])
        
        optimal_q, H_trace, optimal_H, UKF_H = adjust_state_with_optimal_q(true_state, estimated_state, params['q_min'], params['q_max'], params['N'], general_config["plotting_val"], H_true)

        adjusted_state = [i for i in fractional_derivative(estimated_state, optimal_q, params['N'])]
        # adjusted_state = [i for i in fractional_derivative(true_state, optimal_q, params['N'])]
        adjusted_state = [i*(np.mean(estimated_state)/np.mean(adjusted_state)) for i in adjusted_state]
        # adjusted_state = [i*(np.mean(true_state)/np.mean(adjusted_state)) for i in adjusted_state]
        plot_results(params['t_min'], params['t_max'], true_state, measurements, estimated_state, adjusted_state)
        global_H_trace.append(H_trace)
        frac_q_list.append(optimal_q)
        frac_H_list.append(optimal_H)
        UKF_H_list.append(UKF_H)
        true_H_list.append(H_true)
        
        print(f'Time taken: {time.time() - s} seconds')
    frac_true_diff, ukf_true_diff = [abs(i-j) for i, j in zip(true_H_list, frac_H_list)], [abs(i-j) for i, j in zip(true_H_list, UKF_H_list)]
    # Uncomment line below if you want to see the plot for Hurst vs q for 5 random realizations of any function
    
    # plot_trace(global_H_trace, frac_q_list, frac_H_list, true_H_list, params['q_min'], params['q_max'])


    # d = {}
    # d["logmap_global_H_trace"] = global_H_trace
    # d["logmap_frac_q_list"] = frac_q_list
    # d["logmap_frac_H_list"] = frac_H_list
    # d["logmap_true_H_list"] = true_H_list
    # d["logmap_q_min"] = params["q_min"]
    # d["logmap_q_max"] = params["q_max"]
    
    # with open('plot_values.json', 'a') as f:
    #     json.dump(d, f)


    print(f'In {num_runs} iterations, average UKF MSE is {np.mean(ukf_errors)} and average Fractional UKF MSE is {np.mean(frac_ukf_errors)}')
    print(f'The average difference between true hurst and UKF hurst is {np.mean(ukf_true_diff)} and average difference between true hurst and Fractional UKF hurst is {np.mean(frac_true_diff)}')
    # df = pd.DataFrame([UKF_H_list, frac_H_list])
    res = wilcoxon(ukf_true_diff, frac_true_diff, alternative='greater')
    print(f"Result of Wilcoxon test {res}")

    print(len(ukf_true_diff))

    # print(f"Mean of UKF's Hurst exponent is {np.mean(UKF_H_list)} and variance is {np.var(UKF_H_list)}")
    # print(f"Mean of Fractal UKF's exponent is  is {np.mean(frac_H_list)} and variance is {np.var(frac_H_list)}")
    # print(f"Number of entries is {len(UKF_H_list)}")



if __name__ == '__main__':
    main()
