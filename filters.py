import numpy as np
from functions import generate_fBm, logistic_map, mackey_glass, tent_map

def unscented_kalman_filter_fbm(y, hurst_exponent, Q, R, x_init, P_init):

    n_states = len(x_init)

    x_hat = x_init
    P = P_init

    beta = 0.2
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
            x_pred += W_m[i] * generate_fBm(sigma_points[i], hurst_exponent=hurst_exponent)

        P_pred = np.zeros((n_states, n_states))
        for i in range(2 * n_states + 1):
            err = generate_fBm(sigma_points[i], hurst_exponent=hurst_exponent) - x_pred
            P_pred += W_c[i] * np.outer(err, err)

        P_pred += Q

        # Update step
        K = P_pred @ np.linalg.inv(P_pred + R)
        x_hat = x_pred + K @ (measurement - x_pred)
        P = P_pred - K @ P_pred

        # Store or use x_hat as your estimated state at this time step
        estimated_state.append(x_hat[0])

    return estimated_state

def unscented_kalman_filter_logistic(y, r, Q, R, x_init, P_init):
    n_states = len(x_init)
    x_hat = x_init
    P = P_init

    alpha = 0.001
    kappa = 0.1
    beta = 0.2
    lambda_ = alpha**2 * (n_states + kappa) - n_states
    W_m = np.zeros(2 * n_states + 1)
    W_c = np.zeros(2 * n_states + 1)

    W_m[0] = lambda_ / (n_states + lambda_)
    W_c[0] = lambda_ / (n_states + lambda_) + (1 - alpha**2 + beta)
    for i in range(1, 2 * n_states + 1):
        W_m[i] = 1 / (2 * (n_states + lambda_))
        W_c[i] = W_m[i]

    estimated_state = []

    for measurement in y:
        sigma_points = np.zeros((2 * n_states + 1, n_states))
        sigma_points[0] = x_hat
        for i in range(n_states):
            sqrt_term = np.sqrt((n_states + lambda_) * P[i, i])
            sigma_points[i + 1] = x_hat + sqrt_term
            sigma_points[n_states + i + 1] = x_hat - sqrt_term

        x_pred = np.zeros(n_states)
        for i in range(2 * n_states + 1):
            x_pred += W_m[i] * logistic_map(sigma_points[i], r)

        P_pred = np.zeros((n_states, n_states))
        for i in range(2 * n_states + 1):
            err = logistic_map(sigma_points[i], r) - x_pred
            P_pred += W_c[i] * np.outer(err, err)

        P_pred += Q

        K = P_pred @ np.linalg.inv(P_pred + R)
        x_hat = x_pred + K @ (measurement - x_pred)
        P = P_pred - K @ P_pred

        estimated_state.append(x_hat[0])

    return estimated_state



def unscented_kalman_filter_mackey(y, beta, gamma, tao, n, Q, R, x_init, P_init):
    n_states = len(x_init)
    x_hat = x_init
    P = P_init
    alpha = 0.001
    kappa = 0.1
    lambda_ = alpha**2 * (n_states + kappa) - n_states
    W_m = np.zeros(2 * n_states + 1)
    W_c = np.zeros(2 * n_states + 1)
    W_m[0] = lambda_ / (n_states + lambda_)
    W_c[0] = lambda_ / (n_states + lambda_) + (1 - alpha**2 + beta)
    for i in range(1, 2 * n_states + 1):
        W_m[i] = 1 / (2 * (n_states + lambda_))
        W_c[i] = W_m[i]
    estimated_state = []
    for measurement in y:
        sigma_points = np.zeros((2 * n_states + 1, n_states))
        sigma_points[0] = x_hat
        for i in range(n_states):
            sqrt_term = np.sqrt((n_states + lambda_) * P[i, i])
            sigma_points[i + 1] = x_hat + sqrt_term
            sigma_points[n_states + i + 1] = x_hat - sqrt_term
        x_pred = np.zeros(n_states)
        for i in range(2 * n_states + 1):
            x_pred += W_m[i] * mackey_glass(sigma_points[i], beta, gamma, tao, n)
        P_pred = np.zeros((n_states, n_states))
        for i in range(2 * n_states + 1):
            err = mackey_glass(sigma_points[i], beta, gamma, tao, n) - x_pred
            P_pred += W_c[i] * np.outer(err, err)
        P_pred += Q
        K = P_pred @ np.linalg.inv(P_pred + R)
        x_hat = x_pred + K @ (measurement - x_pred)
        P = P_pred - K @ P_pred
        estimated_state.append(x_hat[0])
    return estimated_state

def unscented_kalman_filter_tent_map(y, mu, Q, R, x_init, P_init):
    n_states = len(x_init)
    x_hat = x_init
    P = P_init

    alpha = 0.001
    kappa = 0.1
    beta = 0.2
    lambda_ = alpha**2 * (n_states + kappa) - n_states
    W_m = np.zeros(2 * n_states + 1)
    W_c = np.zeros(2 * n_states + 1)

    W_m[0] = lambda_ / (n_states + lambda_)
    W_c[0] = lambda_ / (n_states + lambda_) + (1 - alpha**2 + beta)
    for i in range(1, 2 * n_states + 1):
        W_m[i] = 1 / (2 * (n_states + lambda_))
        W_c[i] = W_m[i]

    estimated_state = []

    for measurement in y:
        sigma_points = np.zeros((2 * n_states + 1, n_states))
        sigma_points[0] = x_hat
        for i in range(n_states):
            sqrt_term = np.sqrt((n_states + lambda_) * P[i, i])
            sigma_points[i + 1] = x_hat + sqrt_term
            sigma_points[n_states + i + 1] = x_hat - sqrt_term

        x_pred = np.zeros(n_states)
        for i in range(2 * n_states + 1):
            x_pred += W_m[i] * tent_map(sigma_points[i], mu)

        P_pred = np.zeros((n_states, n_states))
        for i in range(2 * n_states + 1):
            err = tent_map(sigma_points[i], mu) - x_pred
            P_pred += W_c[i] * np.outer(err, err)

        P_pred += Q

        K = P_pred @ np.linalg.inv(P_pred + R)
        x_hat = x_pred + K @ (measurement - x_pred)
        P = P_pred - K @ P_pred

        estimated_state.append(x_hat[0])

    return estimated_state
