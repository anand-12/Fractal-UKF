import numpy as np
from test_funcs.mackey_glass import mackey_glass


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