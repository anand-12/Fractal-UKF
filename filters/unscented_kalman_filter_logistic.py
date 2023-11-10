import numpy as np
from test_funcs.functions import logistic_map


def unscented_kalman_filter_logistic(y, r, Q, R, x_init, P_init):
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


